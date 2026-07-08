"""
bench_mmlu.py — MMLU 精度评测
===============================
在 HOBBIT 缝合版 Mixtral 上跑 MMLU 选择题评测。
利用选择题特性：单次前向传播后比较 A/B/C/D 的概率，无需逐 token 生成。

用法（服务器上）：
    HF_ENDPOINT=https://hf-mirror.com \
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b \
    python bench_mmlu.py 2>&1 | tee ../logs/mmlu_$(date +%Y%m%d_%H%M%S).log

可选参数：
    MMLU_SUBJECTS=high_school_physics,high_school_mathematics,professional_law
    MMLU_MAX_QUESTIONS=50  # 每个学科最多做多少题
"""

import sys, os, time
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import MixtralForCausalLM, AutoTokenizer
from datasets import load_dataset

# ============================================================
# 配置
# ============================================================
HOBBIT_CONFIG = {"T1": 0.6, "T2": 0.9}
SUBJECTS = os.environ.get("MMLU_SUBJECTS", "high_school_physics,high_school_mathematics,professional_law").split(",")
MAX_Q = int(os.environ.get("MMLU_MAX_QUESTIONS", "0") or "9999")


def make_hobbit_forward(stats):
    """和 server_hobbit.py 完全一致的 HOBBIT 猴子补丁"""
    def f(self, hidden_states):
        B, S, D = hidden_states.shape
        x = hidden_states.view(-1, D)
        _, top_k_weights, top_k_index = self.gate(x)
        idx_cpu = top_k_index.cpu().numpy()
        w_cpu = top_k_weights.cpu().numpy()
        for t in range(B * S):
            tw = w_cpu[t].sum(); cum = 0.0
            for i in range(len(idx_cpu[t])):
                eid = int(idx_cpu[t][i])
                score = 0.0 if i == 0 else cum / tw
                cum += w_cpu[t][i-1] if i > 0 else 0
                if score <= HOBBIT_CONFIG["T1"]:
                    stats["hit" if eid in stats["cache"] else "miss"] += 1
                elif score <= HOBBIT_CONFIG["T2"]:
                    stats["int4"] += 1
                else:
                    stats["skip"] += 1
        x = self.experts(x, top_k_index, top_k_weights)
        return x.reshape(B, S, D)
    return f


def load_model_with_hobbit():
    model_id = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    local = bool(model_id)
    if not model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1"

    n_gpus = torch.cuda.device_count()
    print(f"[MMLU] Loading model...")
    model = MixtralForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=({i: "40GB" for i in range(n_gpus)} | ({"cpu": "200GB"} if n_gpus else {})),
        low_cpu_mem_usage=True,
        local_files_only=local,
    )

    # 缝合 HOBBIT
    for layer in model.model.layers:
        s = {"hit": 0, "miss": 0, "int4": 0, "skip": 0, "cache": {0, 1}}
        layer.mlp.forward = make_hobbit_forward(s).__get__(layer.mlp, type(layer.mlp))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_mmlu_prompt(question, choices, subject=""):
    """MMLU 标准 4 选 1 格式"""
    letters = ["A", "B", "C", "D"]
    prompt = f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
    prompt += f"Question: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{letters[i]}. {c}\n"
    prompt += "Answer:"
    return prompt


def evaluate_subject(model, tokenizer, subject, device):
    print(f"\n[MMLU] Subject: {subject}")
    ds = load_dataset("cais/mmlu", subject, split="test")
    total = min(len(ds), MAX_Q)
    if total == 0:
        print(f"[MMLU]   No questions found, skipping")
        return 0.0
    print(f"[MMLU]   Questions: {total}")

    correct = 0
    letter_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"}
    
    t0 = time.time()
    for i in range(total):
        item = ds[i]
        prompt = format_mmlu_prompt(item["question"], item["choices"], subject)
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inp)

        # 取最后一个 token 的 logits，比较 A/B/C/D 概率
        last_logits = out.logits[0, -1, :]
        probs = {l: last_logits[tid].item() for l, tid in letter_ids.items()}
        pred = max(probs, key=probs.get)
        answer = {0: "A", 1: "B", 2: "C", 3: "D"}[item["answer"]]

        if pred == answer:
            correct += 1

        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"[MMLU]   {i+1}/{total}, acc={correct/(i+1)*100:.1f}%, {dt/(i+1):.1f}s/q")

    acc = correct / total * 100
    dt = time.time() - t0
    print(f"[MMLU]   Done: {correct}/{total} = {acc:.1f}%, {dt/total:.1f}s/q avg")
    return acc


def main():
    model, tokenizer = load_model_with_hobbit()
    device = model.device

    print(f"\n{'='*60}")
    print(f"MMLU Evaluation — HOBBIT Mixed Precision")
    print(f"Subjects: {SUBJECTS}")
    print(f"{'='*60}")

    results = {}
    t0 = time.time()
    for subj in SUBJECTS:
        results[subj] = evaluate_subject(model, tokenizer, subj, device)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'Subject':<35} {'Accuracy':>10}")
    print(f"{'-'*47}")
    avg_acc = 0
    for subj, acc in results.items():
        print(f"{subj:<35} {acc:>9.1f}%")
        avg_acc += acc
    avg_acc /= len(results)
    print(f"{'-'*47}")
    print(f"{'Average':<35} {avg_acc:>9.1f}%")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"[MMLU] DONE")


if __name__ == "__main__":
    main()
