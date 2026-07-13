"""
bench_mmlu.py — MMLU 精度评测
===============================
在 HOBBIT 缝合版 Mixtral 上跑 MMLU 选择题评测。
支持真实 bitsandbytes NF4 量化 + Skip，与 bench_gsm8k.py 共享 HOBBIT 核心。

用法（服务器 nohup 推荐）：
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b \
        nohup python bench_mmlu.py --mode baseline > ../logs/mmlu_bl_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    LOCAL_MODEL_PATH=~/models/mixtral-8x7b \
        nohup python bench_mmlu.py --mode hobbit > ../logs/mmlu_hb_$(date +%Y%m%d_%H%M%S).log 2>&1 &

可选参数：
    --mode baseline|hobbit  评测模式（必选）
    MMLU_SUBJECTS=physics,math,law  学科（逗号分隔）
    MMLU_MAX_QUESTIONS=50  每个学科最多多少题（默认全部）
"""

import sys, os, time, argparse

os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import MixtralForCausalLM, AutoTokenizer

# 从 bench_gsm8k.py 导入 HOBBIT 核心
from bench_gsm8k import (
    HOBBIT_CONFIG,
    quantize_weight_to_int4,
    _patch_experts_forward,
    _load_meta_int4,
    patch_hobbit,
)

SUBJECTS = os.environ.get(
    "MMLU_SUBJECTS", "high_school_physics,high_school_mathematics,professional_law"
).split(",")
_mq = os.environ.get("MMLU_MAX_QUESTIONS", "")
MAX_Q = int(_mq) if _mq else 9999


def load_model():
    """加载 Mixtral-8x7B（和 bench_gsm8k.py 一致）"""
    model_id = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    local = bool(model_id)
    if not model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1"

    n_gpus = torch.cuda.device_count()
    print(f"[MMLU] Loading model ({'local' if local else 'hub'})...")
    model = MixtralForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=(
            {i: "40GB" for i in range(n_gpus)} | ({"cpu": "200GB"} if n_gpus else {})
        ),
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
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
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "data", f"mmlu_{subject}.json")
    if not os.path.exists(json_path):
        print(f"[MMLU]   File not found: {json_path}")
        return 0.0
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = min(len(data), MAX_Q)
    if total == 0:
        print(f"[MMLU]   No questions found, skipping")
        return 0.0
    print(f"[MMLU]   Questions: {total}")

    correct = 0
    letter_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"}

    t0 = time.time()
    for i in range(total):
        item = data[i]
        prompt = format_mmlu_prompt(item["question"], item["choices"], subject)
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inp)

        last_logits = out.logits[0, -1, :]
        probs = {l: last_logits[tid].item() for l, tid in letter_ids.items()}
        pred = max(probs, key=probs.get)
        answer = {0: "A", 1: "B", 2: "C", 3: "D"}[item["answer"]]

        if pred == answer:
            correct += 1

        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(
                f"[MMLU]   {i+1}/{total}, acc={correct/(i+1)*100:.1f}%, {dt/(i+1):.1f}s/q"
            )

    acc = correct / total * 100
    dt = time.time() - t0
    print(f"[MMLU]   Done: {correct}/{total} = {acc:.1f}%, {dt/total:.1f}s/q avg")
    return acc


def main():
    parser = argparse.ArgumentParser(description="MMLU Evaluation")
    parser.add_argument(
        "--mode", choices=["baseline", "hobbit"], required=True,
        help="baseline (全FP16) 或 hobbit (混合精度)"
    )
    args = parser.parse_args()

    model, tokenizer = load_model()
    device = model.device

    hobbit_stats = None
    if args.mode == "hobbit":
        hobbit_stats = patch_hobbit(model)

    print(f"\n{'='*60}")
    print(f"MMLU Evaluation — Mode: {args.mode.upper()}")
    print(f"Subjects: {SUBJECTS}")
    print(f"{'='*60}")

    results = {}
    t0 = time.time()
    for subj in SUBJECTS:
        results[subj] = evaluate_subject(model, tokenizer, subj, device)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Summary ({args.mode.upper()}):")
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

    if hobbit_stats:
        agg = hobbit_stats.aggregate()
        total_calls = sum(agg[k] for k in agg if k != "cache")
        print(f"\n[HOBBIT Stats]")
        print(f"  FP16 hit:  {agg['hit']:>6} ({agg['hit']/total_calls*100:5.1f}%)")
        print(f"  FP16 miss: {agg['miss']:>6} ({agg['miss']/total_calls*100:5.1f}%)")
        print(f"  INT4:      {agg['int4']:>6} ({agg['int4']/total_calls*100:5.1f}%)")
        print(f"  Skip:      {agg['skip']:>6} ({agg['skip']/total_calls*100:5.1f}%)")
        print(f"  INT4+Skip: {(agg['int4']+agg['skip'])/total_calls*100:.1f}% of calls")

    print(f"\n[MMLU] DONE")


if __name__ == "__main__":
    main()
