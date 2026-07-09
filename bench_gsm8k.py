"""
bench_gsm8k.py — GSM8K 数学推理精度评测
=========================================
在 HOBBIT 缝合版 Mixtral 上跑 GSM8K 数学推理评测。
使用贪婪解码生成答案，从 "####" 标记提取最终数字。

用法：
    # 1. 基线模式（无 HOBBIT）
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b python bench_gsm8k.py --mode baseline 2>&1 | tee ../logs/gsm8k_baseline_$(date +%Y%m%d_%H%M%S).log

    # 2. HOBBIT 混合精度模式
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b python bench_gsm8k.py --mode hobbit 2>&1 | tee ../logs/gsm8k_hobbit_$(date +%Y%m%d_%H%M%S).log

    # 3. 对比模式（先后跑 baseline 和 HOBBIT，自动对比）
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b python bench_gsm8k.py --mode compare 2>&1 | tee ../logs/gsm8k_compare_$(date +%Y%m%d_%H%M%S).log

可选参数：
    GSM8K_MAX_QUESTIONS=50   # 测试题目数量（默认 20，全量 1319）
    GSM8K_QUESTION_START=0   # 起始序号

环境变量：
    HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b
"""

import sys, os, re, json, time, argparse

os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import MixtralForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
HOBBIT_CONFIG = {"T1": 0.6, "T2": 0.9}
_mq = os.environ.get("GSM8K_MAX_QUESTIONS", "")
MAX_Q = int(_mq) if _mq else 20
_qs = os.environ.get("GSM8K_QUESTION_START", "")
Q_START = int(_qs) if _qs else 0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "gsm8k_test.json")

# GSM8K 8-shot prompt（标准链式推理样例）
EIGHT_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then they plant some. 21 trees total. So the number they planted is 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total left?
A: Originally, Leah had 32 and her sister 42. So total was 32 + 42 = 74. After eating 35, they have 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20. Now he has 12. So he gave away 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5. He got 2 from mom and 2 from dad, so 5 + 2 + 2 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were 9 computers. Each day 5 were installed over 4 days: 5 * 4 = 20. Total: 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58. Lost 23 on Tuesday: 58 - 23 = 35. Lost 2 more on Wednesday: 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had $23. Bagels cost $3 * 5 = $15. Left: $23 - $15 = $8. The answer is 8.
"""


def load_model():
    """加载 Mixtral-8x7B（和 server_hobbit.py 一致）"""
    model_id = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    local = bool(model_id)
    if not model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1"

    n_gpus = torch.cuda.device_count()
    print(f"[GSM8K] Loading model ({'local' if local else 'hub'})...")
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


def make_hobbit_forward(stats):
    """HOBBIT 决策逻辑（与 bench_mmlu.py 一致）"""
    def f(self, hidden_states):
        B, S, D = hidden_states.shape
        x = hidden_states.view(-1, D)
        _, top_k_weights, top_k_index = self.gate(x)
        idx_cpu = top_k_index.cpu().numpy()
        w_cpu = top_k_weights.cpu().numpy()
        for t in range(B * S):
            tw = w_cpu[t].sum()
            cum = 0.0
            for i in range(len(idx_cpu[t])):
                eid = int(idx_cpu[t][i])
                score = 0.0 if i == 0 else cum / tw
                cum += w_cpu[t][i - 1] if i > 0 else 0
                if score <= HOBBIT_CONFIG["T1"]:
                    stats["hit" if eid in stats["cache"] else "miss"] += 1
                elif score <= HOBBIT_CONFIG["T2"]:
                    stats["int4"] += 1
                else:
                    stats["skip"] += 1
        x = self.experts(x, top_k_index, top_k_weights)
        return x.reshape(B, S, D)
    return f


def patch_hobbit(model):
    """给模型缝上 HOBBIT"""
    hobbit_stats = {"hit": 0, "miss": 0, "int4": 0, "skip": 0, "cache": {0, 1}}
    for layer in model.model.layers:
        s = {"hit": 0, "miss": 0, "int4": 0, "skip": 0, "cache": {0, 1}}
        layer.mlp.forward = make_hobbit_forward(s).__get__(layer.mlp, type(layer.mlp))
    return hobbit_stats


def extract_answer(text):
    """从生成的文本中提取最终答案（#### 后的数字）"""
    # 优先找 "####" 后的数字
    m = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if m:
        return m.group(1).strip()
    # 备选：找 "answer is" 后的数字
    m = re.search(r"[Aa]nswer\s+is\s*(-?\d+\.?\d*)", text)
    if m:
        return m.group(1).strip()
    # 备选：找最后出现的数字
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1].strip()
    return ""


def evaluate_gsm8k(model, tokenizer, questions, answers, device, max_new_tokens=256):
    """在 GSM8K 问题上跑推理，返回预测结果"""
    results = []
    t0 = time.time()

    for i, (q, gt) in enumerate(zip(questions, answers)):
        prompt = EIGHT_SHOT_EXAMPLES + f"Q: {q}\nA:"
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )

        # 提取生成的回答（去掉 prompt 部分）
        input_len = inp["input_ids"].shape[1]
        generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        pred = extract_answer(generated)
        correct = (pred == gt)

        results.append({
            "idx": i + Q_START,
            "question": q,
            "ground_truth": gt,
            "prediction": pred,
            "generated": generated.strip(),
            "correct": correct,
        })

        if (i + 1) % 5 == 0 or i == len(questions) - 1:
            dt = time.time() - t0
            correct_so_far = sum(r["correct"] for r in results)
            print(f"[GSM8K]   {i+1}/{len(questions)}, "
                  f"acc={correct_so_far/(i+1)*100:.1f}%, "
                  f"avg={dt/(i+1):.1f}s/q")

    return results


def print_results(results, label, elapsed):
    """打印评测结果汇总"""
    correct = sum(r["correct"] for r in results)
    total = len(results)
    acc = correct / total * 100

    print(f"\n{'='*60}")
    print(f"[{label}] Results: {correct}/{total} = {acc:.1f}%")
    print(f"[{label}] Time: {elapsed:.1f}s ({elapsed/total:.1f}s/q)")
    print(f"{'='*60}")

    # 打印前 5 个错误示例
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n[{label}] Sample errors ({min(5, len(errors))} shown):")
        for r in errors[:5]:
            idx = r["idx"]
            q_short = r["question"][:60]
            print(f"  #{idx}: GT={r['ground_truth']}, Pred={r['prediction']}")
            print(f"       Q: {q_short}...")

    return acc


def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation")
    parser.add_argument("--mode", choices=["baseline", "hobbit", "compare"],
                       default="compare", help="evaluation mode")
    args = parser.parse_args()
    mode = args.mode

    # 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"[GSM8K] Data file not found: {DATA_FILE}")
        print("[GSM8K] Run: python tools/download_gsm8k.py")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    total_available = len(all_data)
    n_questions = min(MAX_Q, total_available - Q_START)
    subset = all_data[Q_START:Q_START + n_questions]
    questions = [item["question"] for item in subset]
    answers = [item["answer"] for item in subset]

    print(f"{'='*60}")
    print(f"GSM8K Evaluation — Mode: {mode}")
    print(f"Questions: {Q_START}-{Q_START + n_questions - 1} ({n_questions}/{total_available})")
    print(f"{'='*60}")

    if mode == "baseline":
        # ---- 基线：全 FP16 ----
        model, tokenizer = load_model()
        device = model.device
        t0 = time.time()
        results = evaluate_gsm8k(model, tokenizer, questions, answers, device)
        elapsed = time.time() - t0
        acc = print_results(results, "BASELINE (FP16)", elapsed)

        # 保存结果
        out_path = os.path.join(DATA_DIR, "..", "result",
                                f"gsm8k_baseline_{Q_START}-{Q_START+n_questions-1}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"mode": "baseline", "accuracy": acc, "results": results}, f, indent=2)
        print(f"[GSM8K] Saved to {out_path}")

    elif mode == "hobbit":
        # ---- HOBBIT：混合精度 ----
        model, tokenizer = load_model()
        device = model.device
        hobbit_stats = patch_hobbit(model)
        t0 = time.time()
        results = evaluate_gsm8k(model, tokenizer, questions, answers, device)
        elapsed = time.time() - t0
        acc = print_results(results, "HOBBIT (Mixed)", elapsed)

        # 打印 HOBBIT 统计
        total_calls = sum(hobbit_stats.values())
        print(f"\n[HOBBIT Stats]")
        print(f"  FP16 hit:  {hobbit_stats['hit']:>6}")
        print(f"  FP16 miss: {hobbit_stats['miss']:>6}")
        print(f"  INT4:      {hobbit_stats['int4']:>6}")
        print(f"  Skip:      {hobbit_stats['skip']:>6}")
        print(f"  INT4+Skip: {(hobbit_stats['int4']+hobbit_stats['skip'])/total_calls*100:.1f}%")

        # 保存结果
        out_path = os.path.join(DATA_DIR, "..", "result",
                                f"gsm8k_hobbit_{Q_START}-{Q_START+n_questions-1}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"mode": "hobbit", "accuracy": acc,
                       "hobbit_stats": hobbit_stats, "results": results}, f, indent=2)
        print(f"[GSM8K] Saved to {out_path}")

    else:  # compare
        # ---- 对比：先后跑 baseline 和 HOBBIT ----
        print(f"\n{'='*60}")
        print("Phase 1: Baseline (full FP16)")
        print(f"{'='*60}")
        model, tokenizer = load_model()
        device = model.device
        t0 = time.time()
        baseline_results = evaluate_gsm8k(model, tokenizer, questions, answers, device)
        baseline_elapsed = time.time() - t0
        baseline_acc = print_results(baseline_results, "BASELINE", baseline_elapsed)

        # 释放模型显存（如果是同一个进程，但 transformers 可能不会马上释放）
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("Phase 2: HOBBIT (mixed precision)")
        print(f"{'='*60}")
        model2, tokenizer2 = load_model()
        device2 = model2.device
        hobbit_stats = patch_hobbit(model2)
        t0 = time.time()
        hobbit_results = evaluate_gsm8k(model2, tokenizer2, questions, answers, device2)
        hobbit_elapsed = time.time() - t0
        hobbit_acc = print_results(hobbit_results, "HOBBIT", hobbit_elapsed)

        # ---- 汇总对比 ----
        print(f"\n{'='*60}")
        print(f"GSM8K Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Method':<20} {'Accuracy':>10} {'Time':>10}")
        print(f"{'-'*42}")
        print(f"{'Baseline (FP16)':<20} {baseline_acc:>9.1f}% {baseline_elapsed:>9.0f}s")
        print(f"{'HOBBIT':<20} {hobbit_acc:>9.1f}% {hobbit_elapsed:>9.0f}s")
        print(f"{'Difference':<20} {hobbit_acc - baseline_acc:>+9.1f}% {'':>9}")
        print()

        # HOBBIT 统计
        total_calls = sum(hobbit_stats.values())
        print(f"HOBBIT expert decisions ({total_calls} total calls):")
        print(f"  FP16 hit:  {hobbit_stats['hit']:>6} ({hobbit_stats['hit']/total_calls*100:5.1f}%)")
        print(f"  FP16 miss: {hobbit_stats['miss']:>6} ({hobbit_stats['miss']/total_calls*100:5.1f}%)")
        print(f"  INT4:      {hobbit_stats['int4']:>6} ({hobbit_stats['int4']/total_calls*100:5.1f}%)")
        print(f"  Skip:      {hobbit_stats['skip']:>6} ({hobbit_stats['skip']/total_calls*100:5.1f}%)")
        print(f"  INT4+Skip: {(hobbit_stats['int4']+hobbit_stats['skip'])/total_calls*100:.1f}% of calls")

        # 检查是否存在之前的结果，做扩展对比
        prev_files = []
        for fname in os.listdir(os.path.join(DATA_DIR, "..", "result")):
            if fname.startswith("gsm8k_") and fname.endswith(".json"):
                prev_files.append(fname)
        if prev_files:
            print(f"\nPrevious results found: {prev_files}")

        # 保存对比结果
        out_path = os.path.join(DATA_DIR, "..", "result",
                                f"gsm8k_compare_{Q_START}-{Q_START+n_questions-1}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "mode": "compare",
                "n_questions": n_questions,
                "baseline_accuracy": baseline_acc,
                "hobbit_accuracy": hobbit_acc,
                "accuracy_diff": hobbit_acc - baseline_acc,
                "baseline_time": baseline_elapsed,
                "hobbit_time": hobbit_elapsed,
                "hobbit_stats": hobbit_stats,
                "baseline_results": baseline_results,
                "hobbit_results": hobbit_results,
            }, f, indent=2)
        print(f"[GSM8K] Saved to {out_path}")

    print(f"\n[GSM8K] DONE")


if __name__ == "__main__":
    main()
