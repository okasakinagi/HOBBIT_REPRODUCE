"""
re_eval_gsm8k.py — 用修复后的 extract_answer 重新评估已有结果
==============================================================
用法：
    python tools/re_eval_gsm8k.py ../logs/gsm8k_baseline_0-19.json
    python tools/re_eval_gsm8k.py ../logs/gsm8k_hobbit_0-19.json

不需要重新跑模型，只是修正字符串比较逻辑。
"""

import sys, os, json, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_answer_fixed(text):
    """修复版 extract_answer：去掉末尾小数点"""
    m = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if m:
        ans = m.group(1).strip()
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    m = re.search(r"[Aa]nswer\s+is\s*(-?\d+\.?\d*)", text)
    if m:
        ans = m.group(1).strip()
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        ans = nums[-1].strip()
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/re_eval_gsm8k.py <result.json>")
        sys.exit(1)

    in_path = sys.argv[1]
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data.get("baseline_results", []))
    if not results:
        results = data.get("hobbit_results", [])

    corrected = 0
    for r in results:
        old_pred = r["prediction"]
        generated = r.get("generated", "")
        new_pred = extract_answer_fixed(generated)
        r["prediction"] = new_pred
        old_correct = r["correct"]
        r["correct"] = (new_pred == r["ground_truth"])
        if old_correct != r["correct"]:
            corrected += 1
            print(f"  #{r['idx']}: '{old_pred}' -> '{new_pred}' (GT={r['ground_truth']}) "
                  f"{'FIXED' if r['correct'] else 'STILL WRONG'}")

    correct = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Corrected: {corrected}/{total}")
    print(f"New accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    # 更新顶层 accuracy
    if "accuracy" in data:
        data["accuracy"] = correct / total * 100
    if "results" in data:
        data["results"] = results

    out_path = in_path.replace(".json", "_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
