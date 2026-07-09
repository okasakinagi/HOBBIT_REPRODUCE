"""
download_gsm8k.py — 下载 GSM8K 数据集并保存为 JSON
====================================================
从 HuggingFace 下载 GSM8K 测试集，保存到 data/gsm8k_test.json

用法：
    # 方式 1: 用 datasets 库（推荐）
    pip install datasets
    python tools/download_gsm8k.py

    # 方式 2: 通过 hf-mirror 下载 parquet + pyarrow 转换
    # （需安装 pyarrow: pip install pyarrow pandas）
    python tools/download_gsm8k.py --parquet

    # 方式 3: 直接从 HF 下载 JSONL（备选）
    python tools/download_gsm8k.py --fallback
"""

import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_PATH = os.path.join(DATA_DIR, "gsm8k_test.json")


def download_with_datasets():
    """使用 datasets 库下载 GSM8K（主方案）"""
    print("[download_gsm8k] Using `datasets` library...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[download_gsm8k] `datasets` not installed.")
        return False

    ds = load_dataset("gsm8k", "main", split="test")
    records = []
    for item in ds:
        answer_raw = item["answer"]
        final_ans = answer_raw.split("####")[-1].strip()
        records.append({
            "question": item["question"],
            "answer": final_ans,
            "answer_raw": answer_raw,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[download_gsm8k] Saved {len(records)} questions to {OUT_PATH}")
    return True


def download_parquet():
    """通过 hf-mirror 下载 parquet 格式（国内网络友好），用 pyarrow 转换"""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    parquet_url = f"{hf_endpoint}/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
    print(f"[download_gsm8k] Downloading parquet from {parquet_url}...")

    # 先用 curl 下载
    import subprocess
    parquet_path = os.path.join(DATA_DIR, "test.parquet")
    result = subprocess.run(
        ["curl", "-L", "-o", parquet_path, parquet_url],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(parquet_path) or os.path.getsize(parquet_path) < 1000:
        print(f"[download_gsm8k] curl failed: {result.stderr}")
        return False

    print(f"[download_gsm8k] Parquet downloaded ({os.path.getsize(parquet_path)//1024} KB)")

    # 用 pyarrow 转换
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[download_gsm8k] pyarrow not installed. Install: pip install pyarrow pandas")
        return False

    table = pq.read_table(parquet_path)
    records = []
    for i in range(table.num_rows):
        q = table.column("question")[i].as_py()
        a = table.column("answer")[i].as_py()
        final_ans = a.split("####")[-1].strip()
        records.append({"question": q, "answer": final_ans, "answer_raw": a})

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    os.remove(parquet_path)
    print(f"[download_gsm8k] Saved {len(records)} questions to {OUT_PATH}")
    return True


def download_fallback():
    """直接从 HuggingFace 下载原始 JSONL（备选方案）"""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = f"{hf_endpoint}/datasets/gsm8k/resolve/main/gsm8k/test.jsonl"
    print(f"[download_gsm8k] Fallback: downloading JSONL from {url}...")

    import urllib.request
    try:
        urllib.request.urlretrieve(url, os.path.join(DATA_DIR, "test.jsonl"))
    except Exception as e:
        print(f"[download_gsm8k] Failed: {e}")
        return False

    records = []
    jsonl_path = os.path.join(DATA_DIR, "test.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            answer_raw = item["answer"]
            final_ans = answer_raw.split("####")[-1].strip()
            records.append({
                "question": item["question"],
                "answer": final_ans,
                "answer_raw": answer_raw,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    os.remove(jsonl_path)
    print(f"[download_gsm8k] Saved {len(records)} questions to {OUT_PATH}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download GSM8K dataset")
    parser.add_argument("--fallback", action="store_true",
                       help="Download raw JSONL from HF (fallback)")
    parser.add_argument("--parquet", action="store_true",
                       help="Download parquet via hf-mirror + convert with pyarrow")
    args = parser.parse_args()

    if args.parquet:
        success = download_parquet()
    elif args.fallback:
        success = download_fallback()
    else:
        success = download_with_datasets()
        if not success:
            print("[download_gsm8k] Trying parquet download (hf-mirror)...")
            success = download_parquet()

    if success:
        print(f"[download_gsm8k] Done. {OUT_PATH}")
    else:
        print("[download_gsm8k] FAILED. Manual steps:")
        print("  1. curl -L -o data/test.parquet https://hf-mirror.com/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
        print("  2. pip install pyarrow pandas")
        print("  3. python tools/convert_gsm8k.py")


if __name__ == "__main__":
    main()
