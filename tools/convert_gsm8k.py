"""Convert GSM8K parquet to JSON"""
import json, os
import pyarrow.parquet as pq

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
parquet_path = os.path.join(data_dir, "test.parquet")
out_path = os.path.join(data_dir, "gsm8k_test.json")

table = pq.read_table(parquet_path)
records = []
for i in range(table.num_rows):
    q = table.column("question")[i].as_py()
    a = table.column("answer")[i].as_py()
    final_ans = a.split("####")[-1].strip()
    records.append({"question": q, "answer": final_ans, "answer_raw": a})

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

os.remove(parquet_path)
sample_q = records[0]["question"][:80]
print(f"Done. {len(records)} questions saved to {out_path}")
print(f"Sample: {sample_q}...")
