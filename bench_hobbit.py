"""
bench_hobbit.py — HOBBIT 吞吐量基准测试
========================================
在真实 Mixtral-8x7B 上测不同配置下的 tokens/s，与 llama.cpp 基准对比。

用法（服务器上）：
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b python bench_hobbit.py 2>&1 | tee ../logs/hobbit_bench_$(date +%Y%m%d_%H%M%S).log
"""

import sys, os, time

os.environ["HF_HUB_ENABLE_HF_XET"] = "0"

import torch
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
HOBBIT_CONFIG = {"T1": 0.6, "T2": 0.9}
PROMPT_LENGTHS = [32, 64, 128, 256, 512]
GEN_TOKENS = 128
N_WARMUP = 1
N_RUNS = 3


def make_hobbit_forward(layer_idx, stats):
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


def main():
    model_id = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    if not model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        local = False
    else:
        print(f"[BENCH] Using local model: {model_id}")
        local = True

    n_gpus = torch.cuda.device_count()
    print(f"[BENCH] GPUs: {n_gpus}")

    # 加载模型（和 server_hobbit.py 同样的策略）
    print(f"[BENCH] Loading model...")
    t0 = time.time()
    model = MixtralForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={i: "40GB" for i in range(n_gpus)}
        | ({"cpu": "200GB"} if n_gpus else {}),
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    print(f"[BENCH] Loaded in {time.time()-t0:.1f}s")

    # 缝合 HOBBIT
    print("[BENCH] Patching MoE layers...")
    for layer in model.model.layers:
        s = {"hit": 0, "miss": 0, "int4": 0, "skip": 0, "cache": {0, 1}}
        layer.mlp.forward = make_hobbit_forward(0, s).__get__(
            layer.mlp, type(layer.mlp)
        )
    print("[BENCH] Patch done")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 生成测试 prompt（用一段重复文本凑长度）
    base_text = "The HOBBIT system is a mixed precision expert offloading framework for fast MoE inference. "
    results = []

    for plen in PROMPT_LENGTHS:
        # 构建 prompt
        text = base_text * ((plen // 5) + 1)
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=plen)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        actual_len = inp["input_ids"].shape[1]
        print(f"\n[BENCH] pp{plen} (actual={actual_len} tokens):")

        # Warmup
        for _ in range(N_WARMUP):
            with torch.no_grad():
                _ = model(**inp)

        # Bench prompt processing
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_RUNS):
            with torch.no_grad():
                _ = model(**inp)
            torch.cuda.synchronize()
        dt = time.time() - t0
        pp_tps = actual_len * N_RUNS / dt
        print(f"  pp: {pp_tps:.1f} t/s ({dt/N_RUNS:.2f}s avg)")

        # Bench token generation（用 generate 测 tg）
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inp,
                max_new_tokens=GEN_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        new_tokens = gen_out.shape[1] - actual_len
        tg_tps = new_tokens / dt
        print(f"  tg: {tg_tps:.1f} t/s ({new_tokens} tokens in {dt:.2f}s)")

        results.append((f"pp{plen}", pp_tps, tg_tps))

    # 汇总
    print(f"\n{'='*70}")
    print(f"{'Test':>10} {'pp t/s':>10} {'tg t/s':>10}")
    print(f"{'-'*30}")
    for name, pp, tg in results:
        print(f"{name:>10} {pp:>10.1f} {tg:>10.1f}")

    print(f"\n[BENCH] Compare with llama.cpp baseline (Q4_K_M, 2xL20):")
    print(f"  llama.cpp ngl=0:  pp512=6.9  tg128=6.9")
    print(f"  llama.cpp ngl=20: pp512=17.3 tg128=16.2")
    print(f"  llama.cpp ngl=32: pp512=124.9 tg128=74.0")
    print(
        f"[BENCH] HOBBIT target: close to ngl=32 by eliminating transfer stalls via INT4 fallback."
    )


if __name__ == "__main__":
    main()
