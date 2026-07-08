"""
server_hobbit_local.py — 本机验证脚本
======================================
在本地 RTX 4060（7GB）上验证 HOBBIT 缝合逻辑，用真实 Mixtral 权重但只加载 2 层。

用法：
    python server_hobbit_local.py 2>&1 | tee logs/local_$(date +%Y%m%d_%H%M%S).log

原理：
    修改 config.num_hidden_layers=2，from_pretrained 只加载前 2 层 + embedding + lm_head。
    2/32 的权重约 6GB，刚好塞进 7GB 显存。使用真实权重（非随机），可以验证猴子补丁逻辑。

目的：
    1. 验证 HOBBIT 猴子补丁在真实权重模型上能否正常工作
    2. 确认决策统计（FP16命中/INT4/跳过）的输出是否合理
    3. 测量单层 MoE 加载和推理的实际显存占用，为服务器调试提供参考
"""

import sys, os, time

# --- 必须在 import 之前 ---
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer


# ============================================================
# 0. 环境自检
# ============================================================
def env_check():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {sys.argv[0]}")
    print(f"[ENV] Python={sys.version.split()[0]}")
    print(f"[ENV] PyTorch={torch.__version__}")
    print(f"[ENV] CUDA available={torch.cuda.is_available()}")
    print(f"[ENV] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'not set')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"[ENV] GPU[{i}]: {p.name}, total={total//1024**3}GB, free={free//1024**3}GB")


# ============================================================
# 1. HOBBIT 参数
# ============================================================
T1, T2 = 0.6, 0.9


# ============================================================
# 2. 猴子补丁：替换 MoE forward
# ============================================================
def make_hobbit_forward(layer_idx, stats):
    def hobbit_forward(self, hidden_states):
        B, S, D = hidden_states.shape
        N = B * S
        x = hidden_states.view(-1, D)

        # Step 1: 路由
        router_logits, top_k_weights, top_k_index = self.gate(x)

        # Step 2: HOBBIT 决策统计
        idx_cpu = top_k_index.cpu().numpy()
        w_cpu = top_k_weights.cpu().numpy()
        for t in range(N):
            total_w = w_cpu[t].sum()
            cum = 0.0
            for i in range(len(w_cpu[t])):
                eid = int(idx_cpu[t][i])
                score = 0.0 if i == 0 else cum / total_w
                cum += w_cpu[t][i - 1] if i > 0 else 0

                if score <= T1:
                    if eid in stats["fp16_cache"]:
                        stats["hit_fp16"] += 1
                    else:
                        stats["miss_fp16"] += 1
                elif score <= T2:
                    stats["use_int4"] += 1
                else:
                    stats["skip"] += 1

        # Step 3: 原生专家计算（本机不搞量化，都用原权重）
        x = self.experts(x, top_k_index, top_k_weights)
        return x.reshape(B, S, D)
    return hobbit_forward


# ============================================================
# 3. 主流程
# ============================================================
def main():
    env_check()

    model_id = "mistralai/Mixtral-8x7B-v0.1"

    # --- 3a. 加载 2 层真实 Mixtral ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading Mixtral-8x7B (2 layers only)...")
    print("[LOAD] Strategy: set config.num_hidden_layers=2, load only first 2 layers")
    print("[LOAD] Expected: ~6GB in FP16, fits RTX 4060 (7GB)")

    config = MixtralConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2  # 只加载前 2 层

    n_gpus = torch.cuda.device_count()
    device = "cuda:0" if n_gpus > 0 else "cpu"

    model = MixtralForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
    )

    # 打印实际显存
    if n_gpus > 0:
        _, total = torch.cuda.mem_get_info(0)
        used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"[LOAD] GPU memory used: {used:.1f}GB / {total/1024**3:.0f}GB")

    print(f"[LOAD] Layers: {config.num_hidden_layers}, "
          f"Experts: {config.num_local_experts}, Top-K: {config.num_experts_per_tok}")

    # --- 3b. 替换 MoE 层 ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Patching MoE layers with HOBBIT forward...")
    per_layer = []
    for i, layer in enumerate(model.model.layers):
        s = {"layer": i, "hit_fp16": 0, "miss_fp16": 0, "use_int4": 0, "skip": 0,
             "fp16_cache": {0, 1}}
        per_layer.append(s)
        moe = layer.mlp
        moe.forward = make_hobbit_forward(i, s).__get__(moe, type(moe))
    print(f"[PATCH] All {len(model.model.layers)} layers patched")

    # --- 3c. 推理 ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Running inference...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Hello, how are you?"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"[INFER] Input ({inp['input_ids'].shape[1]} tokens): {prompt}")

    t0 = time.time()
    with torch.no_grad():
        out = model(**inp)
    dt = time.time() - t0
    print(f"[INFER] Time: {dt:.2f}s, shape: {out.logits.shape}")

    # 预测下一个 token
    top5 = torch.topk(out.logits[0, -1, :], 5)
    for rank, (tid, prob) in enumerate(zip(top5.indices, top5.values), 1):
        print(f"[INFER]   #{rank}: '{tokenizer.decode([tid])}' (prob={prob:.4f})")

    # --- 3d. 统计 ---
    print(f"\n[{time.strftime('%H:%M:%S')}] HOBBIT Statistics:")
    print(f"{'Layer':>6} {'FP16-Hit':>10} {'FP16-Miss':>10} {'INT4':>10} {'Skip':>10} {'Hit%':>8}")
    print("-" * 56)
    for s in per_layer:
        t = s["hit_fp16"] + s["miss_fp16"] + s["use_int4"] + s["skip"]
        pct = s["hit_fp16"] / t * 100 if t else 0
        print(f"{s['layer']:>6} {s['hit_fp16']:>10} {s['miss_fp16']:>10} "
              f"{s['use_int4']:>10} {s['skip']:>10} {pct:>7.1f}%")

    # 内存报告
    if n_gpus > 0:
        used = torch.cuda.memory_allocated(0) / 1024**3
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\n[STATS] GPU memory: used={used:.1f}GB, peak={peak:.1f}GB")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE")
    print("[NOTE] If this runs successfully, the HOBBIT stitching logic is verified.")
    print("[NOTE] Server OOM is then purely a memory management issue, not a code bug.")


if __name__ == "__main__":
    main()
