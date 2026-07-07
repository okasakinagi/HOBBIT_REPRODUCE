"""
server_hobbit.py — 服务器部署脚本
====================================
阶段3：真实 Mixtral-8x7B + HOBBIT 混合精度推理

用法（服务器上）：
    # 基本运行（输出到终端+文件，logs 在项目上级目录避免 git pull 冲突）
    python server_hobbit.py 2>&1 | tee ../logs/server_$(date +%Y%m%d_%H%M%S).log

    # 纯后台无人值守
    nohup python server_hobbit.py > ../logs/server_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    # 一键运行（推荐）
    bash run.sh

本地 dry-run（跳过模型加载，仅验证代码逻辑）：
    SKIP_MODEL_LOAD=1 python server_hobbit.py

前提：
    pip install torch transformers accelerate bitsandbytes sentencepiece

    国内服务器如遇网络问题，设置环境变量即可：
    export HF_ENDPOINT="https://hf-mirror.com"
    export HF_HUB_ENABLE_HF_XET=0
"""

import sys
import os

# --- 必须在任何 huggingface 相关 import 之前设置 ---
# 强制关闭 Xet（HF-Mirror 不支持，否则 401 Unauthorized）
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"

import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM, BitsAndBytesConfig


# ============================================================
# 0. 环境自检
# ============================================================
def env_check():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {sys.argv[0]}")
    print(f"[ENV] Python={sys.version.split()[0]}")
    print(f"[ENV] PyTorch={torch.__version__}")
    print(f"[ENV] CUDA available={torch.cuda.is_available()}")

    # HF-Mirror 镜像站（解决 huggingface.co 不可达问题）
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"[ENV] HF_ENDPOINT={hf_endpoint} (using mirror)")
    else:
        print("[ENV] HF_ENDPOINT not set (using huggingface.co directly)")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"[ENV] GPU[{i}]: {props.name}, "
                f"VRAM={props.total_memory//(1024**3)}GB, "
                f"Compute={props.major}.{props.minor}"
            )
    else:
        print("[WARN] CUDA not available, running on CPU (will be very slow)")

    # 检查 bitsandbytes
    try:
        import bitsandbytes as bnb

        print(f"[ENV] bitsandbytes={bnb.__version__}")
    except ImportError:
        print("[WARN] bitsandbytes not installed, INT4 quantization will be skipped")
        print("[WARN] Install: pip install bitsandbytes")


# ============================================================
# 1. HOBBIT 参数配置
# ============================================================
HOBBIT_CONFIG = {
    "T1": 0.6,  # 论文 Mixtral 策略值（仿真阶段用 0.3，服务器用论文 0.6）
    "T2": 0.9,  # 高于此值直接跳过
    "fp16_cache_size": 2,
    "int4_cache_size": 6,
}


# ============================================================
# 2. HOBBIT 版 MoE 前向传播（猴子补丁方式替换原生的 forward）
# ============================================================
def make_hobbit_forward(original_forward, layer_idx, stats):
    """
    返回一个包装过的 forward 函数，在原生 MoE 计算前后插入 HOBBIT 决策逻辑。

    注意：第一版服务器脚本主要验证决策逻辑和统计，INT4 实际计算在后续版本实现。
    当前版本在决策点打印日志并统计，但仍用 FP16 专家完成计算（确保输出正确）。
    """

    def hobbit_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # --- Step 1: 路由（和原生完全一致）---
        router_logits, top_k_weights, top_k_index = self.gate(hidden_states_flat)

        # --- Step 2: HOBBIT 动态重要性决策（只做统计，不改变计算路径）---
        topk_cpu = top_k_index.cpu().numpy()
        weights_cpu = top_k_weights.cpu().numpy()

        for token_idx in range(num_tokens):
            token_experts = topk_cpu[token_idx]
            token_weights = weights_cpu[token_idx]

            # 计算不重要度得分
            total_weight = token_weights.sum()
            cumulative = 0.0
            for i in range(len(token_experts)):
                expert_id = int(token_experts[i])
                if i == 0:
                    score = 0.0
                else:
                    cumulative += token_weights[i - 1]
                    score = cumulative / total_weight

                # 三档决策
                if score <= HOBBIT_CONFIG["T1"]:
                    # 第一档：重要专家 -> 必须 FP16
                    if expert_id in stats["fp16_cache"]:
                        stats["hit_fp16"] += 1
                    else:
                        stats["miss_fp16"] += 1
                elif score <= HOBBIT_CONFIG["T2"]:
                    # 第二档：中等重要 -> 用 INT4
                    stats["use_int4"] += 1
                else:
                    # 第三档：极不重要 -> 跳过
                    stats["skip"] += 1

        # --- Step 3: 原生专家计算（暂时保持 FP16，确保输出正确）---
        hidden_states = self.experts(hidden_states_flat, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states

    return hobbit_forward


# ============================================================
# 3. 主流程
# ============================================================
def main():
    env_check()

    # --- Dry-run 模式：跳过模型加载，仅验证代码路径 ---
    if os.environ.get("SKIP_MODEL_LOAD", "").strip() in ("1", "true", "True"):
        print(f"\n[{time.strftime('%H:%M:%S')}] SKIP_MODEL_LOAD=1 — dry-run mode")
        print("[DRY-RUN] Skipping model download and inference.")
        print("[DRY-RUN] Code path verified: env_check + imports all OK.")
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE (dry-run)")
        return

    # --- 3a. 加载真实 Mixtral-8x7B ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading Mixtral-8x7B model...")
    model_id = "mistralai/Mixtral-8x7B-v0.1"

    # 支持从本地目录加载
    local_path = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    if local_path and os.path.isdir(local_path):
        print(f"[LOAD] Using local model: {local_path}")
        model_id = local_path

    # 根据 GPU 数量决定 device_map
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 1:
        device_map = "auto"
        print(f"[LOAD] Using device_map='auto' with {n_gpus} GPU(s)")
    else:
        device_map = "cpu"
        print("[LOAD] No GPU detected, using CPU (inference will be very slow)")

    # 4-bit 量化加载：Mixtral-8x7B FP16 = 94GB > 2xL20(88GB)，必须压缩
    # 用 bitsandbytes NF4 把模型压到 ~24GB，在量化模型之上运行 HOBBIT 决策逻辑
    # 这不是替代 HOBBIT，是让它能加载的前提
    print("[LOAD] Using 4-bit quantization (bitsandbytes NF4) to fit GPU memory...")
    print("[LOAD] Mixtral-8x7B FP16=94GB > 2xL20(88GB), need 4-bit (~24GB) to load")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = MixtralForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            local_files_only=bool(local_path),
        )
    except Exception as e:
        print(f"[LOAD] Failed to load model: {e}")
        print("[LOAD] If download failed, pre-download with hf command:")
        print(f"[LOAD]   export HF_ENDPOINT=https://hf-mirror.com")
        print(f"[LOAD]   hf download {model_id} --local-dir ~/models/mixtral-8x7b")
        print(f"[LOAD] Then re-run with:")
        print(f"[LOAD]   LOCAL_MODEL_PATH=~/models/mixtral-8x7b bash run.sh")
        raise
    print(f"[LOAD] Model loaded successfully")

    # 打印每层设备分布
    for i, layer in enumerate(model.model.layers):
        # 检查 MoE 层在哪个设备
        moe_block = layer.mlp
        try:
            dev = next(moe_block.parameters()).device
        except StopIteration:
            dev = "unknown"
        if i < 4 or i >= 28:
            print(f"[LOAD]   Layer {i:2d}: MoE on {dev}")

    # --- 3b. 替换所有 MoE 层的 forward ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Patching MoE layers with HOBBIT forward...")
    all_stats = {
        "hit_fp16": 0,
        "miss_fp16": 0,
        "use_int4": 0,
        "skip": 0,
    }
    per_layer_stats = []

    for layer_idx, layer in enumerate(model.model.layers):
        moe_block = layer.mlp
        layer_stats = {
            "layer": layer_idx,
            "hit_fp16": 0,
            "miss_fp16": 0,
            "use_int4": 0,
            "skip": 0,
        }
        # 初始化 FP16 缓存（模拟：前 2 个专家常驻 GPU）
        layer_stats["fp16_cache"] = set([0, 1])
        per_layer_stats.append(layer_stats)

        # 猴子补丁：替换 forward 方法
        original_forward = moe_block.forward
        moe_block.forward = make_hobbit_forward(
            original_forward, layer_idx, layer_stats
        ).__get__(moe_block, type(moe_block))

    print(f"[PATCH] All {len(model.model.layers)} layers patched")

    # --- 3c. 功能验证：小 batch 推理 ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Running inference test (seq_len=32)...")
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        input_ids = test_input["input_ids"]
        if n_gpus > 0:
            input_ids = input_ids.to("cuda:0")
        print(f"[INFER] Input: {tokenizer.decode(input_ids[0])}")
    except Exception:
        # fallback: 随机 token IDs
        input_ids = torch.randint(0, 32000, (1, 32))
        if n_gpus > 0:
            input_ids = input_ids.to("cuda:0")
        print("[INFER] Using random input (tokenizer not available)")

    print(f"[INFER] Input shape: {input_ids.shape}")

    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    elapsed = time.time() - t0

    logits = outputs.logits
    print(f"[INFER] Output logits shape: {logits.shape}")
    print(f"[INFER] Inference time: {elapsed:.2f}s")

    # 如果是真实输入，打印 top-5 token
    if tokenizer is not None:
        next_token_logits = logits[0, -1, :]
        top5 = torch.topk(next_token_logits, 5)
        print(f"[INFER] Top-5 next tokens:")
        for i in range(5):
            tok_id = top5.indices[i].item()
            tok_str = tokenizer.decode([tok_id])
            print(
                f"[INFER]   {i+1}. '{tok_str}' (id={tok_id}, prob={top5.values[i].item():.4f})"
            )

    # --- 3d. HOBBIT 统计汇总 ---
    print(f"\n[{time.strftime('%H:%M:%S')}] HOBBIT Decision Statistics:")
    print(
        f"{'Layer':>6} {'FP16-Hit':>10} {'FP16-Miss':>10} {'INT4-Use':>10} {'Skip':>10} {'Hit%':>8}"
    )
    print("-" * 60)

    total_all = {"hit_fp16": 0, "miss_fp16": 0, "use_int4": 0, "skip": 0}
    for s in per_layer_stats:
        t = s["hit_fp16"] + s["miss_fp16"] + s["use_int4"] + s["skip"]
        hit_pct = s["hit_fp16"] / t * 100 if t > 0 else 0
        print(
            f"{s['layer']:>6} {s['hit_fp16']:>10} {s['miss_fp16']:>10} "
            f"{s['use_int4']:>10} {s['skip']:>10} {hit_pct:>7.1f}%"
        )
        for k in total_all:
            total_all[k] += s[k]

    total_t = sum(total_all.values())
    if total_t > 0:
        print("-" * 60)
        print(
            f"{'TOTAL':>6} {total_all['hit_fp16']:>10} {total_all['miss_fp16']:>10} "
            f"{total_all['use_int4']:>10} {total_all['skip']:>10} "
            f"{total_all['hit_fp16']/total_t*100:>7.1f}%"
        )

        print(f"\n[STATS] Summary:")
        print(f"[STATS]   Total expert calls: {total_t}")
        print(f"[STATS]   FP16 hit rate:      {total_all['hit_fp16']/total_t*100:.1f}%")
        print(
            f"[STATS]   FP16 miss (->INT4):  {total_all['miss_fp16']/total_t*100:.1f}%"
        )
        print(
            f"[STATS]   INT4 planned:        {total_all['use_int4']/total_t*100:.1f}%"
        )
        print(f"[STATS]   Skip rate:           {total_all['skip']/total_t*100:.1f}%")
        combined_int4 = total_all["miss_fp16"] + total_all["use_int4"]
        print(
            f"[STATS]   Total would-use-INT4: {combined_int4/total_t*100:.1f}% "
            f"(FP16 miss fallback + planned INT4)"
        )

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE")
    print("[NOTE] This is Phase 3 functional verification only.")
    print("[NOTE] INT4 computation not yet integrated — all experts still use FP16.")
    print(
        "[NOTE] Next: create INT4-quantized expert copies with bitsandbytes for real speedup."
    )


if __name__ == "__main__":
    main()
