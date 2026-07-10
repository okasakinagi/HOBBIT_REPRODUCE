"""
explore_hooks.py — 探查 meta 层的 hooks 机制
在服务器上运行：python tools/explore_hooks.py
"""
import os, sys
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
import torch
from transformers import MixtralForCausalLM

model_id = os.environ.get("LOCAL_MODEL_PATH", "~/models/mixtral-8x7b")
model = MixtralForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
    max_memory={0: "40GB", 1: "40GB", "cpu": "200GB"},
    low_cpu_mem_usage=True, local_files_only=True,
)

for i in [0, 25]:
    mlp = model.model.layers[i].mlp
    print(f"\nLayer {i} mlp:")
    if hasattr(mlp, "_hf_hook") and mlp._hf_hook is not None:
        h = mlp._hf_hook
        print(f"  _hf_hook: {type(h).__name__}")
        # 列出 hook 的所有非私有属性
        for attr in dir(h):
            if not attr.startswith("_"):
                val = getattr(h, attr)
                if isinstance(val, (int, str, torch.device)):
                    print(f"    {attr}: {val}")
                elif isinstance(val, dict):
                    print(f"    {attr}: dict with {len(val)} keys")
                    if len(val) > 0:
                        sample_key = list(val.keys())[0]
                        sample_val = val[sample_key]
                        print(f"      sample key: {sample_key}")
                        if hasattr(sample_val, 'shape'):
                            print(f"      sample val: shape={sample_val.shape}, device={sample_val.device}")
                        else:
                            print(f"      sample val type: {type(sample_val).__name__}")
                elif hasattr(val, 'shape'):
                    print(f"    {attr}: tensor shape={val.shape}, device={val.device}")
                elif val is not None:
                    print(f"    {attr}: {type(val).__name__} = {str(val)[:80]}")
    else:
        print(f"  _hf_hook: None")

# === 关键实验：通过 decoder layer 做 forward 后检查 meta 权重 ===
print("\n\n=== Forward through decoder layer 25 (with correct dtype) ===")
dummy = torch.zeros(1, 1, model.config.hidden_size, device="cuda:0", dtype=torch.bfloat16)
with torch.no_grad():
    _ = model.model.layers[25](dummy)

exp = model.model.layers[25].mlp.experts
print(f"After forward: gate_up_proj: device={exp.gate_up_proj.device}, is_meta={exp.gate_up_proj.is_meta}")
if not exp.gate_up_proj.is_meta:
    w = exp.gate_up_proj.data[0]
    print(f"  weight[0] shape={w.shape}, dtype={w.dtype}, min={w.min().item():.2f}, max={w.max().item():.2f}")
else:
    print("  Still meta after decoder_layer forward!")

# 再试一次: 看 post-forward hook 后权重去哪了
print(f"After forward: gate_up_proj device={exp.gate_up_proj.device}")

