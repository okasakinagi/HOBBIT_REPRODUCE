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

for i in [0, 25, 26]:
    mlp = model.model.layers[i].mlp
    experts = mlp.experts
    print(f"\nLayer {i} mlp:")
    print(f"  gate_up_proj: device={experts.gate_up_proj.device}, is_meta={experts.gate_up_proj.is_meta}")

    if hasattr(mlp, "_hf_hook") and mlp._hf_hook is not None:
        h = mlp._hf_hook
        print(f"  _hf_hook: {type(h).__name__}")
        for attr in ["execution_device", "offload_device"]:
            if hasattr(h, attr):
                print(f"    {attr}: {getattr(h, attr)}")
    else:
        print(f"  _hf_hook: None or not found")

# === 关键实验：dummy forward 后 meta 层能否读取 ===
print("\n\n=== Dummy forward through layer 25 mlp ===")
dummy = torch.zeros(1, 1, model.config.hidden_size, device="cuda:0")
with torch.no_grad():
    _ = model.model.layers[25].mlp(dummy)

exp = model.model.layers[25].mlp.experts
print(f"After forward: gate_up_proj: device={exp.gate_up_proj.device}, is_meta={exp.gate_up_proj.is_meta}")
if not exp.gate_up_proj.is_meta:
    w = exp.gate_up_proj.data[0]
    print(f"  weight[0] shape={w.shape}, dtype={w.dtype}, min={w.min().item():.2f}, max={w.max().item():.2f}")
else:
    print("  Still meta after forward!")

