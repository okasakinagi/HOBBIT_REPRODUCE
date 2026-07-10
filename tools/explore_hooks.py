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
    print(f"  down_proj:    device={experts.down_proj.device}, is_meta={experts.down_proj.is_meta}")

    # 查找所有 hook 相关属性
    hook_attrs = [a for a in dir(mlp) if "hook" in a.lower()]
    print(f"  hook attrs: {hook_attrs}")

    # 检查 _hf_hook
    if hasattr(mlp, "_hf_hook"):
        h = mlp._hf_hook
        print(f"  _hf_hook: {type(h).__name__}")
        if h is not None:
            print(f"    execution_device: {h.execution_device}")
            print(f"    offload_device: {h.offload_device}")
    else:
        print(f"  _hf_hook: not found")

    # 检查是否有 hooks dict
    for attr_name in ["_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_modules"]:
        if hasattr(mlp, attr_name):
            hooks = getattr(mlp, attr_name)
            print(f"  {attr_name}: {len(hooks)} entries" if isinstance(hooks, dict) else f"  {attr_name}: present")

# 对 meta 层：做一次 forward 后检查权重状态
print("\n\nAfter dummy forward on layer 25:")
dummy = torch.zeros(1, 1, model.config.hidden_size, device="cuda:0")
with torch.no_grad():
    out = model.model.layers[25].mlp(dummy)

experts = model.model.layers[25].mlp.experts
print(f"  gate_up_proj: device={experts.gate_up_proj.device}, is_meta={experts.gate_up_proj.is_meta}")
print(f"  data[0] device: {experts.gate_up_proj.data[0].device}")
print(f"  data[0] is_meta: {experts.gate_up_proj.data[0].is_meta}")

# 再试一次读取权重
if not experts.gate_up_proj.is_meta:
    w = experts.gate_up_proj.data[0]
    print(f"  weight[0] shape: {w.shape}, dtype: {w.dtype}, min: {w.min().item():.4f}")
else:
    print("  Still meta!")
