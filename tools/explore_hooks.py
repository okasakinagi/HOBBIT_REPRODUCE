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

# === 直接读 safetensors 文件 ===
print("\n\n=== Read expert weights from safetensors ===")
model_path = os.path.expanduser("~/models/mixtral-8x7b")
import glob

# 找 tensor 名
layer25_param_names = [n for n, p in model.model.layers[25].mlp.experts.named_parameters()]
print(f"Layer 25 experts param names: {layer25_param_names}")

# 找 safetensors 文件
sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
print(f"Safetensors files: {[os.path.basename(f) for f in sf_files]}")

# 在 safetensors 中搜索对应的 tensor name
from safetensors import safe_open
target_names = [
    "model.layers.25.mlp.experts.gate_up_proj",
    "model.layers.25.mlp.experts.down_proj",
]
for sf_path in sf_files:
    fname = os.path.basename(sf_path)
    try:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for t in target_names:
                if t in keys:
                    tensor = f.get_tensor(t)
                    print(f"  FOUND in {fname}: {t}: shape={tensor.shape}, dtype={tensor.dtype}, "
                          f"min={tensor.min().item():.2f}, max={tensor.max().item():.2f}, "
                          f"device={tensor.device}, is_meta={tensor.is_meta}")
    except Exception as e:
        print(f"  Error reading {fname}: {e}")

