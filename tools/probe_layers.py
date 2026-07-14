"""探查模型加载后各层的实际设备分布"""
import os, sys
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"

import torch
from transformers import MixtralForCausalLM

model_id = os.environ.get("LOCAL_MODEL_PATH", "~/models/mixtral-8x7b")
model = MixtralForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "40GB", 1: "40GB", "cpu": "200GB"},
    low_cpu_mem_usage=True,
    local_files_only=True,
)

gpu0, gpu1, cpu_layers = [], [], []
for i, layer in enumerate(model.model.layers):
    # 取 self_attn 的第一个参数的 device 来判断
    p = next(layer.self_attn.parameters())
    d = str(p.device)
    if "cuda:0" in d:
        gpu0.append(i)
    elif "cuda:1" in d:
        gpu1.append(i)
    else:
        cpu_layers.append(i)

print(f"GPU 0: 层 {gpu0[0]}-{gpu0[-1]} 共 {len(gpu0)} 层")
print(f"GPU 1: 层 {gpu1[0]}-{gpu1[-1]} 共 {len(gpu1)} 层")
print(f"CPU:   层 {cpu_layers[0]}-{cpu_layers[-1]} 共 {len(cpu_layers)} 层")
print(f"总计: {len(gpu0)}+{len(gpu1)}+{len(cpu_layers)} = {len(gpu0)+len(gpu1)+len(cpu_layers)}")
