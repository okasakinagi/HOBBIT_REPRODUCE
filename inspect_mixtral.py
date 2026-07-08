"""探查真实Mixtral模型结构，找到MoE层的正确位置和接口"""

from transformers import MixtralConfig, MixtralForCausalLM
from accelerate import init_empty_weights
import torch

print("=" * 60)
print("探查Mixtral模型真实结构")
print("=" * 60)

# 创建真实配置（用官方默认参数，不缩小）
config = MixtralConfig(
    num_hidden_layers=2,  # 只建2层，快
    num_local_experts=8,
    num_experts_per_tok=2,
)

print("\n1. 配置参数：")
print(f"   num_local_experts: {config.num_local_experts}")
print(f"   num_experts_per_tok: {config.num_experts_per_tok}")
print(f"   hidden_size: {config.hidden_size}")
print(f"   intermediate_size: {config.intermediate_size}")

# 加载空壳模型
print("\n2. 加载空壳模型...")
with init_empty_weights():
    model = MixtralForCausalLM(config)

# 查看第一层DecoderLayer的结构
print("\n3. DecoderLayer包含的子模块：")
layer0 = model.model.layers[0]
for name, module in layer0.named_children():
    print(f"   - {name}: {module.__class__.__name__}")

# 找到MoE层（在新版transformers中，MoE层属性名是mlp，类型是MixtralSparseMoeBlock）
moe_layer = None
moe_attr_name = None
for name, module in layer0.named_children():
    class_name = module.__class__.__name__
    if "Moe" in class_name or "Sparse" in class_name:
        moe_layer = module
        moe_attr_name = name
        break

print(f"\n4. 找到MoE层，属性名：{moe_attr_name}")
print(f"   MoE层类名：{moe_layer.__class__.__name__}")

# 查看MoE层的子模块
print("\n5. MoE层包含的子模块：")
for name, module in moe_layer.named_children():
    print(f"   - {name}: {module.__class__.__name__}")

# 查看MoE层的forward接口签名
import inspect

print("\n6. MoE层forward方法签名：")
sig = inspect.signature(moe_layer.forward)
print(f"   {sig}")

# 测试MoE层输入输出形状
print("\n7. 测试MoE层输入输出形状：")
test_input = torch.randn(1, 4, config.hidden_size)  # batch=1, seq_len=4
with torch.no_grad():
    # 空壳模型前向需要meta张量，我们直接看输入输出要求
    print(f"   输入形状：{test_input.shape}")
    print(f"   输出形状应该和输入一致：{test_input.shape}")

print("\n" + "=" * 60)
print("探查完成！我们会完全按照这个结构写HOBBIT版MoE层，保证接口完全兼容")
print("=" * 60)
