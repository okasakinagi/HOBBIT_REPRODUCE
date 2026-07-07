import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM
from typing import Optional, Tuple

# ========================================================
# 阶段2：Mixtral空壳模型 + HOBBIT逻辑缝合
# 用极小配置创建和真实Mixtral结构完全一致的迷你模型，本地CPU就能跑
# ========================================================


def create_minimal_mixtral_config():
    """创建迷你Mixtral配置，结构和真实Mixtral-8x7B完全一致，只是维度缩小"""
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=256,  # 真实是4096，缩小到256
        intermediate_size=512,  # 真实是14336，缩小到512
        num_hidden_layers=2,  # 真实是32层，缩小到2层方便调试
        num_attention_heads=8,  # 真实是32，缩小到8
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        num_experts=8,  # 新版transformers参数名：8个专家，和真实一致
        num_experts_per_tok=2,  # Top-2专家，和真实一致
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        architectures=["MixtralForCausalLM"],
    )
    return config


# ========================================================
# 自定义HOBBIT版的MoE层，替换原生的MixtralSparseMoeBlock
# 结构和原生完全一致，只是插入了HOBBIT动态精度决策逻辑
# ========================================================
class HobbitSparseMoeBlock(nn.Module):
    def __init__(self, config, hobbit_config=None):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # HOBBIT配置（和我们仿真里的参数一致）
        if hobbit_config is None:
            hobbit_config = {
                "T1": 0.3,  # 重要性阈值1：低于T1必须FP16
                "T2": 0.9,  # 重要性阈值2：高于T2直接跳过
                "fp16_cache_size": 2,  # FP16缓存大小
                "int4_cache_size": 6,  # INT4缓存大小
            }
        self.hobbit_config = hobbit_config

        # 路由器（和原生完全一致）
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # 专家：和原生结构一致，这里我们用普通线性层模拟（真实模型是MLP）
        # 空壳阶段只验证结构和形状，不需要真实权重
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.intermediate_size, self.hidden_size, bias=False),
                )
                for _ in range(self.num_experts)
            ]
        )

        # HOBBIT状态：缓存状态、传输队列（空壳阶段先做占位，上服务器再实现真实传输）
        self.fp16_cache = set([0, 1])  # 初始0、1号专家在FP16缓存
        self.int4_cache = set(
            range(self.num_experts)
        )  # INT4全部常驻（调试阶段先设为全部在，后面改动态加载）
        self.loading_queue = set()

        # 统计计数器
        self.hit_count = 0
        self.int4_count = 0
        self.skip_count = 0
        self.miss_count = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE层前向传播，和原生结构完全一致，插入HOBBIT逻辑
        注意：虽然类型标注是tuple，但原生实际只返回hidden_states单个tensor
        router_logits通过transformers的OutputRecorder机制单独收集
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(
            -1, hidden_dim
        )  # 展平为[num_tokens, hidden_dim]

        # 1. 路由计算（和原生完全一致）
        router_logits = self.gate(hidden_states)
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(router_weights, self.top_k, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)  # 归一化权重

        # 2. HOBBIT动态重要性决策（我们插入的核心逻辑）
        # 对每个Token的Top-K专家计算不重要度得分
        final_hidden_states = torch.zeros_like(hidden_states)
        for token_idx in range(hidden_states.shape[0]):
            token_input = hidden_states[token_idx : token_idx + 1]
            token_experts = topk_ids[token_idx]
            token_weights = topk_weights[token_idx]

            # 计算不重要度得分（和仿真里逻辑完全一致）
            total_weight = token_weights.sum().item()
            unimportance_scores = []
            cumulative = 0.0
            for i, w in enumerate(token_weights):
                if i == 0:
                    unimportance_scores.append(0.0)
                else:
                    cumulative += token_weights[i - 1].item()
                    unimportance_scores.append(cumulative / total_weight)

            token_output = torch.zeros_like(token_input)
            for i, expert_idx in enumerate(token_experts):
                expert_idx = expert_idx.item()
                score = unimportance_scores[i]
                expert_weight = token_weights[i]

                if score <= self.hobbit_config["T1"]:
                    # 第一档：重要专家，必须用FP16
                    if expert_idx in self.fp16_cache:
                        # 命中FP16缓存
                        self.hit_count += 1
                        expert_out = self.experts[expert_idx](token_input)
                        token_output += expert_weight * expert_out
                    else:
                        # FP16 Miss，空壳阶段先直接用INT4兜底（真实系统会等传输/后台加载）
                        self.miss_count += 1
                        expert_out = self.experts[expert_idx](
                            token_input
                        )  # 空壳阶段用同一个专家模拟INT4
                        token_output += expert_weight * expert_out
                        # 加入加载队列（真实系统后台异步传输）
                        self.loading_queue.add(expert_idx)

                elif score <= self.hobbit_config["T2"]:
                    # 第二档：中等重要，用INT4
                    self.int4_count += 1
                    expert_out = self.experts[expert_idx](token_input)
                    token_output += expert_weight * expert_out
                    # 后台异步加载FP16
                    if expert_idx not in self.fp16_cache:
                        self.loading_queue.add(expert_idx)

                else:
                    # 第三档：极不重要，直接跳过
                    self.skip_count += 1
                    # 不计算，贡献为0

            final_hidden_states[token_idx] = token_output

        # 3. 返回和输入形状一致的输出（和原生完全一致，只返回hidden_states）
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

    def reset_stats(self):
        """重置统计计数器"""
        self.hit_count = 0
        self.int4_count = 0
        self.skip_count = 0
        self.miss_count = 0
        self.loading_queue.clear()


# ========================================================
# 主函数：加载空壳模型，替换MoE层，跑通前向传播
# ========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("阶段2：Mixtral空壳模型 + HOBBIT逻辑缝合")
    print("=" * 70)

    # 1. 创建迷你配置
    print("\n1. 创建迷你Mixtral配置（结构和真实Mixtral-8x7B完全一致）...")
    config = create_minimal_mixtral_config()
    print(
        f"   配置：{config.num_hidden_layers}层，{config.num_local_experts}专家，Top-{config.num_experts_per_tok}，hidden_size={config.hidden_size}"
    )
    # 2. 创建迷你模型（迷你模型才2M参数，CPU完全够用）
    print("\n2. 创建迷你Mixtral模型（2M参数，CPU完全够用）...")
    model = MixtralForCausalLM(config)

    # 3. 先看看DecoderLayer有哪些属性，找到MoE层的名字
    print("\n3. 查看DecoderLayer结构，找到MoE层位置...")
    sample_layer = model.model.layers[0]
    print(
        "   DecoderLayer包含的子模块：",
        [name for name, _ in sample_layer.named_children()],
    )

    # 替换所有MoE层为我们的HOBBIT版MoE层
    print("\n4. 替换原生MoE层为HOBBIT版MoE层...")
    for layer_idx, layer in enumerate(model.model.layers):
        # 新版transformers里MoE层属性名是mlp，类型是MixtralSparseMoeBlock
        moe_attr_name = None
        for name, module in layer.named_children():
            if "MixtralSparseMoeBlock" in module.__class__.__name__:
                moe_attr_name = name
                break

        if moe_attr_name is None:
            raise AttributeError(
                f"找不到MoE层，第{layer_idx}层的属性：{[name for name, _ in layer.named_children()]}"
            )
        # 替换成我们的HOBBIT版
        setattr(layer, moe_attr_name, HobbitSparseMoeBlock(config))
        print(f"   替换第{layer_idx}层MoE层完成（属性名：{moe_attr_name}）")

    # 4. 模型已在CPU上，迷你模型无需额外初始化
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"   模型总参数量：{total_params/1e6:.2f}M（真实Mixtral-8x7B是47B，缩小了几百倍方便调试）"
    )

    # 5. 生成随机输入，跑通前向传播
    print("\n5. 生成随机输入，测试前向传播...")
    input_ids = torch.randint(0, config.vocab_size, (1, 32))  # batch=1，序列长度32
    print(f"   输入形状：{input_ids.shape}")

    # 跑前向
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    print(f"   输出logits形状：{logits.shape}")

    # 6. 统计HOBBIT运行情况
    print("\n6. HOBBIT层运行统计：")
    total_hit = 0
    total_int4 = 0
    total_skip = 0
    total_miss = 0
    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer.mlp  # 新版transformers中MoE层属性名是mlp
        total_hit += moe.hit_count
        total_int4 += moe.int4_count
        total_skip += moe.skip_count
        total_miss += moe.miss_count
        print(
            f"   第{layer_idx}层：FP16命中{moe.hit_count}次，INT4使用{moe.int4_count}次，跳过{moe.skip_count}次，Miss{moe.miss_count}次"
        )
        moe.reset_stats()

    total_calls = total_hit + total_int4 + total_skip + total_miss
    print(
        f"\n   总计：FP16命中率{total_hit/total_calls*100:.1f}%，INT4比例{total_int4/total_calls*100:.1f}%，跳过比例{total_skip/total_calls*100:.1f}%"
    )

    print("\n" + "=" * 70)
    print(
        "[OK] 空壳模型前向传播跑通！HOBBIT逻辑成功缝合到Mixtral结构中，张量形状完全对齐！"
    )
    print(
        "   上服务器只需要把迷你配置换成真实Mixtral配置，加载真实权重即可，不需要改核心逻辑。"
    )
    print("=" * 70)
