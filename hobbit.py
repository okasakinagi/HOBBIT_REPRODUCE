import torch
import torch.nn as nn
import time


# ========================================================
# 1. 定义一个用于本地调试的 HOBBIT 模拟专家层
# ========================================================
class HobbitMoELayer(nn.Module):
    def __init__(self, num_experts=8, d_model=16):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # 模拟 Router（路由器），给每个 Token 分配专家
        self.router = nn.Linear(d_model, num_experts)

        # 本地不加载大模型，我们只初始化极小的线性层来代表专家
        # 模拟常驻内存的 高精度(FP16) 专家（传输慢）
        self.experts_fp16 = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_experts)]
        )
        # 模拟常驻显存的 低精度(INT4) 备份专家（无需传输，原地计算）
        self.experts_int4 = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_experts)]
        )

        # 模拟显存的缓存状态（Cache Status）
        # 假设当前只有 0 号和 1 号专家在显存里（True），其余 2~7 号都在内存里（False）
        self.gpu_cache_status = [True, True, False, False, False, False, False, False]

    def forward(self, x):
        # x 的形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        # 将输入展平为 [Token数量, d_model]
        tokens = x.view(-1, d_model)

        # 计算每个 Token 对每个专家的路由分数
        router_logits = self.router(tokens)
        # 假设每个 Token 激活 Top-1 个专家
        _, selected_experts = torch.topk(router_logits, k=1, dim=-1)

        # 打印路由结果方便理解
        print("\n[路由调试] 每个Token的专家得分（前3个Token示例）：")
        for i in range(min(3, len(tokens))):
            print(
                f"Token {i} 专家得分: {router_logits[i].detach().numpy().round(2)} → 选中专家: {selected_experts[i].item()}"
            )

        output_tokens = torch.zeros_like(tokens)

        # 计数器：记录发生了多少次拦截切换
        hit_count = 0
        hobbit_switch_count = 0

        # 遍历每一个 Token，处理路由和 HOBBIT 逻辑
        for i, token in enumerate(tokens):
            expert_idx = selected_experts[i].item()
            token_input = token.unsqueeze(0)  # 保持二维形状用于矩阵乘法

            # --- 核心：HOBBIT 动态控制流 ---
            if self.gpu_cache_status[expert_idx] == True:
                # 情况 A：专家就在显存缓存中，直接用高精度计算
                hit_count += 1
                expert_output = self.experts_fp16[expert_idx](token_input)
            else:
                # 情况 B：Cache Miss（未命中）！
                # 按照传统做法，这里应该卡住去加载 FP16。
                # 但 HOBBIT 触发拦截：强行切换到显存中已有的低精度备用版本！
                hobbit_switch_count += 1
                # print(f"[HOBBIT 拦截] Token {i} 目标专家 {expert_idx} 未命中显存！已动态降级为显存内低精度版本计算。")
                expert_output = self.experts_int4[expert_idx](token_input)

            output_tokens[i] = expert_output.squeeze(0)

        print(
            f"\n[统计结果] 总 Token 数: {len(tokens)} | 正常命中高精度: {hit_count} | HOBBIT拦截换低精度: {hobbit_switch_count}"
        )

        return output_tokens.view(batch_size, seq_len, d_model)


# ========================================================
# 2. 本地直接运行测试
# ========================================================
if __name__ == "__main__":
    print("=== 开始本地 HOBBIT 控制流调试 ===")

    # 初始化我们的模拟层（仅需几KB内存，本地瞬间启动）
    hobbit_layer = HobbitMoELayer(num_experts=8, d_model=16)

    # 伪造一个极小的输入：Batch=1, 句子长度=10, 向量维度=16
    mock_input = torch.randn(1, 10, 16)

    # 跑通前向传播
    output = hobbit_layer(mock_input)

    print("=== 前向传播成功跑通！张量形状完全对齐 ===")
    print("输出形状:", output.shape)
