import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

try:
    import matplotlib.pyplot as plt

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ========================================================
# INT4 量化工具函数
# ========================================================
def quantize_int4(tensor_fp):
    """对称 INT4 量化：映射至 [-8, 7] 整数范围，再反量化回浮点。

    模拟论文中使用的真实 INT4 量化过程。
    返回：(反量化张量, 均方误差)。
    """
    w_max = tensor_fp.abs().max()
    if w_max == 0:
        return tensor_fp.clone(), 0.0
    scale = w_max / 7.0
    w_int4 = torch.round(tensor_fp / scale).clamp(-8, 7)
    w_dequant = w_int4 * scale
    mse = ((tensor_fp - w_dequant) ** 2).mean().item()
    return w_dequant, mse


# ========================================================
# HOBBIT：混合精度专家卸载仿真系统
#
# 实现论文三大核心创新：
#   1. Token 级动态重要性决策（双阈值 T1/T2）
#   2. Layer 级自适应预取
#   3. Sequence 级 LHU 多维缓存策略
#
# 采用真实 INT4 量化（量化-反量化-计算），测量精度损失。
# ========================================================
class HobbitFinal:
    def __init__(
        self,
        num_experts=8,
        top_k=2,
        d_model=4096,
        num_layers=32,
        # 双独立缓存（FP16 与 INT4 分立管理）
        fp16_cache_size=2,
        int4_cache_size=6,
        # 计算延迟
        compute_latency_ms=2,
        int4_compute_latency_ms=1.6,
        # 传输延迟（CPU → GPU，PCIe 串行；INT4 体积为 FP16 的 1/4）
        fp16_transfer_ms=4,
        int4_transfer_ms=1,
        max_concurrent_transfers=1,
        # 双阈值（论文核心参数）
        T1=0.3,
        T2=0.9,
        # 预取参数
        prefetch_layers=2,
        # LHU 缓存权重（FP16 Miss 惩罚更大，故 LHU 权重最高）
        w_lru=0.2,
        w_lfu=0.3,
        w_lhu=0.4,
        w_fld=0.1,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.num_layers = num_layers
        self.fp16_cache_size = fp16_cache_size
        self.int4_cache_size = int4_cache_size
        self.compute_latency_ms = compute_latency_ms
        self.int4_compute_latency_ms = int4_compute_latency_ms
        self.fp16_transfer_ms = fp16_transfer_ms
        self.int4_transfer_ms = int4_transfer_ms
        self.max_concurrent_transfers = max_concurrent_transfers
        self.T1 = T1
        self.T2 = T2
        self.prefetch_layers = prefetch_layers
        self.w_lru = w_lru
        self.w_lfu = w_lfu
        self.w_lhu = w_lhu
        self.w_fld = w_fld

        # 每层一个路由器
        self.routers = nn.ModuleList(
            [nn.Linear(d_model, num_experts, bias=False) for _ in range(num_layers)]
        )

        # 合成专家权重矩阵，用于真实 INT4 量化仿真。
        # 每个专家模拟为一个线性投影层（hidden → intermediate）。
        self.intermediate_size = 256  # 仿真用的小尺寸，仅用于测量 INT4 量化误差
        self.expert_weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(num_experts, self.intermediate_size, d_model) * 0.02,
                    requires_grad=False,
                )
                for _ in range(num_layers)
            ]
        )

    # ========================================================
    # 生成多层专家访问 Trace
    # ========================================================
    def generate_multi_layer_trace(self, num_tokens=1000):
        """生成多层专家访问序列，模拟残差连接导致的层间相似性。

        论文指出相邻层门控输入的余弦相似度均值约为 0.86。
        返回的 trace 包含每层的 hidden states，用于计算真实 MoE 输出。
        """
        all_layer_traces = []
        x = torch.randn(1, num_tokens, self.d_model)
        tokens = x.view(-1, self.d_model)

        for layer_idx in range(self.num_layers):
            router = self.routers[layer_idx]
            router_logits = router(tokens)
            router_weights = F.softmax(router_logits, dim=-1)
            topk_weights, topk_indices = torch.topk(
                router_weights, k=self.top_k, dim=-1
            )
            all_layer_traces.append(
                {
                    "indices": topk_indices.detach().numpy(),
                    "weights": topk_weights.detach().numpy(),
                    "hidden": tokens.detach().clone(),  # 保留 hidden states 用于输出对比
                }
            )
            # 残差连接：相邻层输入相似度 ~0.86
            tokens = tokens + torch.randn_like(tokens) * 0.1

        return all_layer_traces

    # ========================================================
    # 计算不重要度得分（论文公式）
    # ========================================================
    def compute_unimportance_score(self, expert_weights):
        total_weight = sum(expert_weights)
        scores = []
        cumulative = 0.0
        for i, w in enumerate(expert_weights):
            if i == 0:
                scores.append(0.0)
            else:
                cumulative += expert_weights[i - 1]
                scores.append(cumulative / total_weight)
        return scores

    # ========================================================
    # MoE 层输出计算（用于精度对比）
    # ========================================================
    def _compute_moe_output(
        self, hidden, indices, weights, expert_w_layer, int4_mask_per_token=None
    ):
        """计算 MoE 层的加权专家输出。

        Args:
            hidden: (num_tokens, d_model) 隐藏状态
            indices: (num_tokens, top_k) 专家索引
            weights: (num_tokens, top_k) 路由权重
            expert_w_layer: (num_experts, intermediate_size, d_model) 专家权重
            int4_mask_per_token: list of set, 每个 token 走 INT4 的专家集合；
                                 None 表示全部使用 FP16

        Returns:
            output: (num_tokens, intermediate_size) MoE 层输出
        """
        num_tokens = hidden.shape[0]
        output = torch.zeros(num_tokens, self.intermediate_size)
        for t in range(num_tokens):
            int4_set = int4_mask_per_token[t] if int4_mask_per_token else set()
            for i, expert_idx in enumerate(indices[t]):
                w_fp = expert_w_layer[expert_idx]
                if expert_idx in int4_set:
                    w_use, _ = quantize_int4(w_fp)
                else:
                    w_use = w_fp
                expert_out = hidden[t] @ w_use.T
                output[t] += weights[t][i] * expert_out
        return output

    def _compute_layer_cosine(
        self, hidden, indices, weights, expert_w_layer, int4_mask_per_token
    ):
        """计算 MoE 层 FP16 输出与混合精度输出之间的余弦相似度。"""
        out_fp16 = self._compute_moe_output(
            hidden, indices, weights, expert_w_layer, int4_mask_per_token=None
        )
        out_mixed = self._compute_moe_output(
            hidden,
            indices,
            weights,
            expert_w_layer,
            int4_mask_per_token=int4_mask_per_token,
        )
        cos = F.cosine_similarity(
            out_fp16.flatten().unsqueeze(0), out_mixed.flatten().unsqueeze(0)
        ).item()
        return cos

    # ========================================================
    # LHU 多维缓存策略
    # FP16 与 INT4 各维护一个独立的 LHU 缓存实例（论文分立式缓存管理）。
    # 淘汰优先级：分数越低越优先被淘汰。
    # ========================================================
    class LHUCache:
        def __init__(self, capacity, w_lru, w_lfu, w_lhu, w_fld, is_high_precision):
            self.capacity = capacity
            self.w_lru = w_lru
            self.w_lfu = w_lfu
            self.w_lhu = w_lhu
            self.w_fld = w_fld
            self.is_high_precision = is_high_precision
            self.experts = {}
            self.current_time = 0
            self.current_layer = 0

        def _compute_score(self, expert_id):
            """计算淘汰优先级分数（越低越先淘汰）。"""
            info = self.experts[expert_id]
            time_since_use = self.current_time - info["last_use"]
            lru_score = 1.0 / (1.0 + time_since_use / 100.0)
            lfu_score = 1.0 / (1.0 + info["use_count"] / 10.0)
            lhu_score = 1.0 / (1.0 + info["hp_use_count"] / 10.0)
            layers_since_use = self.current_layer - info["last_layer"]
            fld_score = 1.0 / (1.0 + layers_since_use / 5.0)
            total = (
                self.w_lru * lru_score
                + self.w_lfu * lfu_score
                + self.w_lhu * lhu_score
                + self.w_fld * fld_score
            )
            return total

        def has(self, expert_id):
            return expert_id in self.experts

        def access(self, expert_id):
            """记录一次访问，更新使用统计。"""
            self.current_time += 1
            if expert_id in self.experts:
                self.experts[expert_id]["last_use"] = self.current_time
                self.experts[expert_id]["use_count"] += 1
                self.experts[expert_id]["last_layer"] = self.current_layer
                if self.is_high_precision:
                    self.experts[expert_id]["hp_use_count"] += 1
            else:
                if len(self.experts) >= self.capacity:
                    min_score = float("inf")
                    min_id = None
                    for eid in self.experts:
                        score = self._compute_score(eid)
                        if score < min_score:
                            min_score = score
                            min_id = eid
                    del self.experts[min_id]
                self.experts[expert_id] = {
                    "last_use": self.current_time,
                    "use_count": 1,
                    "hp_use_count": 1 if self.is_high_precision else 0,
                    "last_layer": self.current_layer,
                }

        def set_layer(self, layer_idx):
            self.current_layer = layer_idx

    # ========================================================
    # 统一传输队列（FP16 与 INT4 共享 PCIe 总线，串行传输）
    # ========================================================
    class TransferQueue:
        def __init__(self, max_concurrent, fp16_latency, int4_latency):
            self.queue = deque()
            self.current_transfer = None
            self.max_concurrent = max_concurrent
            self.fp16_latency = fp16_latency
            self.int4_latency = int4_latency

        def add(self, expert_id, is_fp16, current_time):
            """将传输任务加入队列（已在传输或已排队则跳过）。"""
            if self.current_transfer is not None:
                eid, is_fp, _ = self.current_transfer
                if eid == expert_id and is_fp == is_fp16:
                    return False
            for eid, is_fp in self.queue:
                if eid == expert_id and is_fp == is_fp16:
                    return False
            self.queue.append((expert_id, is_fp16))
            if self.current_transfer is None:
                self._start_next(current_time)
            return True

        def _start_next(self, current_time):
            if len(self.queue) > 0:
                expert_id, is_fp16 = self.queue.popleft()
                latency = self.fp16_latency if is_fp16 else self.int4_latency
                self.current_transfer = (expert_id, is_fp16, current_time + latency)

        def process(self, current_time, fp16_cache, int4_cache):
            """处理已完成传输，将专家加入对应缓存。"""
            completed_count = 0
            if self.current_transfer is not None:
                expert_id, is_fp16, finish_time = self.current_transfer
                if current_time >= finish_time:
                    if is_fp16:
                        fp16_cache.access(expert_id)
                    else:
                        int4_cache.access(expert_id)
                    completed_count += 1
                    self.current_transfer = None
                    self._start_next(current_time)
            return completed_count

    # ========================================================
    # 基线 1：传统阻塞式卸载（全 FP16，Miss 时阻塞等待传输）
    # ========================================================
    def run_baseline_blocking(self, all_layer_traces):
        fp16_cache = self.LHUCache(
            self.fp16_cache_size,
            self.w_lru,
            self.w_lfu,
            self.w_lhu,
            self.w_fld,
            is_high_precision=True,
        )
        current_time = 0.0
        total_tokens = len(all_layer_traces[0]["indices"])
        stats = {
            "fp16_hit": 0,
            "fp16_miss": 0,
            "int4_hit": 0,
            "int4_miss": 0,
            "skip": 0,
        }

        for layer_idx in range(self.num_layers):
            fp16_cache.set_layer(layer_idx)
            indices = all_layer_traces[layer_idx]["indices"]
            for token_idx in range(total_tokens):
                for expert_idx in indices[token_idx]:
                    if fp16_cache.has(expert_idx):
                        stats["fp16_hit"] += 1
                        current_time += self.compute_latency_ms
                        fp16_cache.access(expert_idx)
                    else:
                        stats["fp16_miss"] += 1
                        current_time += self.fp16_transfer_ms
                        current_time += self.compute_latency_ms
                        fp16_cache.access(expert_idx)

        return self._build_result(
            "基线：阻塞式 FP16 卸载",
            current_time,
            total_tokens,
            stats,
            int4_cos=1.0,
            layer_cosines=[1.0] * self.num_layers,
        )

    # ========================================================
    # 基线 2：全 INT4（所有专家均量化为 INT4，测量真实量化误差）
    # ========================================================
    def run_baseline_all_int4(self, all_layer_traces):
        int4_cache = self.LHUCache(
            self.int4_cache_size,
            self.w_lru,
            self.w_lfu,
            self.w_lhu,
            self.w_fld,
            is_high_precision=False,
        )
        current_time = 0.0
        total_tokens = len(all_layer_traces[0]["indices"])
        stats = {
            "fp16_hit": 0,
            "fp16_miss": 0,
            "int4_hit": 0,
            "int4_miss": 0,
            "skip": 0,
        }
        layer_cosines = []

        for layer_idx in range(self.num_layers):
            int4_cache.set_layer(layer_idx)
            indices = all_layer_traces[layer_idx]["indices"]
            weights = all_layer_traces[layer_idx]["weights"]
            hidden = all_layer_traces[layer_idx]["hidden"]
            expert_w = self.expert_weights[layer_idx]

            # 该层每个 token 的 INT4 专家集合（全部专家都用 INT4）
            all_experts_per_token = [
                set(int(indices[t][i]) for i in range(len(indices[t])))
                for t in range(total_tokens)
            ]
            layer_cos = self._compute_layer_cosine(
                hidden,
                indices,
                weights,
                expert_w,
                int4_mask_per_token=all_experts_per_token,
            )
            layer_cosines.append(layer_cos)

            for token_idx in range(total_tokens):
                for expert_idx in indices[token_idx]:
                    if int4_cache.has(expert_idx):
                        stats["int4_hit"] += 1
                        current_time += self.int4_compute_latency_ms
                        int4_cache.access(expert_idx)
                    else:
                        stats["int4_miss"] += 1
                        current_time += self.int4_transfer_ms
                        current_time += self.int4_compute_latency_ms
                        int4_cache.access(expert_idx)

        avg_cos = sum(layer_cosines) / len(layer_cosines)
        return self._build_result(
            "基线：全 INT4 量化",
            current_time,
            total_tokens,
            stats,
            int4_cos=avg_cos,
            layer_cosines=layer_cosines,
        )

    # ========================================================
    # HOBBIT 完整方案：混合精度推理（真实 INT4 量化 + 动态决策）
    # ========================================================
    def run_hobbit_full(self, all_layer_traces):
        # 双独立缓存（论文分立式缓存管理）
        fp16_cache = self.LHUCache(
            self.fp16_cache_size,
            self.w_lru,
            self.w_lfu,
            self.w_lhu,
            self.w_fld,
            is_high_precision=True,
        )
        int4_cache = self.LHUCache(
            self.int4_cache_size,
            self.w_lru,
            self.w_lfu,
            self.w_lhu,
            self.w_fld,
            is_high_precision=False,
        )
        transfer_queue = self.TransferQueue(
            self.max_concurrent_transfers, self.fp16_transfer_ms, self.int4_transfer_ms
        )
        current_time = 0.0
        total_tokens = len(all_layer_traces[0]["indices"])
        stats = {
            "fp16_hit": 0,
            "fp16_miss": 0,
            "int4_hit": 0,
            "int4_miss": 0,
            "skip": 0,
        }
        layer_cosines = []

        for layer_idx in range(self.num_layers):
            fp16_cache.set_layer(layer_idx)
            int4_cache.set_layer(layer_idx)
            indices = all_layer_traces[layer_idx]["indices"]
            weights = all_layer_traces[layer_idx]["weights"]
            hidden = all_layer_traces[layer_idx]["hidden"]
            expert_w = self.expert_weights[layer_idx]

            # —— Layer 级自适应预取：预取后续层的 INT4 专家（预取低精度，预测错误代价小）——
            for p in range(1, self.prefetch_layers + 1):
                next_layer = layer_idx + p
                if next_layer < self.num_layers:
                    next_indices = all_layer_traces[next_layer]["indices"]
                    for token_idx in range(min(total_tokens, 100)):
                        expert_idx = next_indices[token_idx][0]
                        if not int4_cache.has(expert_idx):
                            transfer_queue.add(
                                expert_idx, is_fp16=False, current_time=current_time
                            )

            # 逐 token 记录走 INT4 的专家（用于精确的输出余弦对比）
            per_token_int4 = [set() for _ in range(total_tokens)]

            # —— 处理当前层 ——
            for token_idx in range(total_tokens):
                transfer_queue.process(current_time, fp16_cache, int4_cache)

                token_experts = indices[token_idx]
                token_weights = weights[token_idx]
                unimportance_scores = self.compute_unimportance_score(token_weights)

                for i, expert_idx in enumerate(token_experts):
                    score = unimportance_scores[i]

                    if score <= self.T1:
                        # 第一档：重要专家，必须使用 FP16（精度底线）
                        if fp16_cache.has(expert_idx):
                            stats["fp16_hit"] += 1
                            current_time += self.compute_latency_ms
                            fp16_cache.access(expert_idx)
                        else:
                            stats["fp16_miss"] += 1
                            current_time += self.fp16_transfer_ms
                            current_time += self.compute_latency_ms
                            fp16_cache.access(expert_idx)

                    elif score <= self.T2:
                        # 第二档：中等重要，使用 INT4 量化版本
                        per_token_int4[token_idx].add(int(expert_idx))
                        if int4_cache.has(expert_idx):
                            stats["int4_hit"] += 1
                            current_time += self.int4_compute_latency_ms
                            int4_cache.access(expert_idx)
                        else:
                            stats["int4_miss"] += 1
                            current_time += self.int4_transfer_ms
                            current_time += self.int4_compute_latency_ms
                            int4_cache.access(expert_idx)
                        # 后台异步预取 FP16，供后续层使用
                        if not fp16_cache.has(expert_idx):
                            transfer_queue.add(
                                expert_idx, is_fp16=True, current_time=current_time
                            )

                    else:
                        # 第三档：极不重要，直接跳过
                        stats["skip"] += 1

                transfer_queue.process(current_time, fp16_cache, int4_cache)

            # 测量该层输出余弦相似度：FP16 基线 vs HOBBIT 混合精度
            any_int4 = any(s for s in per_token_int4)
            layer_cos = self._compute_layer_cosine(
                hidden,
                indices,
                weights,
                expert_w,
                int4_mask_per_token=per_token_int4 if any_int4 else None,
            )
            layer_cosines.append(layer_cos)

        avg_cos = sum(layer_cosines) / len(layer_cosines)
        return self._build_result(
            "HOBBIT：混合精度方案",
            current_time,
            total_tokens,
            stats,
            int4_cos=avg_cos,
            layer_cosines=layer_cosines,
        )

    # ========================================================
    # 结果汇总
    # ========================================================
    def _build_result(
        self,
        name,
        total_latency_ms,
        total_tokens,
        stats,
        int4_cos=1.0,
        layer_cosines=None,
    ):
        total_calls = (
            stats["fp16_hit"]
            + stats["fp16_miss"]
            + stats["int4_hit"]
            + stats["int4_miss"]
            + stats["skip"]
        )
        fp16_total = stats["fp16_hit"] + stats["fp16_miss"]
        int4_total = stats["int4_hit"] + stats["int4_miss"]
        return {
            "name": name,
            "total_latency_ms": total_latency_ms,
            "avg_latency_per_token_ms": total_latency_ms
            / (total_tokens * self.num_layers),
            "throughput_tokens_per_sec": (total_tokens * self.num_layers)
            / (total_latency_ms / 1000),
            "fp16_rate": fp16_total / max(total_calls, 1),
            "int4_rate": int4_total / max(total_calls, 1),
            "skip_rate": stats["skip"] / max(total_calls, 1),
            "fp16_hit_rate": stats["fp16_hit"] / max(fp16_total, 1),
            "int4_hit_rate": stats["int4_hit"] / max(int4_total, 1),
            "int4_cos": int4_cos,
            "layer_cosines": layer_cosines or [],
            "stats": stats,
        }


# ========================================================
# 主程序
# ========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HOBBIT 混合精度专家卸载仿真系统")
    print("=" * 70)

    SIM_PARAMS = {
        "num_experts": 8,
        "top_k": 2,
        "d_model": 4096,
        "num_layers": 32,
        "fp16_cache_size": 2,
        "int4_cache_size": 6,
        "compute_latency_ms": 2,
        "int4_compute_latency_ms": 1.6,
        "fp16_transfer_ms": 20,  # 边缘设备场景，FP16 传输延迟较高
        "int4_transfer_ms": 5,  # INT4 体积为 FP16 的 1/4
        "max_concurrent_transfers": 1,
        "T1": 0.3,
        "T2": 0.9,
        "prefetch_layers": 2,
        "w_lru": 0.2,
        "w_lfu": 0.3,
        "w_lhu": 0.4,
        "w_fld": 0.1,
        "num_tokens": 500,
    }

    print("\n仿真参数：")
    for k, v in SIM_PARAMS.items():
        print(f"  {k}: {v}")

    sim = HobbitFinal(**{k: v for k, v in SIM_PARAMS.items() if k != "num_tokens"})

    print(
        f"\n生成 {SIM_PARAMS['num_layers']} 层 × {SIM_PARAMS['num_tokens']} Token "
        f"的专家访问序列..."
    )
    all_layer_traces = sim.generate_multi_layer_trace(
        num_tokens=SIM_PARAMS["num_tokens"]
    )

    print("\n运行三种方案仿真...")
    res_blocking = sim.run_baseline_blocking(all_layer_traces)
    res_int4 = sim.run_baseline_all_int4(all_layer_traces)
    res_hobbit = sim.run_hobbit_full(all_layer_traces)
    all_results = [res_blocking, res_int4, res_hobbit]

    # 打印对比表格
    print("\n" + "=" * 70)
    print("仿真结果对比")
    print("=" * 70)
    header = (
        f"{'方案':<24} | {'单Token延迟(ms)':<15} | {'吞吐量(tok/s)':<14} | "
        f"{'FP16%':<8} | {'INT4%':<8} | {'跳过%':<8} | {'FP16命中%':<11} | {'INT4命中%':<11}"
    )
    print(header)
    print("-" * len(header))
    for res in all_results:
        fp16_hit_str = (
            f"{res['fp16_hit_rate']*100:.1f}%" if res["fp16_rate"] > 0 else "-"
        )
        int4_hit_str = (
            f"{res['int4_hit_rate']*100:.1f}%" if res["int4_rate"] > 0 else "-"
        )
        print(
            f"{res['name']:<24} | {res['avg_latency_per_token_ms']:<15.3f} | "
            f"{res['throughput_tokens_per_sec']:<14.0f} | "
            f"{res['fp16_rate']*100:<7.1f}% | {res['int4_rate']*100:<7.1f}% | "
            f"{res['skip_rate']*100:<7.1f}% | "
            f"{fp16_hit_str:<11} | {int4_hit_str:<11}"
        )
    print("-" * len(header))

    # 关键指标
    speedup_vs_blocking = (
        res_blocking["total_latency_ms"] / res_hobbit["total_latency_ms"]
    )
    latency_overhead_vs_int4 = (
        (res_hobbit["avg_latency_per_token_ms"] - res_int4["avg_latency_per_token_ms"])
        / res_int4["avg_latency_per_token_ms"]
        * 100
    )

    print(f"\n性能指标：")
    print(f"  HOBBIT vs 阻塞式基线：加速比 {speedup_vs_blocking:.2f}x")
    print(f"  HOBBIT vs 全 INT4 基线：延迟 overhead {latency_overhead_vs_int4:.1f}%")
    print(
        f"  精度构成：{res_hobbit['fp16_rate']*100:.1f}% FP16 + "
        f"{res_hobbit['int4_rate']*100:.1f}% INT4 + "
        f"{res_hobbit['skip_rate']*100:.1f}% 跳过"
    )

    # MoE 层输出余弦相似度（与 FP16 基线对比）
    print(f"\nMoE 层输出余弦相似度（vs FP16 基线，越高越好）：")
    for res in [res_int4, res_hobbit]:
        if res["int4_cos"] < 1.0:
            cos_min = min(res["layer_cosines"]) if res["layer_cosines"] else 0
            cos_mean = res["int4_cos"]
            print(
                f"  {res['name']:<24} | 均值: {cos_mean:.6f} | "
                f"最低层: {cos_min:.6f}"
            )
    print(
        f"  → HOBBIT 输出余弦 {res_hobbit['int4_cos']:.6f}，"
        f"远优于全 INT4 的 {res_int4['int4_cos']:.6f}，"
        f"说明保留 Top-1 重要专家在 FP16 可大幅减少精度损失。"
    )

    # 画图
    if PLOT_AVAILABLE:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        names = [r["name"] for r in all_results]
        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

        # 图 1：单 Token 延迟
        latencies = [r["avg_latency_per_token_ms"] for r in all_results]
        bars1 = axes[0].bar(names, latencies, color=colors)
        axes[0].set_title("单层单 Token 平均延迟（越低越好）", fontsize=12)
        axes[0].set_ylabel("延迟（ms）", fontsize=10)
        axes[0].bar_label(bars1, fmt="%.3f ms", padding=3)

        # 图 2：吞吐量
        throughputs = [r["throughput_tokens_per_sec"] for r in all_results]
        bars2 = axes[1].bar(names, throughputs, color=colors)
        axes[1].set_title("吞吐量对比（越高越好）", fontsize=12)
        axes[1].set_ylabel("Tokens / 秒", fontsize=10)
        axes[1].bar_label(bars2, fmt="%.0f tok/s", padding=3)

        # 图 3：MoE 层输出余弦相似度（vs FP16 基线）
        cos_values = [res["int4_cos"] for res in all_results]
        bars3 = axes[2].bar(names, cos_values, color=colors)
        axes[2].set_title("MoE 层输出余弦相似度（vs FP16 基线）", fontsize=12)
        axes[2].set_ylabel("余弦相似度", fontsize=10)
        axes[2].set_ylim(0.9, 1.01)
        axes[2].bar_label(bars3, fmt="%.6f", padding=3)
        # 标注差异
        for i, res in enumerate(all_results):
            if res["layer_cosines"] and res["int4_cos"] < 1.0:
                cos_min = min(res["layer_cosines"])
                axes[2].annotate(
                    f"最低层: {cos_min:.4f}",
                    xy=(i, cos_min),
                    fontsize=8,
                    ha="center",
                    va="top",
                    color="red",
                )

        plt.tight_layout()
        plt.savefig("./result/fig7_hobbit_simulate.png", dpi=150, bbox_inches="tight")
        print(f"\n对比图已保存至：./result/fig7_hobbit_simulate.png")

    print("\n仿真结束。")
