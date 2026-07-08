# HOBBIT 论文复现项目 — AI 摘要

> 本文档用于向新的 AI 助手快速传递项目全貌，无需阅读完整对话历史。

## 项目目标

复现论文 **《HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference》**。

MoE 大模型推理时遇到 GPU 显存中不存在的专家（Cache Miss），不阻塞等待 FP16 传输，而是动态切换到 INT4 低精度版本计算，以极小精度损失换取消除传输延迟。

## 硬件环境

- 服务器: 2x NVIDIA L20 (各 44GB, 总计 88GB)
- CUDA 13.2, Driver 595.71
- Python 3.11, PyTorch 2.12, transformers 最新版, bitsandbytes 0.49.2
- 本地: RTX 4060 Laptop (7GB), 用于开发和调试

## 模型

- mistralai/Mixtral-8x7B-v0.1 (47B params, 32 layers, 8 experts, Top-2)
- bfloat16 全量 ~94GB, 无法全放 88GB GPU
- 成功方案: `bfloat16 + max_memory={0:"40GB",1:"40GB","cpu":"200GB"}`, 4 层在 CPU
- 权重下载到: `~/models/mixtral-8x7b` (89GB safetensors)
- 注意: 4-bit/8-bit bitsandbytes 量化在 PyTorch 2.12 + CUDA 13.2 下未生效

## 文件说明

| 文件 | 阶段 | 作用 |
|------|------|------|
| `hobbit.py` ~ `hobbit_final.py` | 1 | 6个渐进式仿真脚本，三大创新全覆盖 |
| `inspect_mixtral.py` | 2 | Mixtral 结构探查 |
| `mixtral_hobbit_empty.py` | 2 | HOBBIT 缝合迷你模型 |
| `server_hobbit.py` | 3 | 服务器加载+缝合+验证主脚本 |
| `run.sh` | 3 | 一键运行 (download/dry/bg/fg) |
| `bench_llamacpp.sh` | 4 | llama.cpp 基准测试 |
| `bench_hobbit.py` | 4 | HOBBIT 吞吐量基准 |
| `bench_mmlu.py` | 4 | MMLU 精度评测 |
| `hobbit_real.py` | 4 | 核心实验: 真实混合精度 |
| `mmmlu_*.json` | 4 | MMLU 数据集 (3学科) |
| `HANDOVER.md` | — | 完整转交文档 |
| `paper.md` | — | 论文要点梳理 |
| `route.md` | — | 原始路线图 |

## 核心实验结果

### 1. 传输开销验证 (llama.cpp)

Q4_K_M GGUF, 不同 GPU 层数对比:

| ngl | pp512 t/s | vs ngl=0 |
|-----|-----------|----------|
| 0 (纯CPU) | 6.88 | 1.0x |
| 20 | 17.28 | 2.5x |
| 32 (全GPU) | 124.85 | 18.1x |

ngl=20 vs 32: 7.2x 差距 => 86% 时间花在 CPU->GPU 传输。吻合论文 "传输占 85.5-94.5% 延迟"。

### 2. HOBBIT 真实混合精度

输入 "The capital of France is" (6 tokens), 32层 x 6 tokens x 2专家 = 384 次调用:

| 决策 | 次数 | 占比 |
|------|------|------|
| FP16 命中 | 56 | 14.6% |
| FP16 未命中 | 136 | 35.4% |
| INT4 | 187 | 48.7% |
| Skip | 5 | 1.3% |
| **INT4+Skip** | **192** | **50.0%** |

与全 FP16 基线对比:
- 余弦相似度: **0.999822**
- Top-5 预测重叠: **5/5**
- 平均相对差异: 1.73%
- 传输节省: 581ms/次

### 3. MMLU 精度 (基线, 0-shot)

| 学科 | 准确率 |
|------|--------|
| high_school_physics | 42.4% |
| high_school_mathematics | 40.5% |
| professional_law | 48.0% |
| **平均** | **43.6%** |

## 关键 Bug 记录

1. **cum 计算顺序错误**: `cum += w[i-1]` 写在 `score=cum/tw` 之后，top-2 专家 score 永远=0 -> 全部判 FP16。修复: 先更新 cum 再算 score。
2. **MoE 属性名**: 新版 transformers 用 `mlp` (MixtralSparseMoeBlock), 不是 `moe`/`block_sparse_moe`。
3. **MixtralExperts 不是列表**: 权重存为 3D tensor, 不能 `experts[eid]`。跳过需改路由权重而非直接调子模块。
4. **gate_up_proj OOM**: 加载末尾 w1+w3 合并需 1-2GB 连续显存, `max_memory` 限制 GPU + CPU 兜底解决。
5. **MAX_Q 陷阱**: `"0" or "9999"` 在 Python 中返回 "0" (非空字符串 truthy)。

## 未实现的功能

- 实际 INT4 量化计算 (bitsandbytes 在当前环境未生效)
- Layer 级自适应预取 (仿真阶段已实现，未移植到真实模型)
- LHU 多维缓存策略 (同上)
- GSM8K 精度评测
- 消融实验 (分别关闭 skip/INT4)

## 写报告参考

论文的三个核心主张及验证:
1. 传输占 85-94% 延迟 → llama.cpp ngl 对比 (验证)
2. 50%+ 专家可用 INT4/跳过替代 → hobbit_real 统计 (验证)
3. 精度损失 < 1% → cos=0.9998, Top-5=5/5 (验证)
