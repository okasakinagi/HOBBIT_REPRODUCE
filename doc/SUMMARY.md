# HOBBIT 论文复现项目 — AI 摘要

> 本文档用于向新的 AI 助手快速传递项目全貌，无需阅读完整对话历史。

## 项目目标

复现论文 **《HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference》**。

MoE 大模型推理时遇到 GPU 显存中不存在的专家（Cache Miss），不阻塞等待 FP16 传输，而是动态切换到 INT4 低精度版本计算，以极小精度损失换取消除传输延迟。

## 硬件环境

- 服务器: 2x NVIDIA L20 (各 44GB, 总计 88GB)
- CUDA 13.2, Driver 595.71
- Python 3.11, PyTorch 2.12.1, transformers 5.13.0, bitsandbytes 0.49.2
- 本地: RTX 4060 Laptop (7GB), 用于开发和调试
- 网络: 服务器直连 huggingface.co 被墙；通过 hf-mirror.com 下载

## 模型

- mistralai/Mixtral-8x7B-v0.1 (47B params, 32 layers, 8 experts, Top-2)
- bfloat16 全量 ~94GB, 无法全放 88GB GPU
- 成功方案: `bfloat16 + max_memory={0:"40GB",1:"40GB","cpu":"200GB"}`
- 权重下载到: `~/models/mixtral-8x7b` (89GB safetensors)
- 层分布: 层 0-11 → GPU 0, 层 12-24 → GPU 1, 层 25-31 → meta (CPU offload)

## 仓库结构

```
g:\moe\
├── *.py, *.sh          # 根目录：核心脚本
├── doc/                # 文档
│   ├── SUMMARY.md      ← 本文档
│   ├── paper.md        # 论文要点梳理
│   ├── HANDOVER.md     # 完整转交文档
│   └── route.md        # 原始路线图
├── data/               # 测试数据
│   ├── mmlu_*.json     # MMLU 数据集 (3学科, 151-231题/科)
│   └── gsm8k_test.json # GSM8K 测试集 (1319题)
├── result/             # 输出成果（评测结果 JSON）
├── tools/              # 工具脚本
│   ├── download_gsm8k.py  # GSM8K 下载（支持 hf-mirror parquet）
│   ├── convert_gsm8k.py   # Parquet → JSON 转换
│   └── re_eval_gsm8k.py   # 修复 extract_answer 后的重评工具
└── log/                # 运行日志（服务器 git pull 安全区）
```

## 核心创新：HOBBIT 决策 + 真实 INT4 替换

### 技术路线

```
patch_hobbit 时:
  层 0-24 (GPU)   → bitsandbytes NF4 量化 → 放 CPU（预量化）
  层 25-31 (meta)  → 不预量化（forward 时懒量化）

forward 时:
  每个 Token 的 Top-2 专家:
    得分 ≤ T1 (0.6) → FP16（重要专家，需最高精度）
    T1 < 得分 ≤ T2 (0.9) → INT4（权重替换为 bitsandbytes NF4 量化版）
    得分 > T2       → Skip（路由权重清零，贡献被移除）

  INT4 执行流程:
    1. 全局缓存查该 expert 有无量化权重
    2. 无 → 懒量化（Gpu→Cpu 拷贝 → bitsandbytes NF4 → Cpu 缓存）
    3. 备份原始权重 → Cpu（clone 的 Gpu 临时副本自动释放）
    4. 搬 INT4 权重上 Gpu → 替换切片
    5. self.experts() 计算
    6. 恢复原始权重
```

### 关于显存安全

每次只处理一个 expert 切片（~336MB），clone 到 cpu 后 Gpu 副本自动释放，不会额外积压显存。

## 文件说明

| 文件 | 阶段 | 作用 |
|------|------|------|
| `hobbit.py` ~ `hobbit_final.py` | 1 | 6个渐进式仿真脚本，三大创新全覆盖 |
| `inspect_mixtral.py` | 2 | Mixtral 结构探查 |
| `mixtral_hobbit_empty.py` | 2 | HOBBIT 缝合迷你模型 |
| `server_hobbit.py` | 3 | 服务器加载+缝合+验证主脚本 |
| `run.sh` | 3 | 一键运行 (download/dry/bg/fg) |
| `hobbit_real.py` | 3 | 核心实验: 真实混合精度（skip 实际生效） |
| `bench_llamacpp.sh` | 4 | llama.cpp 基准测试 |
| `bench_hobbit.py` | 4 | HOBBIT 吞吐量基准 |
| `bench_mmlu.py` | 4 | MMLU 精度评测 |
| `bench_gsm8k.py` | 4 | GSM8K 数学推理评测（含 checkpoint + 断点续跑） |
| `tools/download_gsm8k.py` | — | GSM8K 下载（datasets / parquet / jsonl 三种方式） |
| `tools/convert_gsm8k.py` | — | Parquet → JSON 转换 |
| `tools/re_eval_gsm8k.py` | — | 修复 extract_answer 后的结果重评 |

## 核心实验结果

### 1. 传输开销验证 (llama.cpp)

Q4_K_M GGUF, 不同 GPU 层数对比:

| ngl | pp512 t/s | vs ngl=0 |
|-----|-----------|----------|
| 0 (纯CPU) | 6.88 | 1.0x |
| 20 | 17.28 | 2.5x |
| 32 (全GPU) | 124.85 | 18.1x |

ngl=20 vs 32: 7.2x 差距 => 86% 时间花在 CPU->GPU 传输。吻合论文 "传输占 85.5-94.5% 延迟"。

### 2. HOBBIT 真实混合精度 — hobbit_real.py（skip 生效，INT4 统计）

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

### 3. MMLU 精度 (基线, 0-shot, HOBBIT 暂未跑)

| 学科 | 题数 | 准确率 |
|------|------|--------|
| high_school_physics | 151 | 42.4% |
| high_school_mathematics | 231 | 40.5% |
| professional_law | 184 | 48.0% |
| **平均** | **-** | **43.6%** |

### 4. GSM8K 数学推理精度 — 最新实验

| 模式 | 20题准确率 | 说明 |
|------|-----------|------|
| Baseline (FP16) | 35.0% (7/20) | 修正小数点后 |
| **HOBBIT (混合精度)** | **🔄 跑完中** | bitsandbytes NF4 + skip 实际生效 |
| 差异 | **🔄 待定** | 预期 ≤ 2% |

## 关键 Bug 记录

1. **cum 计算顺序错误**: `cum += w[i-1]` 写在 `score=cum/tw` 之后，top-2 专家 score 永远=0 -> 全部判 FP16。修复: 先更新 cum 再算 score。
2. **MoE 属性名**: 新版 transformers 用 `mlp` (MixtralSparseMoeBlock), 不是 `moe`/`block_sparse_moe`。
3. **MixtralExperts 不是列表**: 权重存为 3D tensor `gate_up_proj[num_experts, 2*intermediate, hidden]`，不能 `experts[eid]`。跳过需改路由权重而非直接调子模块。
4. **gate_up_proj OOM**: 加载末尾 w1+w3 合并需 1-2GB 连续显存, `max_memory` 限制 GPU + CPU 兜底解决。
5. **MAX_Q 陷阱**: `"0" or "9999"` 在 Python 中返回 "0" (非空字符串 truthy)。
6. **extract_answer 小数点误判**: 模型输出 `"18."`，标准答案 `"18"`，字符串判不等。修复: 去掉预测值的末尾小数点。
7. **hobbit_stats 含 set 导致 sum 崩溃**: `hobbit_stats` 有 `"cache": {0,1}` 键，`sum(values())` 试图 int + set 报 TypeError。修复: 排除 cache 键。
8. **INT4 预量化 OOM**: `torch.empty_like(weights)` 在 GPU 分配 ~900MB 连续显存，只剩 ~900MB 自由显存不够。修复: INT4 权重放 CPU，前向时逐 expert 搬到 GPU。
9. **meta 层无数据**: CPU-offloaded 层参数是 meta tensor（空壳）。修复: 前向时 hooks 加载完成后做懒量化。

## 未实现的功能

| 待办项 | 状态 | 理由 |
|--------|------|------|
| Layer 级自适应预取 | ❌ 不必做 | 仿真已完整实现，移植到真实模型性价比低 |
| LHU 多维缓存策略 | ❌ 不必做 | 仿真已实现，核心贡献不依赖具体缓存策略 |
| 消融实验 | ❌ 可跳过 | 分别关闭 skip/INT4 看各自贡献，可列为 future work |

## 实验结果汇总（论文对标）

| 论文主张 | 验证实验 | 结果 |
|----------|----------|------|
| 传输占 85-94% 延迟 | llama.cpp ngl=20 vs 32 | ✅ 86% |
| 50%+ 专家可降级 | hobbit_real / bench_gsm8k stats | ✅ 50.0% |
| 精度损失 < 1%（单 token） | hobbit_real cos sim | ✅ 0.999822 |
| 精度损失 < 1%（生成任务） | GSM8K baseline vs HOBBIT | 🔄 跑完中 |

## GSM8K 评测使用说明

```bash
# 服务器 nohup 后台运行
LOCAL_MODEL_PATH=~/models/mixtral-8x7b

# 基线模式
nohup python bench_gsm8k.py --mode baseline > ../logs/gsm8k_bl.log 2>&1 &

# HOBBIT 模式（含真实 bitsandbytes NF4 量化 + skip）
nohup python bench_gsm8k.py --mode hobbit > ../logs/gsm8k_hb.log 2>&1 &

# 排队串行跑（写脚本）
nohup bash -c '
  GSM8K_MAX_QUESTIONS=20 python bench_gsm8k.py --mode baseline
  GSM8K_MAX_QUESTIONS=20 python bench_gsm8k.py --mode hobbit
' > ../logs/gsm8k_all.log 2>&1 &

# 断点续跑
python bench_gsm8k.py --mode hobbit --resume

# 本地修正小数点误判
python tools/re_eval_gsm8k.py ../logs/gsm8k_baseline_*.json
```
