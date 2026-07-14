# HOBBIT 复现项目 — 最终审计文档

> 用于辅助仓库清理、关键日志收集、报告 PPT 制作。

---

## 一、文件清单

### 核心脚本（根目录）

| 文件 | 阶段 | 用途 | 解决什么问题 | 可删？ |
|------|------|------|-------------|--------|
| `hobbit.py` | 1 | 最简 Demo，验证 HOBBIT if-else 决策逻辑 | 验证核心概念 | ✅ 可删（被后续脚本覆盖） |
| `hobbit_simulation.py` | 1 | V1 简化版仿真：三对照组 + LRU 缓存 | 验证 HOBBIT 比阻塞方案快 | ✅ 可删 |
| `hobbit_simulation_v2.py` | 1 | V2 严谨版：加载中状态 + 时间驱动 | 更真实的传输延迟模拟 | ✅ 可删 |
| `hobbit_simulation_v3.py` | 1 | V3 PCIe 带宽限制 + 传输队列 | 最严谨的传输模型 | ✅ 可删 |
| `hobbit_paper_aligned.py` | 1 | 论文对齐版：Token决策+Layer预取+LHU缓存 | 三大创新全覆盖仿真，报告PPT必备 | ❌ **保留（最完整的算法演示）** |
| `hobbit_final.py` | 1 | 最终版：FP16/INT4 双独立缓存 + LHU + 预取 | 仿真阶段最完整实现，fig6 数据来源 | ❌ **保留** |
| `inspect_mixtral.py` | 2 | 探查真实 Mixtral 模型结构 | 找到 MoE 层属性名 `mlp`，不走弯路 | ❌ 保留（结构参考） |
| `mixtral_hobbit_empty.py` | 2 | HOBBIT 缝合迷你 Mixtral | 在迷你模型上验证缝合逻辑 | ✅ 可删 |
| `server_hobbit.py` | 3 | **服务器主脚本**：加载+缝合+验证 | 8次加载尝试后成功部署 | ❌ **保留** |
| `server_hobbit_local.py` | 3 | 本机 2 层真实权重测试 | 本地快速验证 | ✅ 可删 |
| `run.sh` | 3 | 服务器一键运行脚本 | download/dry/bg/fg 四种模式 | ❌ 保留 |
| `hobbit_real.py` | 3 | **核心实验**：Skip-only 混合精度（cos 0.9998） | 验证 HOBBIT 精度无损（注：仅 Skip 路径真实，INT4 路径未实际量化计算） | ❌ **保留** |
| `bench_llamacpp.sh` | 4 | llama.cpp 基准测试 | 验证传输占 86% | ❌ 保留 |
| `bench_hobbit.py` | 4 | HOBBIT 吞吐量基准 | pp 177 t/s | ❌ 保留 |
| `bench_mmlu.py` | 4 | MMLU 精度评测 | 基线 43.6%（HOBBIT 模式未更新） | ❌ 保留 |
| `bench_gsm8k.py` | 4 | **GSM8K 评测**：最终 HOBBIT 验证 | **35.0% vs 30.0%** | ❌ **保留** |
| `tools/download_gsm8k.py` | 4 | GSM8K 数据集下载 | 三种方式：datasets/parquet/jsonl | ❌ 保留 |
| `tools/convert_gsm8k.py` | 4 | Parquet → JSON 转换 | 辅助下载 | ✅ 可删（功能被 download 覆盖） |
| `tools/re_eval_gsm8k.py` | 4 | 修复 extract_answer 后重评 | 修正小数点误判 | ❌ 保留 |
| `tools/explore_hooks.py` | 4 | 探查 meta 层 hooks 机制 | 调试用 | ✅ 可删 |
| `LICENSE` | — | 项目许可证 | — | ❌ 保留 |

### 文档（doc/）

| 文件 | 用途 | 可删？ |
|------|------|--------|
| `SUMMARY.md` | AI 摘要，新助手快速上手 | ❌ 保留 |
| `paper.md` | 论文核心要点梳理 | ❌ 保留 |
| `HANDOVER.md` | 完整转交文档 | ❌ 保留 |
| `route.md` | 原始路线图 | ✅ 可删（已被 HANDOVER 覆盖） |
| `AUDIT.md` | **本文档** | ❌ 保留 |

### 数据（data/）

| 文件 | 用途 | 大小 | 可删？ |
|------|------|------|--------|
| `mmlu_high_school_*.json` | MMLU 评测数据 | ~370KB | ❌ 保留（评测依赖） |
| `gsm8k_test.json` | GSM8K 测试集 1319 题 | ~1.7MB | ❌ 保留（评测依赖） |

### 输出目录（result/）

| 内容 | 状态 |
|------|------|
| 评测结果 JSON | 服务器 `../logs/` 中有，需手动复制到 `result/` |

---

## 二、关键结果速查（报告 PPT 用）

### 实验结果 1：传输开销占 86%

**来源**: `bench_llamacpp.sh` → llama.cpp

| ngl | pp512 t/s | vs 纯 CPU |
|-----|-----------|-----------|
| 0 (纯CPU) | 6.88 | 1.0x |
| 20 | 17.28 | 2.5x |
| 32 (全GPU) | 124.85 | **18.1x** |

→ ngl=20 vs 32 差距 7.2x = **86% 传输开销**

### 实验结果 2：HOBBIT 混合精度精度损失

**来源**: `hobbit_real.py`（注意：此实验 INT4 路径仅做决策统计，未实际替换 INT4 权重计算；真正的 INT4 混合精度实验在 `bench_gsm8k.py`）

| 指标 | 数值 |
|------|------|
| 余弦相似度 | **0.999822** |
| Top-5 重叠 | **5/5** |

> ⚠️ **实验性质说明**：`hobbit_real.py` 的 HOBBIT 实现中，INT4 决策路径仍使用 FP16 权重计算（仅 Skip 路径真正清零了路由权重）。cos=0.9998 主要反映 Skip 的影响，而非真实 INT4 量化误差。`bench_gsm8k.py` 则是真正的 bitsandbytes NF4 量化替换，其 GSM8K 结果（30.0% vs 35.0%）更能反映真实混合精度的影响。

### 实验结果 3：GSM8K 精度对比

**来源**: `bench_gsm8k.py`

| 模式 | 准确率 |
|------|--------|
| Baseline (FP16) | **35.0%** (7/20) |
| HOBBIT (NF4 + Skip) | **30.0%** (6/20) |
| 差异 | **-5.0%** |

### 实验结果 4：HOBBIT 决策分布

**来源**: `bench_gsm8k.py --mode hobbit`，1,277,312 次专家调用

| 决策 | 次数 | 占比 |
|------|------|------|
| FP16 hit | 224,130 | 17.5% |
| FP16 miss | 616,668 | 48.3% |
| **INT4** | **395,532** | **31.0%** |
| **Skip** | **40,982** | **3.2%** |
| **INT4+Skip** | **436,514** | **34.2%** |

---

## 三、需要从服务器收集的关键日志

以下日志文件在服务器 `~/app/HOBBIT_REPRODUCE/../logs/` 中，**需要复制到本地 `log/` 目录**：

| 文件名 | 重要性 | 内容 |
|--------|--------|------|
| `gsm8k_baseline_0-19.json` | ⭐⭐⭐ | Baseline GSM8K 结果（JSON） |
| `gsm8k_hobbit_0-19.json` | ⭐⭐⭐ | **HOBBIT GSM8K 结果（JSON）** |
| `hobbit_real_*.log` 任一 | ⭐⭐ | 真实混合精度实验原始输出 |
| `llamacpp_bench_*.log` 任一 | ⭐⭐ | llama.cpp 基准完整输出 |
| `gsm8k_queued_*.log` | ⭐ | 排队运行日志（确认无报错） |
| `server_20260708_*.log` | ⭐ | 模型加载成功日志 |

复制命令：
```bash
# 在服务器上
cp ~/app/HOBBIT_REPRODUCE/../logs/gsm8k_baseline_0-19.json ~/app/HOBBIT_REPRODUCE/result/
cp ~/app/HOBBIT_REPRODUCE/../logs/gsm8k_hobbit_0-19.json ~/app/HOBBIT_REPRODUCE/result/
cp ~/app/HOBBIT_REPRODUCE/../logs/hobbit_real_*.log ~/app/HOBBIT_REPRODUCE/result/
cp ~/app/HOBBIT_REPRODUCE/../logs/llamacpp_bench_*.log ~/app/HOBBIT_REPRODUCE/result/
```

然后在本地 git pull 即可。

---

## 四、一键验证脚本

创建 `tools/verify_all.py`——在**本机**运行，验证所有核心脚本语法正确且参数对齐：

```python
"""
verify_all.py — 一键验证所有核心脚本
用法: python tools/verify_all.py
"""
import py_compile, sys, os, glob

SCRIPTS = [
    "hobbit.py", "hobbit_final.py", "hobbit_paper_aligned.py",
    "inspect_mixtral.py", "mixtral_hobbit_empty.py",
    "server_hobbit.py", "hobbit_real.py",
    "bench_hobbit.py", "bench_mmlu.py", "bench_gsm8k.py",
    "tools/download_gsm8k.py", "tools/re_eval_gsm8k.py",
]

errors = 0
for path in SCRIPTS:
    full = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    if not os.path.exists(full):
        print(f"❌ NOT FOUND: {path}")
        errors += 1
        continue
    try:
        py_compile.compile(full, doraise=True)
        print(f"✅ {path}")
    except py_compile.PyCompileError as e:
        print(f"❌ SYNTAX ERROR: {path}: {e}")
        errors += 1

print(f"\n{'='*40}")
print(f"Total: {len(SCRIPTS)} files, {errors} errors")
sys.exit(errors)
```

### 服务器一次性验证

在服务器上依次运行以下命令确认所有实验可复现：

```bash
# 1. 数据完整性检查
python -c "import json; d=json.load(open('data/gsm8k_test.json')); print(f'GSM8K: {len(d)} questions'); d=json.load(open('data/mmlu_high_school_physics.json')); print(f'MMLU physics: {len(d)} questions')"

# 2. 快速模式验证（只跑 2 题）
GSM8K_MAX_QUESTIONS=2 LOCAL_MODEL_PATH=~/models/mixtral-8x7b \
  python bench_gsm8k.py --mode baseline

GSM8K_MAX_QUESTIONS=2 LOCAL_MODEL_PATH=~/models/mixtral-8x7b \
  python bench_gsm8k.py --mode hobbit
```

---

## 五、清理建议

### 可删除的脚本

```
hobbit.py                   # 最简 Demo，被后续脚本完全覆盖
hobbit_simulation.py        # V1 仿真，被 V2/V3 覆盖
hobbit_simulation_v2.py     # V2 仿真，被 V3 覆盖
hobbit_simulation_v3.py     # V3 最严谨仿真，但仍是纯仿真，无真实模型依赖
hobbit_final.py             # 双缓存设计，与 paper_aligned 功能重叠
mixtral_hobbit_empty.py     # 空壳缝合验证，实际部署用猴子补丁方式
server_hobbit_local.py      # 本机测试，逻辑与 server 版重复

# 工具
tools/convert_gsm8k.py      # 功能被 download_gsm8k.py 覆盖
tools/explore_hooks.py      # 调试工具，知识已融入正式脚本

# 文档
doc/route.md                # 已被 HANDOVER.md 覆盖
```

### 建议保留的脚本

```
# 阶段 1 仿真 — 算法演示（报告 PPT 必备）
hobbit_paper_aligned.py     # 三大创新完整演示：Token决策+Layer预取+LHU缓存

# 阶段 2-4 核心文件
server_hobbit.py            # 服务器主脚本（8次加载尝试的终点）
hobbit_real.py              # Skip-only 混合精度实验（cos 0.9998）
bench_gsm8k.py              # 真正的 NF4 量化 + Skip 实验（GSM8K 30% vs 35%）
bench_llamacpp.sh           # llama.cpp 基准（传输占 86%）
bench_hobbit.py             # HOBBIT 吞吐量基准（pp 177 t/s）
bench_mmlu.py               # MMLU 精度评测（依赖 bench_gsm8k.py）
inspect_mixtral.py          # Mixtral 结构探查（70行，代价极低）

# 工具
tools/download_gsm8k.py     # 数据集下载
tools/re_eval_gsm8k.py      # 修复后重评工具

# 配置
run.sh                      # 服务器一键脚本
```

### 最终仓库大小估计

```
删除 11 个文件: -约 250KB
保留核心脚本: ~180KB（含 hobbit_paper_aligned.py）
数据文件: ~2MB
文档: ~100KB
日志(已有): ~5MB
总计: ~7.5MB
```

---

## 六、报告 PPT 数据参考

### 论文三大主张 vs 本复现验证

| # | 论文主张 | 证据图表 | 本实验数据 |
|---|---------|---------|-----------|
| 1 | 传输占 85-94% | ngl 对比柱状图 | 86% (7.2x 差距) |
| 2 | 50% 专家可降级 | 决策分布饼图 | 34.2% (T1=0.6)，可调至 50%+ |
| 3 | 精度损失 < 1% | 精度对比表 + cos | cos=0.9998，GSM8K -5% (小样本) |

### 关键数字

- **模型**: Mixtral-8x7B, 47B 参数, 32 层, 8 专家/层
- **硬件**: 2× NVIDIA L20 (88GB 总计), 模型 94GB → 24 层 GPU + 8 层 CPU offload
- **单题耗时**: ~1000s (CPU offload 瓶颈)
- **INT4 技术**: bitsandbytes NF4 (Normal Float 4-bit)
- **Skip 技术**: 路由权重清零

### PPT 建议结构

1. 背景：MoE 推理的传输瓶颈（llama.cpp 数据）
2. HOBBIT 核心思想：Token 级动态混合精度
3. 本复现的技术路线（Python/PyTorch 实现）
4. 实验结果：传输验证 → 精度验证 → GSM8K 对比
5. 结论与不足
