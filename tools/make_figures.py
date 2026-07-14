"""
make_figures.py — HOBBIT 复现项目报告图表生成
==============================================
一次性生成报告 PPT 所需全部图表，输出到 result/ 目录。

用法：
    python tools/make_figures.py

输出：
    result/fig1_llamacpp_bench.png       llama.cpp 传输开销基准
    result/fig2_gsm8k_accuracy.png       GSM8K 精度对比
    result/fig3_decision_pie.png         HOBBIT 决策分布饼图
    result/fig4_mmlu.png                 MMLU 精度对比（Baseline vs HOBBIT）
    result/fig5_throughput.png           HOBBIT vs llama.cpp 吞吐量
    result/fig6_simulation.png           hobbit_final.py 仿真结果
"""

import os, sys, json, math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# 中文字体配置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 硬编码数据（来源见各函数注释）
# ============================================================


def fig1_llamacpp_bench():
    """图1: llama.cpp 基准 — GPU 层数对吞吐量的影响"""
    ngl_labels = ["0\n(纯CPU)", "10", "20", "32\n(全GPU)"]
    pp512 = [6.88, 10.01, 17.28, 124.85]
    tg128 = [6.92, 9.90, 16.18, 74.03]
    colors_pp = ["#d4a574", "#e8b86d", "#f0c75e", "#4ecdc4"]
    colors_tg = ["#c49564", "#d9a95d", "#e0b74e", "#45b7d1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(range(4), pp512, color=colors_pp, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, pp512):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(ngl_labels, fontsize=12)
    ax1.set_ylabel("Prompt Processing (tokens/s)", fontsize=12)
    ax1.set_title("pp512 吞吐量", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 148)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    bars2 = ax2.bar(range(4), tg128, color=colors_tg, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars2, tg128):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(ngl_labels, fontsize=12)
    ax2.set_ylabel("Token Generation (tokens/s)", fontsize=12)
    ax2.set_title("tg128 吞吐量", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 90)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        "llama.cpp 基准：GPU 层数对吞吐量的影响（Mixtral-8x7B, 2× L20）",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig1_llamacpp_bench.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {path}")


def fig2_gsm8k_accuracy():
    """图2: GSM8K 精度对比"""
    modes = ["Baseline\n(全 FP16)", "HOBBIT\n(NF4 + Skip)"]
    acc = [35.0, 30.0]
    correct = [7, 6]
    wrong = [13, 14]
    colors = ["#3498db", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    bars = ax1.bar(
        modes, acc, color=colors, width=0.45, edgecolor="white", linewidth=1.2
    )
    for bar, val in zip(bars, acc):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=20,
            fontweight="bold",
        )
    ax1.set_ylabel("准确率 (%)", fontsize=12)
    ax1.set_title("准确率（20 题）", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 44)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    bar_w = 0.45
    ax2.bar(
        modes,
        correct,
        bar_w,
        color=["#2ecc71", "#27ae60"],
        label="正确",
        edgecolor="white",
        linewidth=1.2,
    )
    ax2.bar(
        modes,
        wrong,
        bar_w,
        bottom=correct,
        color=["#ecf0f1", "#e0e0e0"],
        label="错误",
        edgecolor="white",
        linewidth=1.2,
    )
    for i, (c, w) in enumerate(zip(correct, wrong)):
        ax2.text(
            i, c / 2, str(c), ha="center", fontsize=20, fontweight="bold", color="white"
        )
        ax2.text(
            i,
            c + w / 2,
            str(w),
            ha="center",
            fontsize=20,
            fontweight="bold",
            color="#7f8c8d",
        )
    ax2.set_ylabel("题目数量", fontsize=12)
    ax2.set_title("正确 / 错误分布", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="upper right")
    ax2.set_ylim(0, 22)

    plt.suptitle("GSM8K 数学推理评测", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig2_gsm8k_accuracy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {path}")


def fig3_decision_pie():
    """图3: HOBBIT 决策分布"""
    labels = [
        "FP16 命中\n(17.5%)",
        "FP16 未命中\n(48.3%)",
        "INT4 替换\n(31.0%)",
        "Skip 跳过\n(3.2%)",
    ]
    sizes = [224130, 616668, 395532, 40982]
    total = sum(sizes)
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    explode = (0, 0, 0.08, 0.12)

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        startangle=140,
        textprops={"fontsize": 12},
        pctdistance=0.6,
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title(
        "HOBBIT 决策分布（1,277,312 次专家调用）", fontsize=15, fontweight="bold"
    )

    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig3_decision_pie.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {path}")


def fig4_mmlu():
    """图4: MMLU 精度对比（Baseline vs HOBBIT）"""
    subjects = ["高中物理", "高中数学", "专业法律", "平均"]
    baseline = [42.4, 40.5, 48.0, 43.6]
    hobbit = [43.7, 39.0, 47.5, 43.4]

    x = np.arange(len(subjects))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        baseline,
        width,
        color="#3498db",
        label="Baseline (FP16)",
        edgecolor="white",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        hobbit,
        width,
        color="#e74c3c",
        label="HOBBIT (NF4 + Skip)",
        edgecolor="white",
        linewidth=1,
    )

    for bar, val in zip(bars1, baseline):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    for bar, val in zip(bars2, hobbit):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=13)
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title("MMLU 精度对比（0-shot）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 58)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.axhline(y=25.0, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(3.45, 26, "随机猜测 25%", fontsize=9, color="#95a5a6", ha="right")

    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig4_mmlu.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {path}")


def fig5_simulation():
    """图5: hobbit_final.py 仿真 — 论文三大创新完整演示"""
    schemes = ["传统阻塞\n(全FP16)", "全INT4\n低精度", "HOBBIT\n(本方案)"]
    latency = [36.195, 5.866, 19.873]
    throughput = [28, 170, 50]
    fp16_pct = [100, 0, 50]
    int4_pct = [0, 100, 50]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars_lat = ax1.bar(
        schemes, latency, color=colors, width=0.5, edgecolor="white", linewidth=1
    )
    for bar, val in zip(bars_lat, latency):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f} ms",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylabel("单 Token 延迟 (ms)", fontsize=12, color="#2c3e50")
    ax1.set_title("推理延迟对比", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 43)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    ax1b = ax1.twinx()
    ax1b.plot(schemes, throughput, "D-", color="#8e44ad", linewidth=2.5, markersize=10)
    for i, v in enumerate(throughput):
        ax1b.text(
            i,
            v + 8,
            f"{v} tok/s",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#8e44ad",
        )
    ax1b.set_ylabel("吞吐量 (tok/s)", fontsize=12, color="#8e44ad")

    ax2.bar(
        schemes,
        fp16_pct,
        0.5,
        color="#2ecc71",
        label="FP16",
        edgecolor="white",
        linewidth=1,
    )
    ax2.bar(
        schemes,
        int4_pct,
        0.5,
        bottom=fp16_pct,
        color="#f39c12",
        label="INT4",
        edgecolor="white",
        linewidth=1,
    )
    for i, (fp, i4) in enumerate(zip(fp16_pct, int4_pct)):
        if fp > 0:
            ax2.text(
                i,
                fp / 2,
                f"{fp}%",
                ha="center",
                fontsize=14,
                fontweight="bold",
                color="white",
            )
        if i4 > 0:
            ax2.text(
                i,
                fp + i4 / 2,
                f"{i4}%",
                ha="center",
                fontsize=14,
                fontweight="bold",
                color="white",
            )
    ax2.set_ylabel("专家调用占比 (%)", fontsize=12)
    ax2.set_title("精度构成（FP16 vs INT4）", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11, loc="upper right")
    ax2.set_ylim(0, 110)

    plt.suptitle(
        "hobbit_final.py 仿真 — 双缓存 + LHU + 动态决策 + 层间预取（1.82× 加速）",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig5_simulation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HOBBIT 复现项目 — 报告图表生成")
    print(f"输出目录: {RESULT_DIR}")
    print("=" * 60)

    fig1_llamacpp_bench()
    fig2_gsm8k_accuracy()
    fig3_decision_pie()
    fig4_mmlu()
    fig5_simulation()

    print(f"\n{'=' * 60}")
    print(f"全部 5 张图表已生成到 result/")
    print(f"文件列表:")
    for f in sorted(os.listdir(RESULT_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(RESULT_DIR, f)) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")
    print(f"{'=' * 60}")
