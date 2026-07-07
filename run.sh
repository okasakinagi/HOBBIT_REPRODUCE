#!/usr/bin/env bash
# ================================================================
# run.sh — HOBBIT 服务器一键运行脚本
#
# 用法：
#   bash run.sh           # 前台运行（输出到终端 + 日志文件）
#   bash run.sh bg        # 后台运行（nohup）
#   bash run.sh dry       # 本地 dry-run（跳过模型加载，验证环境）
#
# 配置 HF-Mirror（国内服务器必备，默认启用）：
#   USE_HF_MIRROR=0 bash run.sh    # 直连 huggingface.co
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

# --- HF-Mirror 镜像站（默认启用）---
if [ "${USE_HF_MIRROR:-1}" = "1" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "[run.sh] HF_ENDPOINT=$HF_ENDPOINT"
else
    echo "[run.sh] Using huggingface.co directly"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/server_${TIMESTAMP}.log"

case "${1:-fg}" in
    dry)
        echo "[run.sh] Dry-run mode — skip model loading"
        SKIP_MODEL_LOAD=1 python server_hobbit.py
        ;;
    bg)
        echo "[run.sh] Background mode — log: $LOG_FILE"
        nohup python server_hobbit.py > "$LOG_FILE" 2>&1 &
        echo "[run.sh] PID: $!"
        echo "[run.sh] Monitor: tail -f $LOG_FILE"
        ;;
    fg|*)
        echo "[run.sh] Foreground mode — log: $LOG_FILE"
        python server_hobbit.py 2>&1 | tee "$LOG_FILE"
        ;;
esac
