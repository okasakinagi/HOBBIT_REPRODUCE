#!/usr/bin/env bash
# ================================================================
# run.sh — HOBBIT 服务器一键运行脚本
#
# 用法：
#   bash run.sh              # 前台运行
#   bash run.sh bg           # 后台运行（nohup）
#   bash run.sh dry          # dry-run（跳过模型加载）
#   bash run.sh download     # 预下载模型到本地（推荐先执行）
#
# 配置：
#   USE_HF_MIRROR=0 bash run.sh       # 直连 huggingface.co
#   LOCAL_MODEL_PATH=~/models/mixtral bash run.sh   # 从本地加载
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

MODEL_ID="mistralai/Mixtral-8x7B-v0.1"
LOCAL_MODEL="${LOCAL_MODEL_PATH:-$HOME/models/mixtral-8x7b}"

# --- HF-Mirror 镜像站（默认启用）---
if [ "${USE_HF_MIRROR:-1}" = "1" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    export HF_HUB_ENABLE_HF_XET=0
    echo "[run.sh] HF_ENDPOINT=$HF_ENDPOINT (Xet disabled)"
else
    echo "[run.sh] Using huggingface.co directly"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/server_${TIMESTAMP}.log"

case "${1:-fg}" in
    download)
        echo "[run.sh] Downloading $MODEL_ID to $LOCAL_MODEL ..."
        echo "[run.sh] This may take 1-2 hours for ~94GB."
        mkdir -p "$LOCAL_MODEL"

        # 用 Python 直接下载（在 import 前设 env var，彻底禁用 Xet）
        python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_XET'] = '0'
os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT:-https://hf-mirror.com}'
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$LOCAL_MODEL',
                  local_dir_use_symlinks=False, resume_download=True)
print('Download complete.')
"
        echo "[run.sh] Done. Now run: LOCAL_MODEL_PATH=$LOCAL_MODEL bash run.sh"
        ;;
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
