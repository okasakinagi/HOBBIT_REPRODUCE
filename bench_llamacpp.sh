#!/usr/bin/env bash
# ================================================================
# bench_llamacpp.sh — llama.cpp 基准测试脚本
#
# 用法（在服务器上）：
#   bash bench_llamacpp.sh build    # 克隆+编译 llama.cpp
#   bash bench_llamacpp.sh convert  # 转换 safetensors → GGUF（需要 ~100GB CPU RAM + 时间）
#   bash bench_llamacpp.sh run      # 跑基准测试（不同 ngl 参数）
#   bash bench_llamacpp.sh all      # 一键全流程
#
# 前提：cmake、gcc、CUDA toolkit 已安装
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../llama.cpp"
MODEL_DIR="${HOME}/models/mixtral-8x7b"
GGUF_FILE="${MODEL_DIR}/mixtral-8x7b-v0.1.Q4_K_M.gguf"
GGUF_F16="${MODEL_DIR}/mixtral-8x7b-v0.1.f16.gguf"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCH_LOG="${LOG_DIR}/llamacpp_bench_${TIMESTAMP}.log"

# ============================================================
build_llamacpp() {
    echo "[bench] Cloning and building llama.cpp..."
    if [ -d "${LLAMA_DIR}" ]; then
        echo "[bench] llama.cpp already exists, pulling latest..."
        cd "${LLAMA_DIR}" && git pull
    else
        git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
    fi
    cd "${LLAMA_DIR}"
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j$(nproc)
    echo "[bench] Build complete. Binary at: ${LLAMA_DIR}/build/bin/"
}

# ============================================================
convert_to_gguf() {
    echo "[bench] Step 1/2: Converting HF safetensors to GGUF f16..."
    echo "[bench] RAM available: 251GB, should be enough."

    pip install -q sentencepiece

    python3 "${LLAMA_DIR}/convert_hf_to_gguf.py" \
        "${MODEL_DIR}" \
        --outfile "${GGUF_F16}" \
        --outtype f16

    echo "[bench] f16 GGUF saved: $(ls -lh ${GGUF_F16} | awk '{print $5}')"

    echo "[bench] Step 2/2: Quantizing f16 → Q4_K_M..."
    "${LLAMA_DIR}/build/bin/llama-quantize" \
        "${GGUF_F16}" "${GGUF_FILE}" Q4_K_M

    echo "[bench] Q4_K_M GGUF saved: $(ls -lh ${GGUF_FILE} | awk '{print $5}')"
    
    # 删掉 f16 中间文件省空间（可选，94GB）
    # rm "${GGUF_F16}"
    echo "[bench] f16 intermediate kept at: ${GGUF_F16}"
}

# ============================================================
run_benchmark() {
    echo "[bench] Running llama.cpp benchmarks..."
    
    if [ ! -f "${GGUF_FILE}" ]; then
        echo "[bench] GGUF not found at ${GGUF_FILE}"
        echo "[bench] Run 'bash bench_llamacpp.sh convert' first."
        exit 1
    fi

    BENCH_BIN="${LLAMA_DIR}/build/bin/llama-bench"
    if [ ! -f "${BENCH_BIN}" ]; then
        echo "[bench] llama-bench not found, building..."
        build_llamacpp
    fi

    echo "[bench] Log: ${BENCH_LOG}"

    # 测试不同 ngl（number of GPU layers）参数
    # ngl=0: 纯 CPU（基准线）
    # ngl=10,20,32: 逐步把更多层放 GPU
    for ngl in 0 10 20 32; do
        echo "" | tee -a "${BENCH_LOG}"
        echo "========== ngl=${ngl} ==========" | tee -a "${BENCH_LOG}"
        "${BENCH_BIN}" \
            -m "${GGUF_FILE}" \
            -ngl ${ngl} \
            -p 32,64,128,256,512 \
            -n 128 \
            -b 1 \
            2>&1 | tee -a "${BENCH_LOG}"
    done

    echo "" | tee -a "${BENCH_LOG}"
    echo "[bench] Done. Results: ${BENCH_LOG}"

    # 提取关键指标
    echo ""
    echo "========== Summary =========="
    grep -E "ngl=|model_init|pp|tg" "${BENCH_LOG}" | head -40
}

# ============================================================
case "${1:-run}" in
    build)
        build_llamacpp
        ;;
    convert)
        convert_to_gguf
        ;;
    run)
        run_benchmark
        ;;
    all)
        build_llamacpp
        convert_to_gguf
        run_benchmark
        ;;
    *)
        echo "Usage: bash bench_llamacpp.sh {build|convert|run|all}"
        ;;
esac
