#!/bin/bash
# Align Python dependencies for the local ONNX -> ATC toolchain.
PYTHON_BIN="${ASCEND_ONNX_ATC_PIPELINE_PYTHON:-python3}"
PIP_TIMEOUT="${ASCEND_ONNX_ATC_PIPELINE_PIP_TIMEOUT:-120}"
PIP_RETRIES="${ASCEND_ONNX_ATC_PIPELINE_PIP_RETRIES:-10}"

pip_install() {
    "$PYTHON_BIN" -m pip install \
        --default-timeout "$PIP_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        "$@"
}

cat > /tmp/requirements.txt << 'EOF'
numpy==1.26.4
onnx==1.14.0
onnxruntime==1.16.0
onnxslim
pandas
decorator
attrs
absl-py
psutil
protobuf
sympy
cloudpickle
tornado
EOF

# ---- main ----

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "pip is not available for ${PYTHON_BIN}" >&2
    exit 1
fi

echo "=== Step 1: Core Python packages ==="
pip_install -r /tmp/requirements.txt

echo "=== Step 2: auto_optimizer wheel (build from source) ==="
(
    git clone https://gitcode.com/Ascend/msit.git -b tag_MindStudio_26.1.0.B090_001 --depth 1
    cd msit/msit
    bash install.sh --surgeon
)

echo "=== Step 3: OM runtime wheels (build from source) ==="
(
    git clone https://gitee.com/Yanguan02/tools.git --depth 1
    cd tools
    bash install.sh
)
