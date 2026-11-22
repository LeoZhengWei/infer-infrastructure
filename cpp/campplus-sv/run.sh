#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/Users/mac/miniconda3/bin/python}"
mkdir -p cpp/lib
ln -sfn /Users/mac/miniconda3/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.32.0.dylib \
  cpp/lib/libonnxruntime.1.dylib
"$PY" python/export_onnx.py
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
"$PY" python/bench.py
