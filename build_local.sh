#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_library=${TANGO_UCX_SOURCE:-"$task_root/../tango-bulk2"}
# Reuse the library's resolved CUDA environment rather than mixing compilers or Python ABIs.
cd "$task_library"
pixi run -e gpu build-gpu
pixi run -e gpu cmake --install build-gpu --prefix "$task_root/stage"
pixi run -e gpu cmake -GNinja -S "$task_root" -B "$task_root/build" \
    -DCMAKE_PREFIX_PATH="$task_root/stage" -DEXAMPLES_CUDA=ON \
    -DEXAMPLES_PYTHON_PREFIX="$task_root/stage"
pixi run -e gpu cmake --build "$task_root/build"
pixi run -e gpu ctest --test-dir "$task_root/build" --output-on-failure
