#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_library=${TANGO_UCX_SOURCE:-"$task_root/../tango-bulk2"}
cd "$task_library"
pixi run cmake -GNinja -S . -B build -DBUILD_TESTING=ON -DTANGO_UCX_CUDA=OFF
pixi run cmake --build build
pixi run cmake --install build --prefix "$task_root/stage-host"
pixi run cmake -GNinja -S "$task_root" -B "$task_root/build-host" \
    -DCMAKE_PREFIX_PATH="$task_root/stage-host" -DEXAMPLES_CUDA=OFF \
    -DEXAMPLES_PYTHON_PREFIX="$task_root/stage-host"
pixi run cmake --build "$task_root/build-host"
pixi run ctest --test-dir "$task_root/build-host" --output-on-failure
