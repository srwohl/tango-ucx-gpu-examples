#!/usr/bin/env bash
set -euo pipefail
audit_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ucx_root=$(cd "$audit_root/../../../tango-ucx" && pwd)
environment_root="$ucx_root/.pixi/envs/gpu"
cmake --build "$ucx_root/build-gpu" --target tango-ucx-transport -j2
"$environment_root/bin/x86_64-conda-linux-gnu-c++" -std=c++20 -O3 -DNDEBUG \
    -I"$ucx_root/include" -I"$ucx_root/src/transport" -I"$ucx_root/tests/transport" \
    -isystem "$environment_root/include" "$audit_root/probes/transport_native.cpp" \
    "$ucx_root/build-gpu/src/transport/libtango-ucx-transport.a" \
    -L"$environment_root/lib" -Wl,-rpath,"$environment_root/lib" \
    -lucp -lucs -lucm -ldl -pthread -o "$audit_root/probes/transport_native"
