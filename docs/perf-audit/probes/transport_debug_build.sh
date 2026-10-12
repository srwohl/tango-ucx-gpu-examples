#!/usr/bin/env bash
set -euo pipefail
audit_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ucx_root=$(cd "$audit_root/../../../tango-ucx" && pwd)
environment_root="$ucx_root/.pixi/envs/gpu"
"$environment_root/bin/x86_64-conda-linux-gnu-c++" -std=c++20 -O0 -g \
    -I"$ucx_root/include" -I"$ucx_root/src/transport" -isystem "$environment_root/include" \
    -c "$ucx_root/src/transport/subscription.cpp" -o /tmp/transport-subscription-debug.o
"$environment_root/bin/x86_64-conda-linux-gnu-c++" -std=c++20 -O0 -g \
    -I"$ucx_root/include" -I"$ucx_root/src/transport" -I"$ucx_root/tests/transport" \
    -isystem "$environment_root/include" "$audit_root/probes/transport_native.cpp" \
    /tmp/transport-subscription-debug.o "$ucx_root/build-gpu/src/transport/libtango-ucx-transport.a" \
    -L"$environment_root/lib" -Wl,-rpath,"$environment_root/lib" \
    -lucp -lucs -lucm -ldl -pthread -o /tmp/transport-native-debug
