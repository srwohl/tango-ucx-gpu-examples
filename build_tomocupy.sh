#!/usr/bin/env bash
# Build TomocuPy's reconstruction modules (FourierRec, LpRec, LineRec) for the examples
# environment, into build-tomocupy/python. Its package __init__ is left out: that imports the
# file readers, OpenCV and the command line configuration, none of which the pipeline uses.
#   TOMOCUPY_CUDA_ARCH  sm_NN to compile for (default: the first GPU nvidia-smi reports)
#   TOMOCUPY_SOURCE     an existing checkout to fetch the pinned commit from
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
environment="$task_root/.pixi/envs/default"
tools="$task_root/tomocupy-tools/.pixi/envs/default"
commit=d14ece0dad8a76fb369fee0078afd139d904ee31
build="$task_root/build-tomocupy"
source="$build/src"
package="$build/python/tomocupy"
arch=${TOMOCUPY_CUDA_ARCH:-}
if [ -z "$arch" ]; then
    capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)
    [ -n "$capability" ] || { echo "no GPU found: set TOMOCUPY_CUDA_ARCH, for example sm_80" >&2; exit 1; }
    arch="sm_${capability/./}"
fi
stamp="$commit $arch"
if [ "$(cat "$build/stamp" 2>/dev/null)" = "$stamp" ]; then
    exit 0
fi

pixi install --manifest-path "$task_root/tomocupy-tools/pixi.toml"
if [ ! -d "$source/.git" ]; then
    git init -q "$source"
fi
if [ "$(git -C "$source" rev-parse HEAD 2>/dev/null)" != "$commit" ]; then
    git -C "$source" fetch -q --depth 1 "${TOMOCUPY_SOURCE:-https://github.com/tomography/tomocupy}" "$commit"
    git -C "$source" checkout -q FETCH_HEAD
fi
rm -rf "$build/python" "$build/wrap" "$build/stamp"
mkdir -p "$package/reconstruction" "$build/wrap"
: > "$package/__init__.py"
: > "$package/reconstruction/__init__.py"
cp "$source/src/tomocupy/reconstruction/"{fourierrec,fbp_filter,linerec,lprec}.py "$package/reconstruction/"
cp "$source/src/tomocupy/logging.py" "$source/LICENSE" "$package/"
python_include=$("$environment/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
for module in fourierrec filter linerec lprec; do
    for variant in "" fp16; do
        name="cfunc_$module$variant"
        "$tools/bin/swig" -python -c++ -I"$source/src/include" -outdir "$package" \
            -o "$build/wrap/$name.cxx" "$source/src/cuda/$name.i"
        "$environment/bin/nvcc" -ccbin="$environment/bin/x86_64-conda-linux-gnu-c++" -std=c++17 -O3 \
            --shared -Xcompiler=-fPIC ${variant:+-DHALF} \
            -I"$source/src/include" -I"$python_include" \
            -I"$tools/targets/x86_64-linux/include" -L"$tools/lib" -lcufft -cudart=shared \
            -gencode=arch=compute_${arch#sm_},code="$arch" -Xlinker=-rpath,"$environment/lib" \
            "$source/src/cuda/cfunc_$module.cu" "$build/wrap/$name.cxx" \
            -o "$package/_$name.so"
    done
done
PYTHONPATH="$build/python" "$environment/bin/python" -c '
from tomocupy.reconstruction import fourierrec, fbp_filter, linerec, lprec'
echo "$stamp" > "$build/stamp"
echo "TomocuPy ${commit:0:7} built for $arch in $build"
