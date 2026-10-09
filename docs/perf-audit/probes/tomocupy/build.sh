#!/usr/bin/env bash
# Build TomocuPy's native modules for the examples environment, without its package __init__.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
examples=$(cd "$here/../../../.." && pwd)
environment="$examples/.pixi/envs/default"
commit=d14ece0dad8a76fb369fee0078afd139d904ee31
source="$here/build/tomocupy-src"
package="$here/build/tomocupy"
modules=${TOMOCUPY_MODULES:-"fourierrec filter linerec lprec"}

pixi install --manifest-path "$here/pixi.toml"
tools="$here/.pixi/envs/default"
if [ ! -d "$source/.git" ]; then
    git init -q "$source"
    git -C "$source" remote add origin https://github.com/tomography/tomocupy
fi
if [ "$(git -C "$source" rev-parse HEAD 2>/dev/null)" != "$commit" ]; then
    git -C "$source" fetch -q --depth 1 origin "$commit"
    git -C "$source" checkout -q FETCH_HEAD
fi
mkdir -p "$package/reconstruction" "$here/build/wrap"
# The upstream package __init__ imports its file readers, OpenCV and the CLI configuration.
: > "$package/__init__.py"
: > "$package/reconstruction/__init__.py"
cp "$source/src/tomocupy/reconstruction/"{fourierrec,fbp_filter,linerec,lprec}.py "$package/reconstruction/"
cp "$source/src/tomocupy/logging.py" "$package/"
python_include=$("$environment/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
for module in $modules; do
    for variant in "" fp16; do
        name="cfunc_$module$variant"
        "$tools/bin/swig" -python -c++ -I"$source/src/include" -outdir "$package" \
            -o "$here/build/wrap/$name.cxx" "$source/src/cuda/$name.i"
        "$environment/bin/nvcc" -ccbin="$environment/bin/x86_64-conda-linux-gnu-c++" -std=c++17 -O3 --shared -Xcompiler=-fPIC ${variant:+-DHALF} \
            -I"$source/src/include" -I"$python_include" \
            -I"$tools/targets/x86_64-linux/include" -L"$tools/lib" -lcufft -cudart=shared \
            -gencode=arch=compute_75,code=sm_75 -Xlinker=-rpath,"$environment/lib" \
            "$source/src/cuda/cfunc_$module.cu" "$here/build/wrap/$name.cxx" \
            -o "$package/_$name.so"
    done
done
PYTHONPATH="$here/build" "$environment/bin/python" -c '
from tomocupy.reconstruction import fourierrec, fbp_filter, linerec, lprec
print("tomocupy native modules import")'
