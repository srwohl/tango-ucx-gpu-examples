"""Local experiment: inject persistent native FBP into embedded pipeline Python."""
import atexit
import importlib.abc
import json
from pathlib import Path
import sys
import threading

PROBES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROBES))
LOCK = threading.Lock()
ENGINE = None
GEOMETRY = None
CALLS = 0


def native_fbp(sinogram, output, theta, gpu, filter_name, *, filter_cutoff=None):
    global ENGINE, GEOMETRY, CALLS
    import cupy as cp
    import numpy as np
    import reconstruction
    from fbp_native import NativeFBP, load_library

    if gpu != 0 or sinogram.shape != (128, 360, 512):
        raise ValueError("native pipeline experiment is limited to GPU0 and 128x360x512")
    reconstruction._gpu_arrays(sinogram, output, theta, gpu)
    reconstruction._fbp_options(gpu, filter_name, filter_cutoff)
    theta = np.asarray(theta, dtype=np.float32)
    geometry = (gpu, sinogram.shape[-1], theta.tobytes())
    with LOCK, cp.cuda.Device(gpu):
        if GEOMETRY != geometry:
            if ENGINE is not None:
                ENGINE.close()
            ENGINE = NativeFBP(theta, sinogram.shape[-1], load_library())
            GEOMETRY = geometry
        size, _ = reconstruction._fbp_filter(sinogram.shape[-1], filter_name, filter_cutoff)
        ENGINE.filter_rows = max(1, reconstruction.FBP_FILTER_SAMPLES // (sinogram.shape[1] * size))
        ENGINE.filter_name = filter_name
        ENGINE.cutoff = filter_cutoff
        ENGINE.run(sinogram, output, 2)
        CALLS += 1
        print(json.dumps(dict(fbp_native_shim=True, completed_calls=CALLS,
                              filter_rows=ENGINE.filter_rows)), file=sys.stderr, flush=True)


def close():
    if ENGINE is not None:
        ENGINE.close()


class NativeInjection(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "processors":
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                execute = spec.loader.exec_module

                def exec_module(module, execute=execute):
                    execute(module)
                    module.fbp_gpu = native_fbp
                spec.loader.exec_module = exec_module
                return spec
        return None


atexit.register(close)
sys.meta_path.insert(0, NativeInjection())
