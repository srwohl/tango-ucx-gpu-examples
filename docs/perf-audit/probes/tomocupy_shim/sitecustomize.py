"""Local experiment: inject TomocuPy filtering and backprojection into embedded pipeline Python."""
import importlib.abc
import json
import os
from pathlib import Path
import sys
import threading

PROBES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROBES))
METHOD = os.environ.get("TOMOCUPY_METHOD", "fourierrec")
CHUNK = int(os.environ.get("TOMOCUPY_CHUNK", "32"))
LOCK = threading.Lock()
ENGINE = SCRATCH = GEOMETRY = None
CALLS = 0


def tomocupy_fbp(sinogram, output, theta, gpu, filter_name, *, filter_cutoff=None):
    global ENGINE, SCRATCH, GEOMETRY, CALLS
    import cupy as cp
    import numpy as np
    import reconstruction
    from tomocupy_fbp import TomocupyFBP

    reconstruction._gpu_arrays(sinogram, output, theta, gpu)
    rows, _, columns = sinogram.shape
    if filter_cutoff is not None or rows % CHUNK or columns % 2:
        raise ValueError("TomocuPy pipeline experiment: no cutoff, whole chunks, even columns")
    theta = np.asarray(theta, dtype=np.float32)
    geometry = (gpu, columns, theta.tobytes(), filter_name)
    # The pipeline's volume convention: ASTRA's row order and scale. LpRec rejects negated
    # angles, so its rows are mirrored through owned scratch instead.
    mirror = METHOD == "lprec"
    scale = np.float32(np.pi / 4)
    with LOCK, cp.cuda.Device(gpu):
        if GEOMETRY != geometry:
            ENGINE = TomocupyFBP(METHOD, CHUNK, theta if mirror else -theta, columns, filter_name,
                                 center=(columns - 1) / 2)
            SCRATCH = cp.empty((CHUNK, columns, columns), dtype=cp.float32) if mirror else None
            GEOMETRY = geometry
        for block in reconstruction.slice_blocks(rows, CHUNK):
            if mirror:
                ENGINE.run(sinogram[block], SCRATCH)
                cp.multiply(SCRATCH[:, ::-1, :], scale, out=output[block])
            else:
                ENGINE.run(sinogram[block], output[block])
                output[block] *= scale
        # As production FBP: complete writes before returning the borrowed output slot.
        cp.cuda.get_current_stream().synchronize()
        CALLS += 1
        print(json.dumps(dict(tomocupy_shim=True, method=METHOD, chunk=CHUNK, completed_calls=CALLS)),
              file=sys.stderr, flush=True)


class TomocupyInjection(importlib.abc.MetaPathFinder):
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
                    module.fbp_gpu = tomocupy_fbp
                spec.loader.exec_module = exec_module
                return spec
        return None


sys.meta_path.insert(0, TomocupyInjection())
