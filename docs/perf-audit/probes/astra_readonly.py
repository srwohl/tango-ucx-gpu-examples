"""What ASTRA 2.5 does with a DLPack tensor flagged read-only, as tango-ucx exports are."""
import warnings

import numpy as np

warnings.simplefilter("ignore")
import astra
import cupy


def attempt(label, function):
    try:
        function()
        print(f"  {label}: ACCEPTED")
    except Exception as error:
        print(f"  {label}: REJECTED ({type(error).__name__}: {str(error).splitlines()[0][:100]})")


rows, angles, columns = 6, 5, 16
theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
projector = astra.projector3d.create(dict(
    type="cuda3d", option={"GPUindex": 0},
    ProjectionGeometry=astra.create_proj_geom("parallel3d", 1.0, 1.0, rows, columns, theta),
    VolumeGeometry=astra.create_vol_geom(columns, columns, rows)))
volume = np.zeros((rows, columns, columns), np.float32)
sinogram = np.zeros((rows, angles, columns), np.float32)
attempt("writable host arrays (control)", lambda: astra.projector3d.direct_FP(projector, volume, out=sinogram))
volume.flags.writeable = False
attempt("read-only as ASTRA input", lambda: astra.projector3d.direct_FP(
    projector, volume, out=cupy.zeros((rows, angles, columns), cupy.float32)))
sinogram.flags.writeable = False
attempt("read-only as ASTRA output", lambda: astra.projector3d.direct_FP(
    projector, cupy.zeros((rows, columns, columns), cupy.float32), out=sinogram))
