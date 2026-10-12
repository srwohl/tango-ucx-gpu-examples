"""Layout and DLPack behaviour of a real tango-ucx GPU receive view (fixture device, GPU 0)."""
import sys
import warnings

import numpy as np
import pytest

import tango_ucx
from test_nodb import server  # noqa: F401

warnings.simplefilter("ignore")
cupy = pytest.importorskip("cupy")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tomography"))


class Spy:
    def __init__(self, exporter):
        self.exporter, self.calls = exporter, []

    def __dlpack_device__(self):
        return self.exporter.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        self.calls.append(kwargs)
        return self.exporter.__dlpack__(**kwargs)


@pytest.fixture
def subscription(server):
    import tango

    with tango_ucx.every(server[1], batch=4, max_wait=0.5, memory="cuda:0") as subscription:
        assert tango.DeviceProxy(server[1]).command_inout("Publish", 8) == 8
        yield subscription


def attempt(label, function):
    try:
        function()
        print(f"  {label}: ACCEPTED")
    except Exception as error:
        print(f"  {label}: REJECTED ({type(error).__name__}: {str(error).splitlines()[0][:100]})")


def test_report(subscription, capsys):
    def array_at(pointer, shape, dtype, gpu):  # verbatim from tomography/processors.py
        dtype = cupy.dtype(dtype)
        memory = cupy.cuda.UnownedMemory(pointer, int(np.prod(shape)) * dtype.itemsize, None,
                                         device_id=gpu)
        return cupy.ndarray(shape, dtype=dtype, memptr=cupy.cuda.MemoryPointer(memory, 0))

    stream = cupy.cuda.Stream(non_blocking=True)
    batch = subscription.read(timeout=5)
    with capsys.disabled(), batch.gpu_view(stream.ptr) as view, stream:
        print()
        print("A. tango-ucx GpuView -> cupy.from_dlpack")
        spy = Spy(view)
        tensor = cupy.from_dlpack(spy)
        pointers = [view.payload(i).ptr for i in range(batch.frames)]
        print(f"  CuPy asked: {spy.calls} (compute stream ptr = {stream.ptr})")
        print(f"  shape={tensor.shape} dtype={tensor.dtype} strides={tensor.strides} "
              f"C={tensor.flags.c_contiguous} F={tensor.flags.f_contiguous}")
        print(f"  same memory as the batch (no copy): {tensor.data.ptr == pointers[0]}")
        print(f"  frame spacing bytes: {sorted({b - a for a, b in zip(pointers, pointers[1:])})} "
              f"(frame is {view.payload(0).nbytes}); base % 256 = {pointers[0] % 256}")

        print("B. the pipeline's way: array_at(pointer) per frame (UnownedMemory)")
        frame = array_at(pointers[1], (view.payload(1).nbytes,), cupy.uint8, 0)
        print(f"  C={frame.flags.c_contiguous} same memory as from_dlpack row 1: "
              f"{frame.data.ptr == tensor[1].data.ptr} equal bytes: {bool((frame == tensor[1]).all())}")

        print("C. the export's own terms")
        attempt("unversioned capsule (what older consumers ask for)", lambda: view.__dlpack__())
        attempt("a different stream than the view's", lambda: view.__dlpack__(stream=1, max_version=(1, 0)))
        attempt("copy=True", lambda: view.__dlpack__(max_version=(1, 0), copy=True))
        attempt("ASTRA's request: max_version=(1, 0), no stream", lambda: view.__dlpack__(max_version=(1, 0)))

        del tensor, frame, spy
