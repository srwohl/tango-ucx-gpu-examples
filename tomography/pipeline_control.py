"""File-backed pipeline settings shared by the live viewer and demo supervisor."""
import json
import math
import os
from pathlib import Path
import tempfile
import threading


CHOICES = {
    "network": ("tcp", "auto", "rdma"),
    "processing_mode": ("scalar", "batched"),
    "sinogram_memory": ("gpu", "host"),
    "output_mode": ("volume", "blocks"),
}
BATCH_KEYS = {"transport_batch", "processing_batch"}
GPU_KEYS = {"gpu", "decompress_gpu", "correct_gpu", "reconstruct_gpu"}
MEMORY_KEYS = {"host_buffer_mib", "pinned_buffer_mib", "output_host_mib"}
FLOAT_KEYS = {"receive_budget_mib", "scan_period"}
GEOMETRY_KEYS = {"pixels", "slices", "angles"}
OPTION_KEYS = set(CHOICES) | BATCH_KEYS | GPU_KEYS | MEMORY_KEYS | FLOAT_KEYS | GEOMETRY_KEYS | {"net_devices"}


class PipelineConflict(RuntimeError):
    """A transition is already pending, or this acquisition no longer accepts edits."""


def validate_options(options, gpu_count=None):
    """Validate a partial or complete options object without supplying defaults."""
    if not isinstance(options, dict) or not options:
        raise ValueError("expected a nonempty pipeline options object")
    unknown = set(options) - OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown pipeline settings: {', '.join(sorted(unknown))}")
    result = dict(options)
    for key, value in result.items():
        if key in CHOICES:
            if not isinstance(value, str) or value not in CHOICES[key]:
                raise ValueError(f"{key} must be one of {', '.join(CHOICES[key])}")
        elif key == "net_devices":
            if value is not None and (not isinstance(value, str) or not value.strip() or
                                      any(ord(c) < 32 for c in value)):
                raise ValueError("net_devices must be null or a nonempty device string")
        elif key in BATCH_KEYS | GPU_KEYS | MEMORY_KEYS | GEOMETRY_KEYS:
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            minimum = 0 if key in GPU_KEYS else 1
            maximum = 16 if key in BATCH_KEYS else None
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"{key} must be {'1..16' if maximum else f'at least {minimum}'}")
            if key in GPU_KEYS and gpu_count is not None and value >= gpu_count:
                raise ValueError(f"{key} must be below the available GPU count ({gpu_count})")
        elif key in FLOAT_KEYS:
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{key} must be a finite number")
            if value < 0 or (key == "receive_budget_mib" and value == 0):
                raise ValueError(f"{key} must be {'positive' if key == 'receive_budget_mib' else 'nonnegative'}")
    if (result.get("processing_mode") == "batched" and "processing_batch" in result and
            "transport_batch" in result and result["processing_batch"] > result["transport_batch"]):
        raise ValueError("batched decompression processing_batch must not exceed transport_batch")
    return result


def atomic_json(path, value):
    """Publish one complete JSON document using an atomic same-directory rename."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PipelineControl:
    def __init__(self, root):
        self.root = Path(root)
        self.state_path = self.root / "pipeline-control.json"
        self.request_path = self.root / "pipeline-request.json"
        self.stop_path = self.root / "pipeline-stop.json"
        self.lock = threading.Lock()

    def _settings(self):
        state = json.loads(self.state_path.read_text())
        try:
            request = json.loads(self.request_path.read_text())
        except FileNotFoundError:
            request = None
        if request is not None and request["revision"] > (state.get("active") or {}).get("revision", -1):
            state["requested"] = request
        if self.stop_path.exists() and state.get("phase") not in ("finished", "failed"):
            state["phase"] = "stopping"
            state["requested"] = state.get("active")
        return state

    def settings(self):
        with self.lock:
            return self._settings()

    def configure(self, options):
        with self.lock:
            state = self._settings()
            active = state.get("active")
            requested = state.get("requested")
            if (state.get("phase") != "running" or not active or self.request_path.exists() or
                    (requested and requested["revision"] > active["revision"])):
                raise PipelineConflict("pipeline changes require a running acquisition with no pending restart")
            updates = validate_options(options, state.get("gpu_count"))
            if state.get("fixed_geometry") and any(
                    key in updates and updates[key] != active["options"].get(key) for key in GEOMETRY_KEYS):
                raise ValueError("HDF5 detector geometry is fixed by the input data")
            merged = validate_options(dict(active["options"], **updates), state.get("gpu_count"))
            request = dict(options=merged, revision=max(active["revision"],
                           requested["revision"] if requested else -1) + 1)
            atomic_json(self.request_path, request)
            state["requested"] = request
            return state

    def stop(self):
        """Stop takes priority over any queued restart and remains usable while draining."""
        with self.lock:
            state = self._settings()
            if state.get("phase") in ("finished", "failed"):
                return state
            atomic_json(self.stop_path, dict(run_id=state.get("run_id")))
            self.request_path.unlink(missing_ok=True)
            state.update(phase="stopping", requested=state.get("active"), error=None)
            return state

    def recommend(self, options):
        """Size working buffers from the scan and draft settings, without starting work."""
        with self.lock:
            state = self._settings()
            active = state.get("active")
            if not active:
                raise PipelineConflict("current scan settings are not available yet")
            if not isinstance(options, dict):
                raise ValueError("expected a pipeline options object")
            updates = validate_options(options, state.get("gpu_count")) if options else {}
            if state.get("fixed_geometry") and any(
                    key in updates and updates[key] != active["options"].get(key) for key in GEOMETRY_KEYS):
                raise ValueError("HDF5 detector geometry is fixed by the input data")
            merged = validate_options(dict(active["options"], **updates), state.get("gpu_count"))
            workload = state.get("workload", {})
            if state.get("run_output"):
                try:
                    workload = json.loads((Path(state["run_output"]) / "status.json").read_text())["workload"]
                except (FileNotFoundError, KeyError):
                    pass
            rows, columns, angles = (merged[key] for key in ("slices", "pixels", "angles"))
            dtype = workload.get("detector_element", "u16")
            detector = rows * columns * (4 if dtype == "f32" else 2)
            corrected = rows * columns * 4
            sinogram = rows * angles * columns * 4
            volume = rows * columns**2 * 4
            recon = workload.get("reconstruction", {})
            block_rows = min(recon.get("slices_per_block", 0) or rows, rows)
            pinned = 2 * (corrected + (block_rows * columns**2 * 4 if recon.get("algorithm") == "gridrec"
                                      else block_rows * angles * columns * 4))
            compressed_bound = detector + (detector + 254) // 255 + 16
            receive = (128 << 10) + max(2 * merged["transport_batch"] * (corrected + 2048),
                                        4 * compressed_bound)
            mib = lambda size: max(1, (size + 1024**2 - 1) // 1024**2)
            merged.update(receive_budget_mib=mib(receive), host_buffer_mib=mib(sinogram),
                          pinned_buffer_mib=mib(pinned), output_host_mib=mib(volume * 3))
            information = dict(detector_shape=[rows, columns], detector_element=dtype,
                               projections_per_volume=angles, detector_frame_bytes=detector,
                               corrected_frame_bytes=corrected, sinogram_bytes=sinogram,
                               volume_shape=[rows, columns, columns], volume_bytes=volume,
                               pinned_staging_bytes=pinned)
            return dict(options=merged, information=information)
