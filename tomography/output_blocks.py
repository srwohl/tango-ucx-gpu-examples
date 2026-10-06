"""Single-scan, bounded host assembly of fixed-capacity GPU output publications."""
import json
import threading

import numpy as np

from reconstruction import postprocess


class OutputCollector:
    """Copy borrowed blocks once; return a reusable host volume only when complete.

    Consumers must finish using the returned array before adding the next scan,
    or explicitly copy it for asynchronous display. Transport indices are ordered;
    slice ranges can arrive in any order within the active scan.
    """

    def __init__(self, scan, settings_for_scan, *, budget_bytes=None, storage_volumes=1):
        self.scan = scan
        self.settings_for_scan = settings_for_scan
        shape = (scan["rows"], scan["columns"], scan["columns"])
        required = int(np.prod(shape)) * 4
        budget = budget_bytes if budget_bytes is not None else scan.get("buffering", {}).get(
            "output_host_budget_bytes", 1024 * 1024**2)
        if required * storage_volumes > budget:
            raise ValueError(f"host output assembly/snapshots require {required * storage_volumes} bytes, exceeds {budget}")
        self.data = np.empty(shape, np.float32)
        self.coverage = np.zeros(shape[0], bool)
        self.index = self.completed = 0
        self.identity = self.settings = None

    @classmethod
    def from_description(cls, description, settings_for_scan, **kwargs):
        scan = json.loads(description["application_text"])
        if scan.get("buffering", {}).get("output_mode", "volume") != "blocks":
            return None
        return cls(scan, settings_for_scan, **kwargs)

    def add(self, array, record):
        fields = {key: int(record[key]) for key in (
            "index", "scan_id", "calibration_id", "settings_revision", "slice_start", "slice_count")}
        if fields["index"] != self.index:
            raise ValueError("missing or out-of-order reconstruction publication")
        identity = tuple(fields[key] for key in ("scan_id", "calibration_id", "settings_revision"))
        if self.identity is None:
            if identity[:2] != (self.scan["scan_id"] + self.completed,
                                self.scan["calibration_id"] + self.completed):
                raise ValueError("missing or out-of-order reconstructed scan/calibration")
            settings = self.settings_for_scan(identity[0])
            if settings["scan_id"] != identity[0] or settings["revision"] != identity[2]:
                raise ValueError("reconstruction settings revision does not match publication")
        else:
            if identity != self.identity:
                raise ValueError("scan/settings changed before all output blocks arrived")
            settings = self.settings
        start, count = fields["slice_start"], fields["slice_count"]
        if (not 0 <= start < start + count <= self.data.shape[0] or
                array.ndim != 3 or array.shape[1:] != self.data.shape[1:] or
                not 0 < count <= array.shape[0] or array.dtype != np.float32):
            raise ValueError("invalid reconstruction block extent or array")
        if self.coverage[start:start + count].any():
            raise ValueError("duplicate or overlapping reconstruction block")
        if not np.isfinite(array[:count]).all():
            raise ValueError("nonfinite reconstruction block")
        np.copyto(self.data[start:start + count], array[:count])
        self.coverage[start:start + count] = True
        self.identity, self.settings = identity, settings
        self.index += 1
        if not self.coverage.all():
            return None
        postprocess(self.data, settings["options"])
        result = (self.data, identity[0], self.completed, settings)
        self.completed += 1
        self.coverage.fill(False)
        self.identity = self.settings = None
        return result

    def finish(self):
        if self.identity is not None:
            raise ValueError("incomplete reconstruction output at End")


def read_complete(sub, collector, *, timeout=0.1):
    """Never return receive-ring-backed data or application records."""
    batch = sub.read(timeout=timeout)
    if batch is None:
        return None
    if batch.frames != 1:
        raise ValueError("expected one reconstruction publication per batch")
    return collector.add(batch.array[0], batch.records[0])


class LatestCompleted:
    """One pending private snapshot, plus at most one owned by its consumer.

    Incomplete volumes never enter this mailbox. Replacing a pending snapshot
    releases it before copying, bounding float storage to two complete volumes.
    """

    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False

    def put(self, result, health):
        with self.condition:
            if self.closed:
                raise ValueError("display mailbox is closed")
            self.pending = None
            volume, scan_id, index, settings = result
            self.pending = (volume.copy(), scan_id, index, settings, health)
            self.condition.notify()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.closed)
            result, self.pending = self.pending, None
            return result

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
