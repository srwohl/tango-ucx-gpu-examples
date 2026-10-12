"""Volumes from several reconstruction devices, read as one stream.

A pull set gives each scan to a reconstructor that has room for it, so every device
publishes an increasing subset of the scans and no rule says which.
"""


class Volumes:
    def __init__(self, subscriptions, scan_id):
        self.subscriptions = list(subscriptions)
        self.scan_id = scan_id
        self.last = [None] * len(self.subscriptions)
        self.outcomes = [None] * len(self.subscriptions)

    def _read(self, worker, timeout):
        """One volume copy and its scan identity; no ring-backed array escapes this call."""
        sub = self.subscriptions[worker]
        outcome = sub.outcome
        batch = sub.read(timeout=timeout)
        if batch is None:
            # Only a read that follows the outcome shows that nothing was left to deliver.
            self.outcomes[worker] = outcome
            return None
        if batch.frames != 1:
            raise ValueError("expected one reconstructed volume per frame")
        record = batch.records[-1]
        scan_id = int(record["scan_id"])
        if scan_id < self.scan_id or (self.last[worker] is not None and scan_id <= self.last[worker]):
            raise ValueError("reconstructed scan identity did not advance")
        self.last[worker] = scan_id
        return batch.array[-1].copy(), scan_id, int(record["calibration_id"])

    @property
    def outcome(self):
        """A failure at once; otherwise "end" after every device has delivered its last volume."""
        failed = [outcome for outcome in self.outcomes if outcome not in (None, "end")]
        if failed:
            return failed[0]
        return None if None in self.outcomes else "end"

    def health(self):
        reports = [sub.health() for sub in self.subscriptions]
        return dict(reports[0], skipped=sum(report["skipped"] for report in reports))

    def close(self):
        for sub in self.subscriptions:
            sub.close()


class OrderedVolumes(Volumes):
    """Every volume, in scan order. One volume per device may wait for an earlier scan.

    A device with a waiting volume is not read again, so it applies pressure instead of
    accumulating volumes here.
    """

    def __init__(self, subscriptions, scan_id, calibration_id):
        super().__init__(subscriptions, scan_id)
        self.calibration_id = calibration_id
        self.completed = 0
        self.waiting = {}

    def read(self, timeout=0.05):
        """(volume, device index) of the next scan, or None."""
        share = timeout / len(self.subscriptions)
        for worker in range(len(self.subscriptions)):
            if worker in [held for _, held in self.waiting.values()]:
                continue
            result = self._read(worker, share)
            if result is None:
                continue
            volume, scan_id, calibration_id = result
            offset = scan_id - self.scan_id
            if offset < self.completed or offset in self.waiting or calibration_id != self.calibration_id + offset:
                raise ValueError("duplicate reconstructed scan or calibration mismatch")
            self.waiting[offset] = (volume, worker)
        if self.completed in self.waiting:
            self.completed += 1
            return self.waiting.pop(self.completed - 1)
        held = [worker for _, worker in self.waiting.values()]
        others = [outcome for worker, outcome in enumerate(self.outcomes) if worker not in held]
        if self.waiting and None not in others:
            # No device that is still read can deliver the scan the waiting volumes follow.
            raise ValueError(f"missing reconstructed scan; devices ended with {others}")
        return None

    @property
    def outcome(self):
        return None if self.waiting else super().outcome


class NewestVolumes(Volumes):
    """Latest delivery: the newest scan any device offers; older arrivals are skipped."""

    def __init__(self, subscriptions, scan_id):
        super().__init__(subscriptions, scan_id)
        self.shown = None

    def read(self, timeout=0.1):
        """(volume, scan_id, scans since the first, device index), or None."""
        share = timeout / len(self.subscriptions)
        newest = None
        for worker in range(len(self.subscriptions)):
            result = self._read(worker, share)
            if result is not None and (self.shown is None or result[1] > self.shown) and (
                    newest is None or result[1] > newest[1]):
                newest = (result[0], result[1], result[1] - self.scan_id, worker)
        if newest is not None:
            self.shown = newest[1]
        return newest
