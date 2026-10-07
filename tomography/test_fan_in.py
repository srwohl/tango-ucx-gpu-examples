"""Scan ordering and latest selection across the reconstruction devices of a pull set."""
from types import SimpleNamespace
import unittest

import numpy as np

from fan_in import NewestVolumes, OrderedVolumes


class Subscription:
    """Volumes of the given scans, one per read, then End."""

    def __init__(self, scans, *, calibration=40, outcome="end", frames=1):
        self.scans = list(scans)
        self.calibration, self.final, self.frames = calibration, outcome, frames
        self.outcome = None
        self.reads = 0
        self.closed = False

    def read(self, timeout=None):
        self.reads += 1
        if not self.scans:
            self.outcome = self.final
            return None
        scan_id = self.scans.pop(0)
        return SimpleNamespace(frames=self.frames, array=np.full((1, 1, 2, 2), scan_id, np.float32),
                               records=[dict(scan_id=scan_id, calibration_id=self.calibration + scan_id - 7)])

    def health(self):
        return dict(skipped=self.reads, transport="tcp")

    def close(self):
        self.closed = True


def drain(volumes, limit=50):
    results = []
    for _ in range(limit):
        result = volumes.read(timeout=0)
        if result is not None:
            results.append(result)
        elif volumes.outcome is not None:
            break
    return results


class OrderedVolumeTests(unittest.TestCase):
    def test_scans_return_in_order_with_their_device_whichever_device_took_them(self):
        subscriptions = [Subscription([8, 9, 12]), Subscription([7, 11]), Subscription([10])]
        volumes = OrderedVolumes(subscriptions, 7, 40)
        results = drain(volumes)
        self.assertEqual([(float(volume[0, 0, 0]), device) for volume, device in results],
                         [(7, 1), (8, 0), (9, 0), (10, 2), (11, 1), (12, 0)])
        self.assertEqual(volumes.outcome, "end")
        self.assertEqual(volumes.health()["transport"], "tcp")
        volumes.close()
        self.assertTrue(all(sub.closed for sub in subscriptions))

    def test_a_device_with_a_waiting_volume_is_not_read_again(self):
        ahead, behind = Subscription([8, 9, 10]), Subscription([], outcome=None)  # still reconstructing
        volumes = OrderedVolumes([ahead, behind], 7, 40)
        for _ in range(5):
            self.assertIsNone(volumes.read(timeout=0))
        self.assertEqual(ahead.reads, 1)
        self.assertIsNone(volumes.outcome)

    def test_one_device_keeps_the_single_stream_checks(self):
        volumes = OrderedVolumes([Subscription([7, 8, 9])], 7, 40)
        self.assertEqual([device for _, device in drain(volumes)], [0, 0, 0])
        for subscription, message in ((Subscription([7, 7]), "did not advance"),
                                      (Subscription([6]), "did not advance"),
                                      (Subscription([7], calibration=41), "calibration"),
                                      (Subscription([7], frames=2), "one reconstructed volume")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                drain(OrderedVolumes([subscription], 7, 40))

    def test_duplicates_gaps_and_failures_are_reported(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            drain(OrderedVolumes([Subscription([7, 8]), Subscription([8])], 7, 40))
        failed = OrderedVolumes([Subscription([7]), Subscription([], outcome="failed")], 7, 40)
        self.assertEqual(len(drain(failed)), 1)
        self.assertEqual(failed.outcome, "failed")
        # Scan 8 never arrives: the device holding scan 9 waits, and the other has ended.
        with self.assertRaisesRegex(ValueError, "missing reconstructed scan"):
            drain(OrderedVolumes([Subscription([7, 9]), Subscription([])], 7, 40))

    def test_end_needs_a_read_after_every_outcome(self):
        late = Subscription([])
        volumes = OrderedVolumes([late], 7, 40)
        self.assertIsNone(volumes.read(timeout=0))
        self.assertIsNone(volumes.outcome)
        self.assertIsNone(volumes.read(timeout=0))
        self.assertEqual(volumes.outcome, "end")


class NewestVolumeTests(unittest.TestCase):
    def test_the_newest_scan_is_shown_and_older_arrivals_are_skipped(self):
        volumes = NewestVolumes([Subscription([7, 10]), Subscription([8, 9, 13])], 7)
        shown = [(scan_id, offset, device) for _, scan_id, offset, device in drain(volumes)]
        self.assertEqual(shown, [(8, 1, 1), (10, 3, 0), (13, 6, 1)])
        self.assertEqual(volumes.outcome, "end")


if __name__ == "__main__":
    unittest.main()
