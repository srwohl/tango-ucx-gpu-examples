"""Archive original compressed frames independently of the GPU processing branch."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=262144)
    args = parser.parse_args()
    for location in environment("tcp", "lo")["PYTHONPATH"].split(":"):
        if location:
            sys.path.insert(0, location)
    import tango
    import tango_ucx

    args.output.mkdir()
    sub = tango_ucx.every(tango.DeviceProxy(args.device), memory="host", budget=args.budget,
                         batch=1, label="compressed-archive")
    try:
        meta = json.loads(sub.description["application_text"])
        if meta["role"] != "source" or meta["codec"] not in ("lz4-raw", "raw"):
            raise ValueError("archive expects the original detector stream")
        (args.output / "description.json").write_text(json.dumps(sub.description, indent=2))
        with ExitStack() as cleanup:
            data = cleanup.enter_context((args.output / "payloads.bin").open("xb"))
            index = cleanup.enter_context((args.output / "records.jsonl").open("x"))
            args.ready.write_text(json.dumps({"ready": True, "device": args.device}))
            count = 0
            frames_per_scan = meta["angles"] + 2
            progress = args.output / "progress.json"

            def report_progress():
                data.flush()
                index.flush()
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(json.dumps(dict(
                    archived_frames=count, completed_scans=count // frames_per_scan,
                    archived_bytes=data.tell())))
                temporary.replace(progress)

            report_progress()

            def append_next():
                # Function scope releases the batch and all its exports before the next read.
                batch = sub.read(timeout=1)
                if batch is None:
                    if sub.outcome is None:
                        return 0
                    if sub.outcome != "end":
                        raise RuntimeError(f"archive failed: {sub.health()}")
                    return None
                for i, record in enumerate(batch.records):
                    position = int(record["index"])
                    cycle, frame = divmod(position, frames_per_scan)
                    if (position != count + i or
                        int(record["scan_id"]) != meta["scan_id"] + cycle or
                        int(record["calibration_id"]) != meta["calibration_id"] + cycle or
                        int(record["kind"]) != min(frame, 2) or
                        int(record["projection"]) != max(0, frame - 2)):
                        raise ValueError("archive scan boundary or frame order mismatch")
                    payload = batch.payload(i)
                    row = {key: (float(record[key]) if key == "theta" else int(record[key]))
                           for key in record.dtype.names}
                    row.update(offset=data.tell(), bytes=payload.nbytes)
                    data.write(memoryview(payload))
                    index.write(json.dumps(row) + "\n")
                return batch.frames

            while True:
                received = append_next()
                if received is None:
                    break
                count += received
                if count % frames_per_scan == 0:
                    report_progress()
            if not count or count % frames_per_scan:
                raise ValueError("archive ended within a scan")
            report_progress()
        health = sub.health()
        if health["failure"] or health["quarantined_bytes"]:
            raise RuntimeError(f"archive unhealthy: {health}")
        summary = dict(archived_frames=count, completed_scans=count // frames_per_scan,
                       archived_bytes=(args.output / "payloads.bin").stat().st_size, outcome=sub.outcome,
                       transport=health["transport"])
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        sub.close()


if __name__ == "__main__":
    main()
