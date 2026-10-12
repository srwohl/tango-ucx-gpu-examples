"""Unprofiled full-pipeline comparison: production FBP, persistent native ASTRA and TomocuPy shims."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[3]
SHIMS = {"production": None, "persistent": "fbp_shim", "fourierrec": "tomocupy_shim",
         "lprec": "tomocupy_shim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=list(SHIMS), choices=list(SHIMS))
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.scans, args.repeats) < 1:
        parser.error("scans and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repeat in range(args.repeats):
        # Rotate the order so no method always runs first or last.
        methods = args.methods[repeat % len(args.methods):] + args.methods[:repeat % len(args.methods)]
        for method in methods:
            output = args.output / f"{method}-{repeat}"
            command = [sys.executable, str(EXAMPLES / "tomography/demo.py"),
                       "--pixels", "512", "--slices", "128", "--angles", "360",
                       "--algorithm", "fbp", "--recon-filter", "ram-lak", "--scans", str(args.scans),
                       "--scan-period", "0", "--network", "auto", "--transport-batch", "16",
                       "--processing-batch", "16", "--processing-mode", "batched",
                       "--no-verify-volumes", "--output", str(output)]
            environment = dict(os.environ)
            environment.pop("UCX_PROTO_INFO", None)
            environment.pop("PYTHONPATH", None)
            if SHIMS[method]:
                environment["PYTHONPATH"] = str(Path(__file__).with_name(SHIMS[method]))
                environment.update(TOMOCUPY_METHOD=method, TOMOCUPY_CHUNK=str(args.chunk))
            with (args.output / f"{method}-{repeat}.log").open("w") as log:
                completed = subprocess.run(command, cwd=EXAMPLES, env=environment,
                                           stdout=log, stderr=log, timeout=300)
            if completed.returncode:
                raise RuntimeError((args.output / f"{method}-{repeat}.log").read_text()[-8000:])
            summary = json.loads((output / "summary.json").read_text())
            if summary["completed_scans"] != args.scans or summary["archived_frames"] != args.scans * 362:
                raise AssertionError("pipeline frame/volume counts differ")
            for stage in summary["stages"].values():
                if stage["failure"] or stage["input_failure"] or stage["quarantined_bytes"]:
                    raise AssertionError("pipeline stage failed or quarantined memory")
            calls = [json.loads(line) for line in (output / "reconstruct.log").read_text().splitlines()
                     if line.startswith(('{"fbp_native_shim":', '{"tomocupy_shim":'))]
            if SHIMS[method] and (len(calls) != args.scans or calls[-1]["completed_calls"] != args.scans
                                  or calls[-1].get("method", method) != method):
                raise AssertionError("import shim was not exercised for every scan")
            # The demo scores volumes only when it also asserts ASTRA's values: score the last one here.
            volume = np.load(output / "volume.npy")
            truth = np.load(output / "scan/reference.npz")["phantom"]
            result = dict(method=method, repeat=repeat, elapsed_seconds=summary["elapsed_seconds"],
                volumes_per_second=summary["throughput"]["volumes_per_second"],
                reconstruct_seconds=summary["stages"]["reconstruct"]["reconstruct_ns"] / 1e9,
                completed_scans=summary["completed_scans"], archived_frames=summary["archived_frames"],
                shim_calls=len(calls), finite=bool(np.isfinite(volume).all()),
                relative_l2_to_phantom=float(np.linalg.norm(volume - truth) / np.linalg.norm(truth)),
                output=str(output), command=command, stages=summary["stages"])
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items()
                              if key not in {"stages", "command", "output"}}), flush=True)


if __name__ == "__main__":
    main()
