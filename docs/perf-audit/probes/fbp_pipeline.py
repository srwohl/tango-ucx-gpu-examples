"""Paired unprofiled full-pipeline comparison with a temporary native FBP import shim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

EXAMPLES = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.scans, args.repeats) < 1:
        parser.error("scans and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repeat in range(args.repeats):
        methods = ["persistent"] if args.candidate_only else (
            ["production", "persistent"] if repeat % 2 == 0 else ["persistent", "production"])
        for method in methods:
            output = args.output / f"{method}-{repeat}"
            command = [sys.executable, str(EXAMPLES / "tomography/demo.py"),
                       "--pixels", "512", "--slices", "128", "--angles", "360",
                       "--algorithm", "fbp", "--recon-filter", "ram-lak", "--scans", str(args.scans),
                       "--scan-period", "0", "--network", "auto", "--transport-batch", "16",
                       "--processing-batch", "16", "--processing-mode", "batched",
                       "--verify-volumes" if args.verify else "--no-verify-volumes",
                       "--output", str(output)]
            environment = dict(os.environ)
            environment.pop("UCX_PROTO_INFO", None)
            environment.pop("PYTHONPATH", None)
            if method == "persistent":
                environment["PYTHONPATH"] = str(Path(__file__).with_name("fbp_shim"))
            with (args.output / f"{method}-{repeat}.log").open("w") as log:
                completed = subprocess.run(command, cwd=EXAMPLES, env=environment,
                                           stdout=log, stderr=log, timeout=180)
            if completed.returncode:
                raise RuntimeError((args.output / f"{method}-{repeat}.log").read_text()[-8000:])
            summary = json.loads((output / "summary.json").read_text())
            if summary["completed_scans"] != args.scans or summary["archived_frames"] != args.scans * 362:
                raise AssertionError("pipeline frame/volume counts differ")
            if summary["volumes_verified"] != args.verify:
                raise AssertionError("verification mode differs")
            for stage in summary["stages"].values():
                if stage["failure"] or stage["input_failure"] or stage["quarantined_bytes"]:
                    raise AssertionError("pipeline stage failed or quarantined memory")
            native_calls = [json.loads(line) for line in (output / "reconstruct.log").read_text().splitlines()
                            if line.startswith('{"fbp_native_shim":')]
            if method == "persistent" and (len(native_calls) != args.scans
                    or native_calls[-1]["completed_calls"] != args.scans
                    or any(call["filter_rows"] != 45 for call in native_calls)):
                raise AssertionError("native import shim was not exercised with production filter batching")
            result = dict(method=method, repeat=repeat, command=command,
                output=str(output), elapsed_seconds=summary["elapsed_seconds"],
                volumes_per_second=summary["throughput"]["volumes_per_second"],
                reconstruct_seconds=summary["stages"]["reconstruct"]["reconstruct_ns"] / 1e9,
                completed_scans=summary["completed_scans"], archived_frames=summary["archived_frames"],
                volumes_verified=summary["volumes_verified"], native_calls=native_calls,
                relative_l2_error=summary["relative_l2_error"], stages=summary["stages"])
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items()
                              if key not in {"stages", "command", "native_calls"}}), flush=True)


if __name__ == "__main__":
    main()
