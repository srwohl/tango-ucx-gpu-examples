"""Run grouped asynchronous scan assembly with unchanged pipeline sinks and science."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROBES = Path(__file__).resolve().parent
AUDIT = PROBES.parent
EXAMPLES = AUDIT.parents[1]


def build(output):
    source_path = EXAMPLES / "tomography/pipeline_device.cpp"
    source = source_path.read_text()
    anchor = "    void process(std::stop_token stop) {"
    if source.count(anchor) != 1 or source.count("namespace {") != 1:
        raise RuntimeError("pipeline source does not match the experiment insertion points")
    source = source.replace("namespace {", "#include <condition_variable>\n#include <cstdlib>\n"
                            "#include <deque>\n#include <exception>\nnamespace {", 1)
    source = source.replace(anchor, f'#include "{PROBES / "scan_overlap_worker.inc"}"\n' + anchor +
                            '\n        if(cfg.role == "reconstruct" && std::getenv("TANGO_SCAN_OVERLAP")) {\n'
                            "            process_overlap(stop);\n            return;\n        }", 1)
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "pipeline_device.cpp"
    generated.write_text(source)
    hashes = {}
    for path in (source_path, EXAMPLES / "tomography/processors.py", EXAMPLES / "tomography/reconstruction.py",
                 EXAMPLES / "tomography/demo.py", PROBES / "scan_overlap_worker.inc", PROBES / "scan_overlap.py"):
        hashes[str(path.relative_to(EXAMPLES))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    commands = [["cmake", *shlex.split(os.environ.get("CMAKE_ARGS", "")),
                 "-S", str(PROBES / "scan_overlap_build"), "-B", str(output), "-GNinja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=" + os.environ.get("CXX", "c++"),
                 "-DCMAKE_PREFIX_PATH=" + str(EXAMPLES / "stage"), "-DPython3_EXECUTABLE=" + sys.executable,
                 "-DPROBE_SOURCE=" + str(generated), "-DTOMOGRAPHY_DIR=" + str(EXAMPLES / "tomography")],
                ["cmake", "--build", str(output), "--target", "pipeline_device", "-j", "2"]]
    with (output / "build.log").open("w") as log:
        for command in commands:
            completed = subprocess.run(command, cwd=EXAMPLES, stdout=log, stderr=log, timeout=180)
            if completed.returncode:
                raise RuntimeError((output / "build.log").read_text()[-8000:])
    return output / "pipeline_device"


def records(path, key):
    found = []
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get(key):
                found.append(record)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=("production", "persistent"),
                        default=["production", "persistent"])
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--buffers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--slices", type=int, default=128)
    parser.add_argument("--angles", type=int, default=360)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device-server", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.scans, args.repeats, args.pixels, args.slices, args.angles, args.batch) < 1:
        parser.error("counts and dimensions must be positive")
    if "persistent" in args.backends and (args.slices, args.angles, args.pixels) != (128, 360, 512):
        parser.error("the existing persistent shim requires 128 slices, 360 angles and 512 pixels")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.device_server.resolve() if args.device_server else build(args.output / "build")
    if args.build_only:
        print(binary, flush=True)
        return
    results = []
    historical = json.loads((AUDIT / "fbp-pipeline-paired/results.json").read_text())
    historical = [{key: row[key] for key in ("method", "repeat", "elapsed_seconds", "volumes_per_second",
                                            "reconstruct_seconds", "volumes_verified")} for row in historical]
    (args.output / "historical-results.json").write_text(json.dumps(historical, indent=2) + "\n")
    for repeat in range(args.repeats):
        backends = args.backends if repeat % 2 == 0 else list(reversed(args.backends))
        for backend in backends:
            methods = ["scalar", "overlap"] if args.with_baseline else ["overlap"]
            if repeat % 2:
                methods.reverse()
            for method in methods:
                name = f"{backend}-{method}-{repeat}"
                output = args.output / name
                command = [sys.executable, str(EXAMPLES / "tomography/demo.py"),
                           "--device-server", str(binary), "--pixels", str(args.pixels),
                           "--slices", str(args.slices), "--angles", str(args.angles),
                           "--algorithm", "fbp", "--recon-filter", "ram-lak", "--scans", str(args.scans),
                           "--scan-period", "0", "--network", "auto", "--transport-batch", str(args.batch),
                           "--processing-batch", str(args.batch), "--processing-mode", "batched",
                           "--verify-volumes" if args.verify else "--no-verify-volumes", "--output", str(output)]
                environment = dict(os.environ)
                for key in ("UCX_PROTO_INFO", "PYTHONPATH", "TANGO_SCAN_OVERLAP", "TANGO_SCAN_BUFFERS", "TANGO_SCAN_BACKEND"):
                    environment.pop(key, None)
                environment["PYTHONPATH"] = str(PROBES / "scan_overlap_shim")
                environment["TANGO_SCAN_BACKEND"] = backend
                if method == "overlap":
                    environment["TANGO_SCAN_OVERLAP"] = "1"
                    environment["TANGO_SCAN_BUFFERS"] = str(args.buffers)
                state = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                                        "--format=csv"], capture_output=True, text=True, check=True).stdout
                with (args.output / f"{name}.log").open("w") as log:
                    completed = subprocess.run(command, cwd=EXAMPLES, env=environment,
                                               stdout=log, stderr=log, timeout=240)
                if completed.returncode:
                    raise RuntimeError((args.output / f"{name}.log").read_text()[-10000:])
                summary = json.loads((output / "summary.json").read_text())
                if summary["completed_scans"] != args.scans or summary["archived_frames"] != args.scans * (args.angles + 2):
                    raise AssertionError("pipeline frame/volume counts differ")
                if summary["volumes_verified"] != args.verify:
                    raise AssertionError("verification mode differs")
                for stage in summary["stages"].values():
                    if stage["failure"] or stage["input_failure"] or stage["quarantined_bytes"]:
                        raise AssertionError("pipeline stage failed or quarantined memory")
                native = records(output / "reconstruct.log", "fbp_native_shim")
                overlap = records(output / "reconstruct.log", "scan_overlap_summary")
                if backend == "persistent" and (len(native) != args.scans or native[-1]["completed_calls"] != args.scans
                        or any(record["filter_rows"] != 45 for record in native)):
                    raise AssertionError("persistent FBP did not retain the previous filtering")
                if method == "overlap" and (len(overlap) != 1 or overlap[0]["completed_scans"] != args.scans
                        or overlap[0]["assembled_scans"] != args.scans or overlap[0]["peak_busy_buffers"] > args.buffers):
                    raise AssertionError("overlap path was not exercised or exceeded its buffer limit")
                result = dict(backend=backend, method=method, repeat=repeat, buffers=args.buffers,
                              command=command, output=str(output), gpu_before=state,
                              elapsed_seconds=summary["elapsed_seconds"],
                              volumes_per_second=summary["throughput"]["volumes_per_second"],
                              reconstruct_seconds=summary["stages"]["reconstruct"]["reconstruct_ns"] / 1e9,
                              completed_scans=summary["completed_scans"], archived_frames=summary["archived_frames"],
                              volumes_verified=args.verify, relative_l2_error=summary["relative_l2_error"],
                              overlap=overlap, stages=summary["stages"])
                results.append(result)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({key: value for key, value in result.items() if key not in {"stages", "command", "gpu_before"}}), flush=True)


if __name__ == "__main__":
    main()
