import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def execute(executable, arguments, environment):
    started = time.monotonic()
    process = subprocess.Popen(
        [str(executable), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ | environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    records = []
    for line in stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {
        "arguments": arguments,
        "environment": environment,
        "returncode": process.returncode,
        "elapsed_s": time.monotonic() - started,
        "result": records[-1] if records else None,
        "stderr": stderr[-3000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    audit_root = Path(__file__).resolve().parents[1]
    executable = audit_root / "probes" / "transport_native"
    library = audit_root.parents[2] / "tango-ucx/build-gpu/src/transport/libtango-ucx-transport.a"
    output = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "current_transport_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "header": [],
        "transfers": [],
        "raw_segments": [],
        "native_segments": [],
    }
    for fields in (0, 40):
        for repeat in range(options.repeats):
            output["header"].append(
                execute(executable, ["--header", "--frames", "3000000", "--fields", str(fields)], {})
            )
    for payload_bytes in (128, 75000):
        for fields in (0, 40):
            for batch in (1, 16, 128):
                for repeat in range(options.repeats):
                    arguments = ["--frames", "5000", "--bytes", str(payload_bytes),
                                 "--fields", str(fields), "--batch", str(batch)]
                    output["transfers"].append(execute(executable, arguments, {}))
                samples = [record["result"]["ns_per_frame"] for record in output["transfers"][-options.repeats:]
                           if record["returncode"] == 0]
                print(f"bytes={payload_bytes} fields={fields} batch={batch}: "
                      f"{statistics.median(samples) / 1000:.2f} us/frame" if samples else "failed", flush=True)
    for segment, sizes in (("8256", (75000,)),
                           ("256K", (65000, 65400, 65535, 65536, 65537, 66000, 75000)),
                           ("1M", (75000,))):
        for payload_bytes in sizes:
            output["raw_segments"].append(execute(
                executable, ["--raw", "--bytes", str(payload_bytes)], {"UCX_SYSV_SEG_SIZE": segment}
            ))
    for segment in ("8256", "32K", "256K", "1M"):
        output["native_segments"].append(execute(
            executable, ["--frames", "100", "--bytes", "75000", "--batch", "16", "--validate-all", "--time-ucx"],
            {"UCX_SYSV_SEG_SIZE": segment}
        ))
    for batch in (1, 16, 128):
        output["native_segments"].append(execute(
            executable, ["--frames", "5000", "--bytes", "75000", "--batch", str(batch), "--time-ucx"], {}
        ))
    options.output.write_text(json.dumps(output, indent=2) + "\n")
    if any(record["returncode"] for record in output["transfers"]):
        raise SystemExit("a default-settings native transfer failed")


if __name__ == "__main__":
    main()
