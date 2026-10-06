"""Run separate publisher/subscriber processes and independently check the archive."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from network import environment
from verify_archive import verify

ROOT = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher", type=Path, default=ROOT / "build" / "opaque_publisher")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--profile", choices=("tcp", "rdma", "auto"), default="tcp")
    parser.add_argument("--net-devices")
    args = parser.parse_args()
    env = environment(args.profile, args.net_devices or ("lo" if args.profile == "tcp" else None))
    # Import from the installed prefix in this process, too.
    for location in env.get("PYTHONPATH", "").split(":"):
        if location:
            sys.path.insert(0, location)
    import tango

    with tempfile.TemporaryDirectory(prefix="tango-ucx-examples-") as directory:
        root = Path(directory)
        device = f"127.0.0.1:{free_port()}/example/opaque/1#dbase=no"
        port = device.split(":", 1)[1].split("/", 1)[0]
        with (root / "publisher.log").open("w+") as log:
            server = subprocess.Popen([str(args.publisher.resolve()), "local", "-nodb", "-dlist",
                                       "example/opaque/1", "-ORBendPoint", f"giop:tcp:127.0.0.1:{port}"],
                                      env=env, stdout=log, stderr=log)
            try:
                until = time.monotonic() + 60
                proxy = None
                while time.monotonic() < until and server.poll() is None:
                    try:
                        proxy = tango.DeviceProxy(device)
                        proxy.set_timeout_millis(1000)
                        proxy.ping()
                        break
                    except tango.DevFailed:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("publisher did not become ready")
                proxy.set_timeout_millis(60000)

                def run(source, script, extra=()):
                    # End sessions from the preceding run before constructing a fresh instance.
                    proxy.command_inout("Configure", json.dumps({"source": source}))
                    command = [sys.executable, str(ROOT / script), device, "--batch", "4",
                               "--budget", "4MiB", "--start", str(args.frames), *extra]
                    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, timeout=90)
                    print(f"{source} -> {script} {' '.join(extra)}\n{result.stdout}", flush=True)
                    if result.returncode:
                        raise RuntimeError(f"example exited {result.returncode}")
                    summaries = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                    if not any(s.get("verified_frames") == args.frames and s.get("outcome") == "end"
                               and s.get("transport") for s in summaries):
                        raise RuntimeError("example did not report verified delivery")

                if args.calibration_only:
                    def run_calibration(source, receive, via_hpc=False):
                        config = dict(source=source, kind="calibration", shape=[129, 131],
                                      dark_frames=4, flat_frames=4, calibrations=2)
                        if via_hpc:
                            command = [sys.executable, str(ROOT / "hpc_subscriber.py"), device,
                                       "--example", "dark_flat", "--source", source, "--frames", "23",
                                       "--shape", "129,131", "--dark-frames", "4", "--flat-frames", "4",
                                       "--calibrations", "2"]
                        else:
                            proxy.command_inout("Configure", json.dumps(config))
                            command = [sys.executable, str(ROOT / "dark_flat.py"), device, "--start", "23"]
                        output = root / f"calibration-{source}-{receive}-{'hpc' if via_hpc else 'direct'}"
                        command.extend(["--receive", receive, "--batch", "5", "--inflight", "3",
                                        "--budget", "1MiB", "--output", str(output)])
                        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, timeout=90)
                        print(f"calibration {source} -> {receive} (HPC launcher: {via_hpc})\n{result.stdout}",
                              flush=True)
                        summaries = [json.loads(line) for line in result.stdout.splitlines()
                                     if line.startswith("{")]
                        if result.returncode or not any(s.get("corrected_data_frames") == 46 and
                            s.get("calibrations") == 2 and s.get("verified_frames") == 62 and
                            s.get("outcome") == "end" for s in summaries):
                            raise RuntimeError("streamed calibration correction failed")
                        import numpy as np
                        for calibration_id in (1, 2):
                            with np.load(output / f"calibration-{calibration_id}.npz") as maps:
                                if maps["dark"].shape != (129, 131) or not np.isnan(maps["inverse"]).any():
                                    raise RuntimeError("invalid saved GPU calibration maps")
                        if np.load(output / "last_corrected.npy").shape != (129, 131):
                            raise RuntimeError("missing corrected output")

                    for source in (("host",) if args.host_only else ("host", "cuda:0")):
                        for receive in ("gpu", "pinned"):
                            run_calibration(source, receive)
                    run_calibration("host", "gpu", via_hpc=True)
                    return

                run("host", "host_archive.py", ("--output", str(root / "host-archive")))
                print(json.dumps(verify(root / "host-archive")))
                if not args.host_only:
                    import cupy
                    print(json.dumps({"gpu": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                                      "driver": cupy.cuda.runtime.driverGetVersion()}), flush=True)
                    run("host", "gpu_receive.py", ("--access", "dlpack"))
                    run("host", "gpu_receive.py", ("--access", "pointer"))
                    run("host", "pinned_upload.py")
                    run("cuda:0", "host_archive.py", ("--output", str(root / "gpu-archive")))
                    print(json.dumps(verify(root / "gpu-archive")))
                    run("cuda:0", "gpu_receive.py", ("--access", "pointer"))
                    run("cuda:0", "pinned_upload.py")
                    # Exercise the HPC launch sequence locally, including its readiness wait,
                    # Configure, installed-package imports, and exec into a GPU subscriber.
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "hpc_subscriber.py"), device, "--source", "cuda:0",
                         "--example", "gpu_receive", "--access", "pointer", "--frames", str(args.frames),
                         "--batch", "4", "--budget", "4MiB"],
                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
                    print(f"HPC subscriber launch\n{result.stdout}", flush=True)
                    if result.returncode or not any(
                        json.loads(line).get("verified_frames") == args.frames
                        for line in result.stdout.splitlines() if line.startswith("{")):
                        raise RuntimeError("HPC subscriber launch failed")
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
