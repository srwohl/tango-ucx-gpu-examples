"""Wait for a remote publisher, configure it, and run one verifying subscriber."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import tango


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device")
    parser.add_argument("--example", choices=("gpu_receive", "pinned_upload", "host_archive", "dark_flat"),
                        default="gpu_receive")
    parser.add_argument("--source", default="cuda:0")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("archive"))
    parser.add_argument("--access", choices=("dlpack", "pointer"), default="pointer")
    parser.add_argument("--shape", default="129,131", help="calibration frame rows,columns")
    parser.add_argument("--dark-frames", type=int, default=4)
    parser.add_argument("--flat-frames", type=int, default=4)
    parser.add_argument("--calibrations", type=int, default=2)
    args, extra = parser.parse_known_args()
    until = time.monotonic() + 90
    while time.monotonic() < until:
        try:
            proxy = tango.DeviceProxy(args.device)
            proxy.set_timeout_millis(1000)
            proxy.ping()
            break
        except tango.DevFailed:
            time.sleep(0.25)
    else:
        raise TimeoutError("remote publisher did not become ready")
    proxy.set_timeout_millis(60000)
    configuration = {"source": args.source}
    if args.example == "dark_flat":
        configuration.update(kind="calibration", shape=[int(n) for n in args.shape.split(",")],
                             dark_frames=args.dark_frames, flat_frames=args.flat_frames,
                             calibrations=args.calibrations)
    proxy.command_inout("Configure", json.dumps(configuration))
    script = Path(__file__).resolve().parent / f"{args.example}.py"
    command = [sys.executable, str(script), args.device, "--start", str(args.frames), *extra]
    if args.example == "gpu_receive":
        command.extend(["--access", args.access])
    elif args.example == "host_archive":
        command.extend(["--output", str(args.output)])
    elif args.example == "dark_flat":
        command.extend(["--output", str(args.output)])
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
