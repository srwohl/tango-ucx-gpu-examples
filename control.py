"""Configure a fresh publisher or start after every subscriber has reported ready."""
import argparse
import json

import tango


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device")
    commands = parser.add_subparsers(dest="command", required=True)
    configure = commands.add_parser("configure")
    configure.add_argument("--source", default="host", help="host or cuda:N")
    configure.add_argument("--kind", choices=("opaque", "calibration"), default="opaque")
    configure.add_argument("--shape", default="129,131", help="calibration frame rows,columns")
    configure.add_argument("--dark-frames", type=int, default=4)
    configure.add_argument("--flat-frames", type=int, default=4)
    configure.add_argument("--calibrations", type=int, default=2)
    configure.add_argument("--lengths", default="1,129,513,65539,262144")
    configure.add_argument("--budget", type=int, default=64 << 20, help="publisher budget in bytes")
    start = commands.add_parser("start")
    start.add_argument("count", type=int)
    args = parser.parse_args()
    proxy = tango.DeviceProxy(args.device)
    proxy.set_timeout_millis(60000)
    if args.command == "configure":
        proxy.command_inout("Configure", json.dumps(dict(source=args.source, budget=args.budget,
                            lengths=[int(n) for n in args.lengths.split(",")], kind=args.kind,
                            shape=[int(n) for n in args.shape.split(",")], dark_frames=args.dark_frames,
                            flat_frames=args.flat_frames, calibrations=args.calibrations)))
    else:
        proxy.command_inout("Start", args.count)


if __name__ == "__main__":
    main()
