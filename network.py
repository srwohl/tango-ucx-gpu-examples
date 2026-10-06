"""Set UCX policy before starting either process; usable on separate HPC nodes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def environment(profile, net_devices=None, prefix=None):
    env = os.environ.copy()
    tls = {"tcp": "tcp,cuda_copy,self", "rdma": "rc,cuda", "auto": "all"}[profile]
    env["UCX_TLS"] = env["TANGO_UCX_UCX_TLS"] = tls
    env["EXAMPLES_NETWORK"] = profile
    if net_devices is not None:
        env["UCX_NET_DEVICES"] = env["TANGO_UCX_UCX_NET_DEVICES"] = net_devices
    install = Path(prefix or env.get("TANGO_UCX_PREFIX", ROOT / "stage"))
    for library in ("lib", "lib64"):
        package = install / library / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if package.is_dir():
            env["PYTHONPATH"] = str(package) + os.pathsep + env.get("PYTHONPATH", "")
            break
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("tcp", "rdma", "auto"), default="tcp")
    parser.add_argument("--net-devices", help="Ethernet interface or RDMA HCA:port on this node")
    parser.add_argument("--prefix", help="installed tango-ucx prefix; defaults to ./stage")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    env = environment(args.profile, args.net_devices, args.prefix)
    print(json.dumps({"network": args.profile, "UCX_TLS": env["UCX_TLS"],
                      "UCX_NET_DEVICES": env.get("TANGO_UCX_UCX_NET_DEVICES", env.get("UCX_NET_DEVICES", "all")),
                      "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "all")}), file=sys.stderr)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
