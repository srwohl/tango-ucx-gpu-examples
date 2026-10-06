"""Exercise a stream, bounded HTTP volumes, a slow latest viewer and graceful Ctrl-C."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

from PIL import Image

from reconstruction import ALGORITHMS, configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-server", required=True, type=Path)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="gridrec")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tomography-stream-test-") as temporary:
        root = Path(temporary)
        output = root / "run"
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("demo.py")),
                 "--device-server", str(args.device_server), "--output", str(output),
                 "--algorithm", args.algorithm,
                 "--loop", "--scan-period", "0", "--live", "--no-browser", "--view-port", "0",
                 "--viewer-delay", "1", "--view-history-volumes", "2",
                 "--budget", "131072"], stdout=log, stderr=log)
            try:
                until = time.monotonic() + 60
                initial_devices = None
                while time.monotonic() < until:
                    if process.poll() is not None:
                        raise RuntimeError("looping demo exited before Stop")
                    if (output / "devices.json").exists():
                        devices = json.loads((output / "devices.json").read_text())
                        if initial_devices is None:
                            initial_devices = devices
                        if devices != initial_devices:
                            raise ValueError("device servers restarted between scans")
                    if (output / "status.json").exists() and (output / "live-ready.json").exists():
                        url = json.loads((output / "live-ready.json").read_text())["url"]
                        with urllib.request.urlopen(url + "/api/status", timeout=3) as response:
                            status = json.load(response)
                        if status["completed_scans"] >= 12 and status["viewer"]["received_volumes"]:
                            break
                    time.sleep(0.1)
                else:
                    raise TimeoutError("looping stream did not produce enough volumes")
                for device in initial_devices.values():
                    os.kill(device["pid"], 0)
                with urllib.request.urlopen(url, timeout=3) as response:
                    if b"Live GPU tomography" not in response.read():
                        raise ValueError("live page was not served")
                with urllib.request.urlopen(url + "/api/status", timeout=3) as response:
                    view = json.load(response)["viewer"]
                if not 1 <= view["buffered_volumes"] <= 2 or view["buffered_bytes"] > view["history_byte_limit"]:
                    raise ValueError("live history exceeded its bounds")
                frame = view["volumes"][-1]
                with urllib.request.urlopen(f"{url}/api/volume?v={frame['version']}", timeout=3) as response:
                    payload = response.read()
                    if (response.headers["X-Volume-Shape"] != "8,64,64" or
                        int(response.headers["X-Volume-Version"]) != frame["version"] or
                        len(payload) != 8 * 64 * 64 or min(payload) == max(payload)):
                        raise ValueError("live volume payload or identity differs")
                for axis, index, expected in ((0, 4, (64,64)), (1, 32, (64,8)), (2, 32, (64,8))):
                    with urllib.request.urlopen(
                        f"{url}/api/slice?axis={axis}&index={index}", timeout=3) as response:
                        with Image.open(BytesIO(response.read())) as image:
                            if image.size != expected or image.getextrema()[0] == image.getextrema()[1]:
                                raise ValueError("live view returned an empty or incorrect slice")
                # Change every backend while the same four devices and subscriptions run.
                for method in ALGORITHMS:
                    settings = configuration(method, iterations=12, scale_factor=1.2)
                    request = urllib.request.Request(url + "/api/reconstruction",
                        data=json.dumps(settings).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        requested = json.load(response)["requested"]
                    until = time.monotonic() + 45
                    while time.monotonic() < until:
                        if process.poll() is not None:
                            raise RuntimeError("looping demo exited during live tuning")
                        with urllib.request.urlopen(url + "/api/status", timeout=5) as response:
                            status = json.load(response)
                        frames = status["viewer"]["volumes"]
                        if frames and frames[-1].get("reconstruction", {}).get("revision") == requested["revision"]:
                            if frames[-1]["reconstruction"]["options"] != settings:
                                raise ValueError("live volume settings differ from the applied configuration")
                            break
                        time.sleep(.1)
                    else:
                        raise TimeoutError(f"live {method} settings did not reach the viewer")
                    if json.loads((output / "devices.json").read_text()) != initial_devices:
                        raise ValueError("device servers restarted during live tuning")
                    for device in initial_devices.values():
                        os.kill(device["pid"], 0)
                process.send_signal(signal.SIGINT)
                if process.wait(timeout=60):
                    raise RuntimeError("looping demo failed on Ctrl-C")
                summary = json.loads((output / "summary.json").read_text())
                scans = summary["completed_scans"]
                if not summary["stopped"] or scans < 12 or summary["archived_frames"] != scans * 98:
                    raise ValueError("graceful stop lost or truncated a scan")
                if not summary["live_view"]["skipped"]:
                    raise ValueError("slow latest viewer did not exercise skips")
                with (output / "volume-settings.jsonl").open() as settings_log:
                    records = [json.loads(line) for line in settings_log]
                if len(records) != scans or {record["options"]["algorithm"] for record in records} != set(ALGORITHMS):
                    raise ValueError("volume settings log did not record every scan and live algorithm")
                for device in initial_devices.values():
                    try:
                        os.kill(device["pid"], 0)
                    except ProcessLookupError:
                        continue
                    raise ValueError("device server remained running after shutdown")
                print(json.dumps(dict(algorithm=args.algorithm, completed_scans=scans, archived_frames=scans*98,
                                      viewer_skipped=summary["live_view"]["skipped"],
                                      live_controls=True, graceful_stop=True)), flush=True)
            except BaseException:
                print((root / "run.log").read_text()[-12000:], file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    main()
