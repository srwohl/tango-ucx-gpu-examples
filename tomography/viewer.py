"""Browser 3D view of a latest UCX subscription with bounded volume history."""
import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from network import environment
from reconstruction import live_configuration
from output_blocks import LatestCompleted, OutputCollector, read_complete


class ReconstructionControl:
    """Serialize HTTP and receive-thread access to a dedicated Tango proxy."""

    def __init__(self, proxy):
        self.proxy = proxy
        self.lock = threading.Lock()

    def settings(self):
        with self.lock:
            return json.loads(self.proxy.command_inout("GetReconstruction"))

    def configure(self, options):
        with self.lock:
            state = json.loads(self.proxy.command_inout("GetReconstruction"))
            options = live_configuration(options, state["detector_columns"])
            return json.loads(self.proxy.command_inout("ConfigureReconstruction", json.dumps(options)))

    def for_scan(self, scan_id):
        with self.lock:
            return json.loads(self.proxy.command_inout("ReconstructionForScan", str(scan_id)))


def read_next(sub):
    # No ring-backed arrays or records escape this call.
    batch = sub.read(timeout=0.1)
    if batch is None:
        return None
    return (batch.array[-1].copy(), int(batch.records[-1]["scan_id"]),
            int(batch.records[-1]["index"]))


def display_voxels(volume, display_max=0.01, *, inplace=False):
    """Quantize the selected display window; C order is z, y, x."""
    scaled = volume if inplace else np.empty_like(volume)
    np.multiply(volume, 255 / display_max, out=scaled)
    np.clip(scaled, 0, 255, out=scaled)
    return scaled.astype(np.uint8)


def png_slice(volume, axis, index, window_max=0.01, display_max=0.01):
    plane = np.take(volume, index, axis=axis)
    if not np.isfinite(window_max) or not 0 < window_max <= display_max:
        raise ValueError(f"slice window maximum must be in (0, {display_max}]")
    if plane.dtype == np.uint8:
        grayscale = (plane if window_max == display_max else
                     np.clip(plane * (display_max / window_max), 0, 255).astype(np.uint8))
    else:
        grayscale = np.clip(plane * (255 / window_max), 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(grayscale).save(buffer, format="PNG")
    return buffer.getvalue()


class VolumeHistory:
    """Own display copies only, bounded by both count and bytes, with stable IDs."""

    def __init__(self, max_volumes=32, max_bytes=64 * 1024**2, display_max=0.01):
        if max_volumes < 1 or max_bytes < 1:
            raise ValueError("history limits must be positive")
        self.max_volumes = max_volumes
        self.max_bytes = max_bytes
        if not np.isfinite(display_max) or display_max <= 0:
            raise ValueError("display max must be finite and positive")
        self.display_max = display_max
        self.lock = threading.Lock()
        self.frames = OrderedDict()
        self.bytes = 0
        self.started = time.monotonic()
        self.state = dict(scan_id=None, version=0, shape=None, failure="", outcome=None,
                          received_volumes=0, skipped=0, transport="", display_max=display_max)

    def append(self, volume, scan_id, frame_index, health, reconstruction=None, *, inplace=False):
        if volume.ndim != 3 or not all(volume.shape) or not np.isfinite(volume).all():
            raise ValueError("expected a finite, nonempty 3D live volume")
        if volume.size > self.max_bytes:
            raise ValueError("display volume exceeds history byte limit; increase --history-mib")
        voxels = display_voxels(volume, self.display_max, inplace=inplace)
        voxels.flags.writeable = False
        with self.lock:
            if self.state["scan_id"] is not None and scan_id <= self.state["scan_id"]:
                raise ValueError("live scan identity did not advance")
            version = self.state["version"] + 1
            meta = dict(version=version, scan_id=scan_id, shape=list(voxels.shape),
                        time_seconds=time.monotonic() - self.started, bytes=voxels.nbytes)
            if reconstruction is not None:
                meta["reconstruction"] = reconstruction
            # Evict before insertion to keep the retained history within both bounds.
            while self.frames and (len(self.frames) >= self.max_volumes or
                                   self.bytes + voxels.nbytes > self.max_bytes):
                _, (old, _) = self.frames.popitem(last=False)
                self.bytes -= old.nbytes
            self.frames[version] = (voxels, meta)
            self.bytes += voxels.nbytes
            received = self.state["received_volumes"] + 1
            self.state.update(scan_id=scan_id, version=version, shape=meta["shape"],
                              received_volumes=received, skipped=frame_index + 1 - received,
                              transport_skipped=health["skipped"], transport=health["transport"])

    def snapshot(self):
        with self.lock:
            return dict(self.state, volumes=[meta.copy() for _, meta in self.frames.values()],
                        buffered_volumes=len(self.frames), buffered_bytes=self.bytes,
                        history_limit=self.max_volumes, history_byte_limit=self.max_bytes)

    def get(self, version=None):
        with self.lock:
            if not self.frames:
                return None
            if version is None:
                version = next(reversed(self.frames))
            return self.frames.get(version)

    def update(self, **values):
        with self.lock:
            self.state.update(values)


def make_handler(history, output, control=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type, status=200, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Scrubbing may cancel an older request.

        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/":
                self.respond((HERE / "live.html").read_bytes(), "text/html; charset=utf-8")
            elif request.path == "/api/status":
                try:
                    progress = json.loads((output / "status.json").read_text())
                except FileNotFoundError:
                    progress = dict(phase="waiting", stages={})
                if control is not None:
                    try:
                        progress["reconstruction_control"] = control.settings()
                    except Exception as error:
                        progress["reconstruction_control_error"] = str(error)
                self.respond(json.dumps(dict(**progress, viewer=history.snapshot())).encode(),
                             "application/json")
            elif request.path == "/api/reconstruction":
                if control is None:
                    self.respond(b"Reconstruction controls unavailable", "text/plain", 503)
                    return
                try:
                    self.respond(json.dumps(control.settings()).encode(), "application/json")
                except Exception as error:
                    self.respond(str(error).encode(), "text/plain", 503)
            elif request.path in ("/api/volume", "/api/slice"):
                try:
                    query = parse_qs(request.query)
                    version = int(query["v"][0]) if "v" in query else None
                    frame = history.get(version)
                    if frame is None:
                        self.respond(b"Volume is no longer buffered" if version is not None else b"",
                                     "text/plain", 410 if version is not None else 204)
                        return
                    volume, meta = frame
                    if request.path == "/api/volume":
                        self.respond(volume.tobytes(), "application/octet-stream", headers={
                            "X-Volume-Version": meta["version"], "X-Scan-Id": meta["scan_id"],
                            "X-Volume-Shape": ",".join(map(str, meta["shape"])),
                            "X-Volume-Format": "uint8-zyx",
                            "X-Display-Range": f"0,{history.display_max}"})
                    else:
                        axis = int(query.get("axis", ["0"])[0])
                        if axis not in (0, 1, 2):
                            raise ValueError("axis must be 0, 1 or 2")
                        index = int(query.get("index", [str(volume.shape[axis] // 2)])[0])
                        if not 0 <= index < volume.shape[axis]:
                            raise ValueError("slice outside volume")
                        window_max = float(query.get("window_max", [str(history.display_max)])[0])
                        self.respond(png_slice(volume, axis, index, window_max, history.display_max), "image/png")
                except ValueError as error:
                    self.respond(str(error).encode(), "text/plain", 400)
            else:
                self.respond(b"Not found", "text/plain", 404)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/reconstruction":
                self.respond(b"Not found", "text/plain", 404)
                return
            # Browser writes must originate from this viewer, including its loopback port.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond(b"Origin does not match this viewer", "text/plain", 403)
                return
            if control is None:
                self.respond(b"Reconstruction controls unavailable", "text/plain", 503)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("expected a JSON settings object of at most 16 KiB")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("expected application/json")
                options = json.loads(self.rfile.read(size))
                state = control.configure(options)
                self.respond(json.dumps(state).encode(), "application/json")
            except (ValueError, TypeError) as error:
                self.respond(str(error).encode(), "text/plain", 400)
            except Exception as error:
                self.respond(str(error).encode(), "text/plain", 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--budget", type=int, default=1048576)
    parser.add_argument("--output-mode", choices=("volume", "blocks"), default="volume")
    parser.add_argument("--delay", type=float, default=0, help="slow viewer for pressure checks")
    parser.add_argument("--history-volumes", type=int, default=32)
    parser.add_argument("--history-mib", type=int, default=64)
    parser.add_argument("--display-max", type=float, default=0.01)
    args = parser.parse_args()
    if args.history_volumes < 1 or args.history_mib < 1:
        parser.error("history limits must be positive")
    if not np.isfinite(args.display_max) or args.display_max <= 0:
        parser.error("display max must be finite and positive")
    for location in environment("tcp", "lo")["PYTHONPATH"].split(":"):
        if location:
            sys.path.insert(0, location)
    import tango
    import tango_ucx

    subscribe = tango_ucx.every if args.output_mode == "blocks" else tango_ucx.latest
    sub = subscribe(tango.DeviceProxy(args.device), memory="host", budget=args.budget,
                          batch=1, label="tomography-live-view")
    proxy = tango.DeviceProxy(args.device)
    proxy.set_timeout_millis(60000)
    control = ReconstructionControl(proxy)
    collector = OutputCollector.from_description(sub.description, control.for_scan, storage_volumes=3)
    if (collector is not None) != (args.output_mode == "blocks"):
        sub.close()
        raise ValueError("viewer output mode differs from reconstruction description")
    history = VolumeHistory(args.history_volumes, args.history_mib * 1024**2, args.display_max)
    if collector is not None:
        history.update(delivery="every-block/latest-completed", assembled_volumes=0,
                       assembly_bytes=collector.data.nbytes, display_snapshot_limit_bytes=2 * collector.data.nbytes)

    def receive():
        mailbox = LatestCompleted() if collector is not None else None
        display = None
        display_failure = []
        if mailbox is not None:
            def display_completed():
                try:
                    while (result := mailbox.take()) is not None:
                        volume, scan_id, index, settings, health = result
                        history.append(volume, scan_id, index, health, settings, inplace=True)
                        # Drop the private float snapshot before waiting for another.
                        result = volume = None
                        if args.delay:
                            time.sleep(args.delay)
                except Exception as error:
                    display_failure.append(error)
            display = threading.Thread(target=display_completed, daemon=True)
            display.start()
        try:
            while True:
                if display_failure:
                    raise display_failure[0]
                result = read_complete(sub, collector) if collector is not None else read_next(sub)
                if result is None:
                    if sub.outcome is not None:
                        break
                    continue
                if mailbox is not None:
                    mailbox.put(result, sub.health())
                    history.update(assembled_volumes=collector.completed)
                    result = None
                else:
                    volume, scan_id, frame_index = result
                    history.append(volume, scan_id, frame_index, sub.health(), control.for_scan(scan_id))
                    if args.delay:
                        time.sleep(args.delay)  # Latest delivery may skip; it applies no pressure.
            if collector is not None:
                collector.finish()
            if sub.outcome != "end":
                raise RuntimeError(f"live subscription failed: {sub.health()}")
        except Exception as error:
            history.update(failure=str(error))
        finally:
            if mailbox is not None:
                mailbox.close()
                display.join()
                if display_failure:
                    history.update(failure=str(display_failure[0]))
            health = sub.health()
            history.update(outcome=sub.outcome, transport_skipped=health["skipped"])
            temporary = args.output / "live-summary.tmp"
            temporary.write_text(json.dumps(history.snapshot(), indent=2))
            temporary.replace(args.output / "live-summary.json")
            sub.close()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(history, args.output, control))
        url = f"http://127.0.0.1:{server.server_port}"
        ready = args.output / "live-ready.tmp"
        ready.write_text(json.dumps(dict(url=url)))
        ready.replace(args.output / "live-ready.json")
        threading.Thread(target=receive, daemon=True).start()
        print(json.dumps(dict(live_view=url)), flush=True)
        server.serve_forever(poll_interval=0.2)
    finally:
        sub.close()


if __name__ == "__main__":
    main()
