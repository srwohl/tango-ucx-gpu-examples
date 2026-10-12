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
from pipeline_control import PipelineControl, PipelineConflict
from fan_in import NewestVolumes
from output_blocks import LatestCompleted, OutputCollector, read_complete
from streaming import Updates, plane_array, plane_list, update_interval


class ReconstructionControl:
    """Serialize HTTP and receive-thread access to dedicated Tango proxies.

    Every reconstructor of a pull set receives the same settings in the same order, so their
    revisions agree. Each applies them to the next scan it takes.
    """

    def __init__(self, *proxies):
        self.proxies = proxies
        self.lock = threading.Lock()

    def _each(self, command, *argument):
        """One state: the requested settings are common, the active scan is the newest started."""
        states = [json.loads(proxy.command_inout(command, *argument)) for proxy in self.proxies]
        started = [state["active"] for state in states if state["active"] is not None]
        return dict(states[0], finished=all(state["finished"] for state in states),
                    active=max(started, key=lambda active: active["scan_id"]) if started else None)

    def settings(self):
        with self.lock:
            return self._each("GetReconstruction")

    def configure(self, options):
        with self.lock:
            columns = self._each("GetReconstruction")["detector_columns"]
            return self._each("ConfigureReconstruction", json.dumps(live_configuration(options, columns)))

    def configure_slices(self, planes):
        """Move the three planes of slice output; the next update uses them."""
        with self.lock:
            return json.loads(self.proxies[0].command_inout(
                "ConfigureSlices", json.dumps(plane_list(plane_array(planes)))))

    def for_scan(self, scan_id, reconstructor=0):
        """Settings of a scan, held by the device that reconstructed it."""
        with self.lock:
            return json.loads(self.proxies[reconstructor].command_inout("ReconstructionForScan", str(scan_id)))


def display_voxels(volume, display_max=0.01, *, inplace=False):
    """Quantize the selected display window; C order is z, y, x."""
    scaled = volume if inplace else np.empty_like(volume)
    np.multiply(volume, 65535 / display_max, out=scaled)
    np.clip(scaled, 0, 65535, out=scaled)
    return scaled.astype('<u2')


def png_slice(volume, axis, index, window_max=0.01, display_max=0.01, window_min=0):
    plane = np.take(volume, index, axis=axis)
    if (not np.isfinite(window_max) or not np.isfinite(window_min) or
            not 0 <= window_min < window_max <= display_max):
        raise ValueError(f"slice window must satisfy 0 <= minimum < maximum <= {display_max}")
    if plane.dtype == np.uint16:
        values = plane.astype(np.float64) * (display_max / 65535)
    else:
        values = plane
    grayscale = np.clip((values - window_min) * (255 / (window_max - window_min)),
                        0, 255).astype(np.uint8)
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

    def append(self, volume, scan_id, frame_index, health, reconstruction=None, *, inplace=False,
               projection=None, planes=None):
        """Updates within a scan name the last projection of their window, and slices their planes."""
        if volume.ndim != 3 or not all(volume.shape) or not np.isfinite(volume).all():
            raise ValueError("expected a finite, nonempty 3D live volume")
        if volume.size * 2 > self.max_bytes:
            raise ValueError("display volume exceeds history byte limit; increase --history-mib")
        voxels = display_voxels(volume, self.display_max, inplace=inplace)
        voxels.flags.writeable = False
        with self.lock:
            identity = (scan_id, -1 if projection is None else projection)
            if self.state["scan_id"] is not None and identity <= (
                    self.state["scan_id"], self.state.get("projection", -1)):
                raise ValueError("live scan identity did not advance")
            version = self.state["version"] + 1
            meta = dict(version=version, scan_id=scan_id, shape=list(voxels.shape),
                        time_seconds=time.monotonic() - self.started, bytes=voxels.nbytes)
            if projection is not None:
                meta["projection"] = projection
                self.state["projection"] = projection
            if planes is not None:
                meta["planes"] = planes
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


def make_handler(history, output, control=None, pipeline_control=None):
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
                if pipeline_control is not None:
                    try:
                        progress["pipeline_control"] = pipeline_control.settings()
                    except Exception as error:
                        progress["pipeline_control_error"] = str(error)
                self.respond(json.dumps(dict(**progress, viewer=history.snapshot())).encode(),
                             "application/json")
            elif request.path == "/api/pipeline":
                if pipeline_control is None:
                    self.respond(b"Pipeline controls unavailable", "text/plain", 503)
                    return
                try:
                    self.respond(json.dumps(pipeline_control.settings()).encode(), "application/json")
                except Exception as error:
                    self.respond(str(error).encode(), "text/plain", 503)
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
                            "X-Volume-Format": "uint16-le-zyx",
                            "X-Display-Range": f"0,{history.display_max}"})
                    else:
                        axis = int(query.get("axis", ["0"])[0])
                        if axis not in (0, 1, 2):
                            raise ValueError("axis must be 0, 1 or 2")
                        index = int(query.get("index", [str(volume.shape[axis] // 2)])[0])
                        if not 0 <= index < volume.shape[axis]:
                            raise ValueError("slice outside volume")
                        window_max = float(query.get("window_max", [str(history.display_max)])[0])
                        window_min = float(query.get("window_min", ["0"])[0])
                        self.respond(png_slice(volume, axis, index, window_max, history.display_max,
                                               window_min), "image/png")
                except ValueError as error:
                    self.respond(str(error).encode(), "text/plain", 400)
            else:
                self.respond(b"Not found", "text/plain", 404)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/reconstruction", "/api/slices", "/api/pipeline",
                            "/api/pipeline/stop", "/api/pipeline/recommend"):
                self.respond(b"Not found", "text/plain", 404)
                return
            # Browser writes must originate from this viewer, including its loopback port.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond(b"Origin does not match this viewer", "text/plain", 403)
                return
            selected_control = pipeline_control if path.startswith("/api/pipeline") else control
            if selected_control is None:
                self.respond(b"Pipeline controls unavailable" if path.startswith("/api/pipeline") else
                             b"Reconstruction controls unavailable", "text/plain", 503)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("expected a JSON settings object of at most 16 KiB")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("expected application/json")
                options = json.loads(self.rfile.read(size))
                if path == "/api/pipeline/stop":
                    if options != {}:
                        raise ValueError("stop expects an empty JSON object")
                    state = selected_control.stop()
                elif path == "/api/pipeline/recommend":
                    state = selected_control.recommend(options)
                elif path == "/api/slices":
                    state = selected_control.configure_slices(options)
                else:
                    state = selected_control.configure(options)
                self.respond(json.dumps(state).encode(), "application/json")
            except PipelineConflict as error:
                self.respond(str(error).encode(), "text/plain", 409)
            except (ValueError, TypeError) as error:
                self.respond(str(error).encode(), "text/plain", 400)
            except Exception as error:
                self.respond(str(error).encode(), "text/plain", 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", nargs="+", help="the reconstruction device, or each one of a pull set")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, help="common pipeline restart control directory")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--budget", type=int, default=1048576)
    parser.add_argument("--output-mode", choices=("volume", "blocks", "slices"), default="volume")
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

    if args.output_mode == "blocks" and len(args.devices) > 1:
        parser.error("block output uses one reconstruction device")
    subscribe = tango_ucx.every if args.output_mode == "blocks" else tango_ucx.latest
    subscriptions = [subscribe(tango.DeviceProxy(device), memory="host", budget=args.budget,
                               batch=1, label="tomography-live-view") for device in args.devices]
    proxies = [tango.DeviceProxy(device) for device in args.devices]
    for proxy in proxies:
        proxy.set_timeout_millis(60000)
    control = ReconstructionControl(*proxies)
    description = subscriptions[0].description
    collector = OutputCollector.from_description(description, control.for_scan, storage_volumes=3)
    scan = json.loads(description["application_text"])
    buffering = scan.get("buffering", {})
    # One device publishes sliding-window updates and slices, several to a scan.
    streamed = collector is None and bool(
        buffering.get("output_mode") == "slices" or buffering.get("update_projections"))
    # Blocks arrive in order from one device; whole volumes come from any device of the set.
    sub = (subscriptions[0] if collector is not None else
           Updates(subscriptions[0], scan["scan_id"], scan["angles"],
                   update_interval(buffering, scan["angles"])) if streamed else
           NewestVolumes(subscriptions, scan["scan_id"]))
    if ((collector is not None) != (args.output_mode == "blocks") or
            (buffering.get("output_mode") == "slices") != (args.output_mode == "slices")):
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
                result = read_complete(sub, collector) if collector is not None else sub.read()
                if result is None:
                    if sub.outcome is not None:
                        break
                    continue
                if mailbox is not None:
                    mailbox.put(result, sub.health())
                    history.update(assembled_volumes=collector.completed)
                    result = None
                elif streamed:
                    volume, record = result
                    scan_id = int(record["scan_id"])
                    history.append(volume, scan_id, int(record["index"]), sub.health(),
                                   control.for_scan(scan_id), projection=int(record["projection"]),
                                   planes=plane_list(record["planes"]) if "planes" in record.dtype.names else None)
                    if args.delay:
                        time.sleep(args.delay)
                else:
                    volume, scan_id, frame_index, reconstructor = result
                    history.append(volume, scan_id, frame_index, sub.health(),
                                   control.for_scan(scan_id, reconstructor))
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
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(history, args.output, control,
                                         PipelineControl(args.control_dir) if args.control_dir else None))
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
