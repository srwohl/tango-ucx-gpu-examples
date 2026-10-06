"""Four real Tango device servers, a concurrent compressed archive, and a volume writer."""
import argparse
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import signal
import threading
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from network import environment
from scan import from_hdf5, generate, hdf5_dimensions, selection
from reconstruction import ALGORITHMS, FILTERS, configuration, reference_reconstruction
from host_buffering import memory_plan
from output_blocks import OutputCollector, read_complete
from pipeline_control import atomic_json, validate_options

DEFAULT_WORKLOAD = dict(pixels=64, slices=8, angles=96, budget=262144, scan_period=1)
STRESS_WORKLOAD = dict(pixels=128, slices=32, angles=360, budget=8388608, scan_period=0)
GPU_STRESS_WORKLOAD = dict(pixels=256, slices=128, angles=720, budget=8388608, scan_period=0)


def launch_environment(args):
    # Keep the local TCP default, but preserve an explicitly configured interface.
    devices = args.net_devices
    if devices is None and args.network == "tcp" and not (
            os.environ.get("TANGO_UCX_UCX_NET_DEVICES") or os.environ.get("UCX_NET_DEVICES")):
        devices = "lo"
    return environment(args.network, devices)


def device_command(args, role, output, port, name, devices):
    gpu = args.gpu if role == "source" else getattr(args, f"{role}_gpu")
    command = [str(args.device_server.resolve()), "local", "-nodb", "-dlist", name,
               "-ORBendPoint", f"giop:tcp:127.0.0.1:{port}",
               "--role", role, "--scan", str(output / "scan" / "scan.json"),
               "--gpu", str(gpu), "--budget", str(args.budget),
               "--allow-gpu-over-tcp", "0" if args.network == "rdma" else "1",
               "--transport-batch", str(args.transport_batch),
               "--processing-batch", str(args.processing_batch),
               "--processing-mode", args.processing_mode,
               "--iterations", str(args.iterations), "--scans", str(args.scans),
               "--scan-period-ms", str(round(args.scan_period * 1000))]
    if role != "source":
        upstream = {"decompress": "source", "correct": "decompress",
                    "reconstruct": "correct"}[role]
        command.extend(["--upstream", devices[upstream]])
    if role == "correct":
        command.extend(["--delay-ms", str(args.correction_delay_ms)])
    return command


def workload_description(args, scan):
    detector_bytes = args.slices * args.pixels * (4 if scan["element"] == "f32" else 2)
    return dict(detector_shape=[args.slices, args.pixels],
                volume_shape=[args.slices, args.pixels, args.pixels],
                projections_per_volume=args.angles, input_frames_per_volume=args.angles + 2,
                detector_element=scan["element"], detector_frame_bytes=detector_bytes,
                corrected_frame_bytes=args.slices * args.pixels * 4,
                detector_bytes_per_volume=detector_bytes * (args.angles + 2),
                compressed_bytes_per_volume=sum(frame["bytes"] for frame in scan["frames"]),
                volume_bytes=args.slices * args.pixels**2 * 4,
                reconstruction=scan["reconstruction"], iterations=scan["reconstruction"]["iterations"],
                buffering=dict(scan.get("buffering", {}), memory_plan=memory_plan(scan, scan["reconstruction"])),
                scan_period_seconds=args.scan_period,
                receive_budget_bytes=args.budget, source=scan["source"],
                stage_gpus={role: getattr(args, f"{role}_gpu")
                            for role in ("decompress", "correct", "reconstruct")})


def streaming_rates(workload, scan, reports, elapsed, completed):
    """Mean application rates since acquisition Start, including startup and draining."""
    if elapsed <= 0:
        return {}
    rates = {}
    for role, report in reports.items():
        count = report["published"]
        if role == "source":
            cycles, remainder = divmod(count, workload["input_frames_per_volume"])
            size = cycles * workload["compressed_bytes_per_volume"] + sum(
                frame["bytes"] for frame in scan["frames"][:remainder])
        else:
            frame_size = dict(decompress=workload["detector_frame_bytes"],
                              correct=workload["corrected_frame_bytes"],
                              reconstruct=report.get("payload_bytes", workload["volume_bytes"]))[role]
            size = count * frame_size
        rates[role] = dict(frames_per_second=count / elapsed,
                           payload_mib_per_second=size / elapsed / 1024**2)
    return dict(mean_since_start=True, stages=rates, volumes_per_second=completed / elapsed)


def free_port():
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        return socket_.getsockname()[1]


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def verify_archive(output, scan, completed_scans):
    """Bounded-memory verification even after an arbitrarily long looping run."""
    with (output / "archive" / "payloads.bin").open("rb") as archived, \
         (output / "archive" / "records.jsonl").open() as records, \
         (output / "scan" / "compressed.bin").open("rb") as source:
        position = offset = 0
        for cycle in range(completed_scans):
            source.seek(0)
            for expected in scan["frames"]:
                line = records.readline()
                if not line:
                    raise ValueError("archive index is incomplete")
                record = json.loads(line)
                if (record["index"] != position or record["offset"] != offset or
                    record["scan_id"] != scan["scan_id"] + cycle or
                    record["calibration_id"] != scan["calibration_id"] + cycle):
                    raise ValueError("archive record identity mismatch")
                for key in ("kind", "projection", "theta", "bytes"):
                    if record[key] != expected[key]:
                        raise ValueError(f"archive metadata mismatch: frame {position}, {key}")
                size = expected["bytes"]
                if archived.read(size) != source.read(size):
                    raise ValueError("compressed archive differs from source bytes")
                position += 1
                offset += size
        if records.read(1) or archived.read(1):
            raise ValueError("unexpected trailing archived frames")
    return position


def run(args, output, stop_requested, env):
    import cupy as cp
    import numpy as np
    import tango
    import tango_ucx

    cp.cuda.Device(args.gpu).use()
    if args.hdf5:
        scan = from_hdf5(output / "scan", args.hdf5, **hdf5_options(args))
    else:
        scan = generate(output / "scan", args.pixels, args.slices, args.angles, args.gpu)
    scan["reconstruction"] = reconstruction_options(args)
    scan["buffering"] = buffering_options(args, scan)
    memory_plan(scan, scan["reconstruction"])
    (output / "scan" / "scan.json").write_text(json.dumps(scan, indent=2))
    workload = workload_description(args, scan)
    workload["network"] = dict(profile=args.network, tls=env["UCX_TLS"],
                               net_devices=env.get("TANGO_UCX_UCX_NET_DEVICES",
                                                   env.get("UCX_NET_DEVICES", "all")),
                               allow_gpu_over_tcp=args.network != "rdma")
    workload["processing"] = dict(transport_batch=args.transport_batch,
                                   processing_batch=args.processing_batch,
                                   processing_mode=args.processing_mode)
    with np.load(output / "scan" / "reference.npz") as reference:
        truth = reference["phantom"] if "phantom" in reference else None
        expected_volume = reference_reconstruction(
            reference["sinogram"], np.asarray(scan["theta"], np.float32), args.reconstruct_gpu,
            scan["reconstruction"], block_rows=args.slices_per_block if args.output_mode == "blocks" else None)
        reference_sinogram = reference["sinogram"].copy()
    if args.display_max is None:
        args.display_max = (0.01 if truth is not None else
                            max(float(np.percentile(expected_volume, 99.5)), 1e-6))
    workload["display_max"] = args.display_max
    print(json.dumps(dict(workload=workload)), flush=True)
    verified_options = scan["reconstruction"]
    last_settings = None
    devices, processes, logs = {}, {}, {}
    pressure = set()
    with ExitStack() as cleanup:
        volume_settings = cleanup.enter_context((output / "volume-settings.jsonl").open("w"))
        for role in ("source", "decompress", "correct", "reconstruct"):
            port = free_port()
            name = f"example/tomography/{role}"
            devices[role] = f"127.0.0.1:{port}/{name}#dbase=no"
            log = cleanup.enter_context((output / f"{role}.log").open("w"))
            command = device_command(args, role, output, port, name, devices)
            processes[role] = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
            cleanup.callback(stop, processes[role])
            logs[role] = output / f"{role}.log"
        try:
            proxies = {}
            for role, device in devices.items():
                until = time.monotonic() + 30
                while time.monotonic() < until and processes[role].poll() is None:
                    try:
                        proxy = tango.DeviceProxy(device)
                        proxy.set_timeout_millis(1000)
                        proxy.ping()
                        proxies[role] = proxy
                        break
                    except tango.DevFailed:
                        time.sleep(0.1)
                else:
                    raise RuntimeError(f"{role} did not become ready")
                proxy.set_timeout_millis(60000)
            (output / "devices.json").write_text(json.dumps(
                {role: dict(device=device, pid=processes[role].pid,
                            gpu=workload["stage_gpus"].get(role)) for role, device in devices.items()},
                indent=2))
            # Join every link before any publisher starts. Start downstream first.
            for role in ("reconstruct", "correct", "decompress", "source"):
                proxies[role].command_inout("Arm")
            volume = tango_ucx.every(proxies["reconstruct"], memory="host",
                                    budget=max(args.budget, memory_plan(scan, scan["reconstruction"])["gpu_output_bytes"] * 4 + 131072),
                                    batch=1, label="volume-writer")
            cleanup.callback(volume.close)
            collector = OutputCollector.from_description(volume.description, lambda scan_id: json.loads(
                proxies["reconstruct"].command_inout("ReconstructionForScan", str(scan_id))))
            archive_log = cleanup.enter_context((output / "archive.log").open("w"))
            archive = subprocess.Popen(
                [sys.executable, str(ROOT / "tomography" / "archive.py"), devices["source"],
                 "--output", str(output / "archive"), "--ready", str(output / "archive-ready.json"),
                 "--budget", str(args.budget)], env=env, stdout=archive_log, stderr=archive_log,
                start_new_session=True)
            cleanup.callback(stop, archive)
            until = time.monotonic() + 30
            while not (output / "archive-ready.json").exists():
                if archive.poll() is not None or time.monotonic() >= until:
                    raise RuntimeError("compressed archive did not become ready")
                time.sleep(0.05)
            live_url = None
            if args.live:
                live_log = cleanup.enter_context((output / "live.log").open("w"))
                viewer_command = [sys.executable, str(ROOT / "tomography" / "viewer.py"), devices["reconstruct"],
                     "--output", str(output), "--port", str(args.view_port),
                     "--budget", str(max(args.budget, memory_plan(scan, scan["reconstruction"])["gpu_output_bytes"] * 4 + 131072)),
                     "--output-mode", args.output_mode,
                     "--delay", str(args.viewer_delay),
                     "--history-volumes", str(args.view_history_volumes),
                     "--history-mib", str(args.view_history_mib),
                     "--display-max", str(args.display_max)]
                if getattr(args, "_control_dir", None) is not None:
                    viewer_command.extend(["--control-dir", str(args._control_dir)])
                viewer = subprocess.Popen(viewer_command, env=env, stdout=live_log, stderr=live_log,
                    start_new_session=True)
                cleanup.callback(stop, viewer)
                until = time.monotonic() + 30
                while not (output / "live-ready.json").exists():
                    if viewer.poll() is not None or time.monotonic() >= until:
                        raise RuntimeError("live viewer did not become ready")
                    time.sleep(0.05)
                live_url = json.loads((output / "live-ready.json").read_text())["url"]
                print(json.dumps(dict(live_view=live_url, output=str(output))), flush=True)
                if not args.no_browser:
                    webbrowser.open(live_url)
            for role in ("reconstruct", "correct", "decompress", "source"):
                proxies[role].command_inout("Start")
            if getattr(args, "_control_dir", None) is not None:
                state_path = args._control_dir / "pipeline-control.json"
                state = json.loads(state_path.read_text())
                state.update(active=dict(options=pipeline_options(args),
                                         revision=state["requested"]["revision"]),
                             phase="running", run_output=str(output),
                             fixed_geometry=bool(args.hdf5),
                             workload=workload,
                             gpu_count=cp.cuda.runtime.getDeviceCount())
                atomic_json(state_path, state)
            started = last_volume = time.monotonic()
            reconstruction = None
            completed = 0
            draining = False
            reports = {}
            error = None
            restart_plan = None

            def publish_status(phase, elapsed=None):
                try:
                    archived = json.loads((output / "archive" / "progress.json").read_text())
                except FileNotFoundError:
                    archived = {}
                temporary = output / "status.tmp"
                if elapsed is None:
                    elapsed = time.monotonic() - started
                temporary.write_text(json.dumps(dict(
                    phase=phase, completed_scans=completed, elapsed_seconds=elapsed,
                    workload=workload,
                    throughput=streaming_rates(workload, scan, reports, elapsed, completed),
                    stages=reports, archive=archived)))
                temporary.replace(output / "status.json")

            def read_volume():
                if collector is not None:
                    collected = read_complete(volume, collector, timeout=0.05)
                    return None if collected is None else collected[0]
                batch = volume.read(timeout=0.05)
                if batch is None:
                    return None
                if batch.frames != 1:
                    raise ValueError("expected one reconstructed volume per frame")
                record = batch.records[0]
                if (int(record["index"]) != completed or
                    int(record["scan_id"]) != scan["scan_id"] + completed or
                    int(record["calibration_id"]) != scan["calibration_id"] + completed):
                    raise ValueError("missing or out-of-order reconstructed scan")
                return batch.array[0].copy()

            while True:
                control_dir = getattr(args, "_control_dir", None)
                if control_dir is not None and (control_dir / "pipeline-stop.json").exists():
                    stop_requested.set()
                    restart_plan = None
                    (control_dir / "pipeline-request.json").unlink(missing_ok=True)
                if not draining and getattr(args, "_control_dir", None) is not None:
                    request_path = args._control_dir / "pipeline-request.json"
                    if request_path.exists() and not stop_requested.is_set():
                        state_path = args._control_dir / "pipeline-control.json"
                        state = json.loads(state_path.read_text())
                        try:
                            request = json.loads(request_path.read_text())
                            settings = json.loads(proxies["reconstruct"].command_inout("GetReconstruction"))
                            restart_plan = pipeline_args(args, request["options"],
                                                         settings["requested"]["options"])
                            state.update(requested=request, phase="restarting", error=None)
                            atomic_json(state_path, state)
                        except FileNotFoundError:
                            pass  # A concurrent Stop cancels a queued restart.
                        except (ValueError, TypeError, OverflowError) as invalid:
                            state.update(requested=state["active"], phase="running", error=str(invalid))
                            atomic_json(state_path, state)
                            request_path.unlink(missing_ok=True)
                if (stop_requested.is_set() or restart_plan is not None) and not draining:
                    proxies["source"].command_inout("Stop")
                    draining = True
                    print("Finishing acquisition and draining the pipeline…", flush=True)
                result = read_volume()
                if result is None and volume.outcome is not None:
                    break
                if result is not None:
                    last_settings = json.loads(proxies["reconstruct"].command_inout(
                        "ReconstructionForScan", str(scan["scan_id"] + completed)))
                    options = last_settings["options"]
                    if options != verified_options:
                        expected_volume = reference_reconstruction(
                            reference_sinogram, np.asarray(scan["theta"], np.float32),
                            args.reconstruct_gpu, options,
                            block_rows=options["slices_per_block"] if args.output_mode == "blocks" else None)
                        verified_options = options
                    np.testing.assert_allclose(result, expected_volume, rtol=3e-4, atol=2e-6)
                    error = (float(np.linalg.norm(result-truth)/np.linalg.norm(truth))
                             if truth is not None else None)
                    # Live tuning can intentionally change scale, smoothing or convergence.
                    if not np.isfinite(result).all():
                        raise ValueError("reconstruction contains nonfinite values")
                    if error is not None and last_settings["revision"] == 0 and error > args.max_phantom_error:
                        raise ValueError(f"reconstruction does not reproduce the phantom: relative L2={error}")
                    volume_settings.write(json.dumps(dict(last_settings, relative_l2_error=error)) + "\n")
                    volume_settings.flush()
                    workload["reconstruction"] = options
                    workload["buffering"]["memory_plan"] = memory_plan(scan, options)
                    workload["iterations"] = options["iterations"]
                    reconstruction = result
                    completed += 1
                    last_volume = time.monotonic()
                    temporary = output / "volume.tmp.npy"
                    np.save(temporary, reconstruction)
                    temporary.replace(output / "volume.npy")
                    print(json.dumps(dict(completed_scan=completed, shape=list(result.shape))), flush=True)
                reports = {role: json.loads(proxy.command_inout("Report"))
                           for role, proxy in proxies.items()}
                for role, report in reports.items():
                    if report["pressure"]:
                        pressure.add(role)
                    if report["failure"] or report["input_failure"] or report["quarantined_bytes"]:
                        raise RuntimeError(f"{role} failed: {report}")
                if any(process.poll() is not None for process in processes.values()):
                    raise RuntimeError("a device server exited unexpectedly")
                if archive.poll() not in (None, 0):
                    raise RuntimeError("compressed archive failed")
                if args.live and viewer.poll() is not None:
                    raise RuntimeError("live viewer exited unexpectedly")
                if time.monotonic() - last_volume > max(120, args.scan_period * 2):
                    raise TimeoutError("pipeline stopped producing volumes")
                publish_status("draining" if draining else "running")
            if volume.outcome != "end" or reconstruction is None:
                raise RuntimeError(f"volume writer ended without a volume: {volume.health()}")
            if collector is not None:
                collector.finish()
            if args.scans and not draining and completed != args.scans:
                raise ValueError("pipeline ended before all requested scans")
            reports = {role: json.loads(proxy.command_inout("Report"))
                       for role, proxy in proxies.items()}
            archive.wait(timeout=20)
            if archive.returncode:
                raise RuntimeError("compressed archive failed")
            elapsed = time.monotonic() - started
            expected_counts = dict(source=(args.angles+2)*completed, decompress=(args.angles+2)*completed,
                                   correct=args.angles*completed,
                                   reconstruct=collector.index if collector is not None else completed)
            for role, expected in expected_counts.items():
                if (reports[role]["published"] != expected or
                    reports[role]["completed_scans"] != completed):
                    raise ValueError(f"{role} scan counts differ: {reports[role]}")
            archived_count = verify_archive(output, scan, completed)
            publish_status("finished", elapsed)
            if control_dir is not None and (control_dir / "pipeline-stop.json").exists():
                state_path = control_dir / "pipeline-control.json"
                state = json.loads(state_path.read_text())
                state.update(phase="finished", requested=state.get("active"), error=None)
                atomic_json(state_path, state)
            live_report = None
            if args.live:
                until = time.monotonic() + max(20, args.viewer_delay * (completed+1))
                while not (output / "live-summary.json").exists():
                    if viewer.poll() is not None or time.monotonic() >= until:
                        raise RuntimeError("live viewer did not reach End")
                    time.sleep(0.05)
                with urllib.request.urlopen(live_url + "/api/status", timeout=5) as response:
                    live_report = json.load(response)["viewer"]
                if (live_report["failure"] or live_report["outcome"] != "end" or
                    live_report["received_volumes"] < 1):
                    raise RuntimeError(f"live viewer failed: {live_report}")
                for axis, length in enumerate(reconstruction.shape):
                    with urllib.request.urlopen(
                        f"{live_url}/api/slice?axis={axis}&index={length//2}", timeout=5) as response:
                        from io import BytesIO
                        from PIL import Image
                        with Image.open(BytesIO(response.read())) as image:
                            expected_shape = [n for i,n in enumerate(reconstruction.shape) if i != axis]
                            if image.size != tuple(reversed(expected_shape)):
                                raise ValueError("live slice dimensions differ from volume")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            panels = [(reconstruction, f"{verified_options['algorithm'].upper()} reconstruction")]
            if truth is not None:
                panels.insert(0, (truth, "Phantom"))
            figure, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3), squeeze=False)
            for axis, (values, title) in zip(axes[0], panels):
                axis.imshow(values[len(values) // 2], cmap="gray", vmin=0, vmax=args.display_max)
                axis.set_title(title)
                axis.axis("off")
            figure.tight_layout()
            figure.savefig(output / "reconstruction.png", dpi=140)
            plt.close(figure)
            summary = dict(device_servers=4, completed_scans=completed, archived_frames=archived_count,
                           live_view=live_report, stopped=draining,
                           reconstruction_shape=list(reconstruction.shape), relative_l2_error=error,
                           max_phantom_error=args.max_phantom_error if truth is not None else None,
                           reconstruction_settings=last_settings,
                           elapsed_seconds=elapsed, pressure_observed=sorted(pressure),
                           workload=workload,
                           throughput=streaming_rates(workload, scan, reports, elapsed, completed),
                           volume_transport=volume.health()["transport"], stages=reports,
                           output=str(output))
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2), flush=True)
            summary["_restart_plan"] = restart_plan
            return summary
        except BaseException:
            for role, path in {**logs, "archive": output / "archive.log", "live": output / "live.log"}.items():
                if path.exists():
                    print(f"{role} log:\n{path.read_text()[-5000:]}", file=sys.stderr)
            raise


def reconstruction_options(args):
    return configuration(args.algorithm, args.recon_filter, args.iterations, args.recon_threads,
                         relaxation=args.relaxation, min_constraint=args.min_constraint,
                         max_constraint=args.max_constraint, filter_cutoff=args.filter_cutoff,
                         center=args.center, gaussian_fwhm=args.gaussian_fwhm,
                         scale_factor=args.scale_factor, slices_per_block=args.slices_per_block)


def pipeline_options(args):
    keys = ("network", "net_devices", "transport_batch", "processing_batch", "processing_mode",
            "sinogram_memory", "host_buffer_mib", "pinned_buffer_mib", "output_mode",
            "output_host_mib", "gpu", "decompress_gpu", "correct_gpu", "reconstruct_gpu", "scan_period",
            "pixels", "slices", "angles")
    return dict({key: getattr(args, key) for key in keys}, receive_budget_mib=args.budget / 1024**2)


def pipeline_args(args, options, reconstruction=None):
    """Validate a prospective restart before interrupting the working acquisition."""
    options = validate_options(options)
    if args.hdf5 and any(options.get(key, getattr(args, key)) != getattr(args, key)
                         for key in ("pixels", "slices", "angles")):
        raise ValueError("HDF5 detector dimensions come from the selected file")
    result = copy.copy(args)
    for key, value in options.items():
        if key == "receive_budget_mib":
            result.budget = int(value * 1024**2)
        else:
            setattr(result, key, value)
    if reconstruction is not None:
        result.algorithm = reconstruction["algorithm"]
        for key, attr in (("filter", "recon_filter"), ("threads", "recon_threads"),
                          ("iterations", "iterations"), ("relaxation", "relaxation"),
                          ("min_constraint", "min_constraint"), ("max_constraint", "max_constraint"),
                          ("filter_cutoff", "filter_cutoff"), ("center", "center"),
                          ("gaussian_fwhm", "gaussian_fwhm"), ("scale_factor", "scale_factor"),
                          ("slices_per_block", "slices_per_block")):
            if reconstruction.get(key) is not None or key in ("max_constraint", "filter_cutoff", "center"):
                setattr(result, attr, reconstruction[key])
    if result.output_mode == "blocks" and not result.slices_per_block:
        result.slices_per_block = result.slices
    recon = reconstruction_options(result)
    memory_plan(dict(rows=result.slices, columns=result.pixels, angles=result.angles,
                     buffering=buffering_options(result)), recon)
    if result.live and result.slices * result.pixels**2 > result.view_history_mib * 1024**2:
        raise ValueError("display volume exceeds the viewer history byte limit")
    if result.live and result.output_mode == "blocks" and result.slices * result.pixels**2 * 12 > result.output_host_mib * 1024**2:
        raise ValueError("block viewer requires host capacity for three float volumes")
    # Leave room for UCX bookkeeping as well as two complete fixed-size batches.
    minimum = (128 << 10) + 2 * result.transport_batch * (result.slices * result.pixels * 4 + 2048)
    if result.budget < minimum:
        raise ValueError(f"receive budget is too small for these batches; use at least {minimum / 1024**2:.3f} MiB")
    return result


def buffering_options(args, scan=None):
    result = dict(sinogram_memory=args.sinogram_memory,
                host_budget_bytes=args.host_buffer_mib * 1024**2,
                pinned_budget_bytes=args.pinned_buffer_mib * 1024**2)
    if args.output_mode == "blocks":
        rows = scan["rows"] if scan is not None else args.slices
        result.update(output_mode="blocks", output_block_rows=min(args.slices_per_block, rows),
                      output_host_budget_bytes=args.output_host_mib * 1024**2,
                      transport_budget_bytes=args.budget)
    return result


def optional_float(value):
    return None if value.lower() == "none" else float(value)


def hdf5_options(args):
    return {name: getattr(args, name) for name in
            ("data_path", "flat_path", "dark_path", "theta_path", "theta_units", "sino", "proj")}


def parse_selection(value):
    try:
        return selection(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-server", type=Path, default=ROOT / "build-pipeline" / "pipeline_device")
    parser.add_argument("--output", type=Path, help="new output directory; temporary if omitted")
    parser.add_argument("--hdf5", type=Path, help="raw HDF5 detector scan; omit to generate a phantom")
    parser.add_argument("--data-path", default="/exchange/data", help="HDF5 projection dataset (angle, row, column)")
    parser.add_argument("--flat-path", default="/exchange/data_white", help="HDF5 flat image or stack")
    parser.add_argument("--dark-path", default="/exchange/data_dark", help="HDF5 dark image or stack")
    parser.add_argument("--theta-path", default="/exchange/theta", help="HDF5 angle dataset")
    parser.add_argument("--theta-units", choices=("auto", "degrees", "radians"), default="auto",
                        help="auto reads theta units attribute, defaulting to degrees")
    parser.add_argument("--sino", type=parse_selection, help="HDF5 detector rows START:STOP[:STEP]")
    parser.add_argument("--proj", type=parse_selection, help="HDF5 projections START:STOP[:STEP]")
    parser.add_argument("--display-max", type=float,
                        help="positive display upper bound; default 0.01 for phantom, automatic for HDF5")
    presets = parser.add_mutually_exclusive_group()
    presets.add_argument("--stress", action="store_true",
                         help="128 pixels, 32 slices, 360 projections, 8 MiB rings, no scan pause; flags override")
    presets.add_argument("--gpu-stress", action="store_true",
                         help="256 pixels, 128 slices, 720 projections, 8 MiB rings, no scan pause; flags override")
    parser.add_argument("--pixels", type=int, help="detector columns / volume width (default 64)")
    parser.add_argument("--slices", type=int, help="detector rows / volume depth (default 8)")
    parser.add_argument("--angles", type=int, help="projections per volume (default 96)")
    parser.add_argument("--iterations", type=int, default=40, help="SIRT iterations (default 40)")
    parser.add_argument("--slices-per-block", type=int, default=0,
                        help="slices per reconstruction backend call (default 0: all slices)")
    parser.add_argument("--sinogram-memory", choices=("gpu", "host"), default="gpu",
                        help="retained corrected scan location (default gpu)")
    parser.add_argument("--output-mode", choices=("volume", "blocks"), default="volume",
                        help="GPU publications: whole volume or bounded slice blocks (default volume)")
    parser.add_argument("--output-host-mib", type=int, default=1024,
                        help="per-consumer host output capacity MiB; block viewer needs three volumes")
    parser.add_argument("--host-buffer-mib", type=int, default=1024,
                        help="maximum retained host sinogram MiB (default 1024)")
    parser.add_argument("--pinned-buffer-mib", type=int, default=128,
                        help="maximum CUDA-pinned transfer staging MiB (default 128)")
    parser.add_argument("--algorithm", choices=ALGORITHMS,
                        help="reconstruction method (default gridrec with --live, otherwise sirt)")
    parser.add_argument("--recon-filter", choices=FILTERS, default="ram-lak",
                        help="analytic reconstruction filter; ignored by SIRT")
    parser.add_argument("--recon-threads", type=int, default=4,
                        help="CPU threads for TomoPy GridRec (default 4)")
    parser.add_argument("--relaxation", type=float, default=1.0,
                        help="SIRT update strength in (0, 2), default 1")
    parser.add_argument("--min-constraint", type=optional_float, default=0.0,
                        help="SIRT lower bound; 'none' disables non-negativity (default 0)")
    parser.add_argument("--max-constraint", type=optional_float,
                        help="SIRT upper bound (default none)")
    parser.add_argument("--filter-cutoff", type=float,
                        help="FBP hann/shepp-logan FilterD in (0, 1]; default backend value 1")
    parser.add_argument("--center", type=float,
                        help="GridRec rotation axis in detector pixels (default columns / 2)")
    parser.add_argument("--gaussian-fwhm", type=float, default=0.0,
                        help="post-reconstruction 3D Gaussian FWHM in voxels (default 0: off)")
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="positive output multiplier after smoothing (default 1)")
    parser.add_argument("--max-phantom-error", type=float, default=0.65,
                        help="maximum relative L2 against raw phantom; adjust for deliberate tuning (default 0.65)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU for scan preparation and default for every processing stage (default 0)")
    for role in ("decompress", "correct", "reconstruct"):
        parser.add_argument(f"--{role}-gpu", type=int,
                            help=f"GPU for {role}; overrides --gpu for this stage")
    parser.add_argument("--network", "--profile", choices=("tcp", "rdma", "auto"), default="tcp",
                        help="UCX profile: local TCP, RC/CUDA without TCP fallback, or automatic selection")
    parser.add_argument("--net-devices", help="UCX interface or HCA:port; TCP defaults to lo, others to UCX selection")
    parser.add_argument("--transport-batch", type=int, choices=range(1, 17), default=1,
                        help="maximum frames per UCX receive batch")
    parser.add_argument("--processing-batch", type=int, choices=range(1, 17), default=1,
                        help="maximum frames per batched decompression call")
    parser.add_argument("--processing-mode", choices=("scalar", "batched"), default="scalar",
                        help="scalar calls or nvCOMP batch decompression")
    parser.add_argument("--budget", type=int, help="receive budget in bytes (default 262144)")
    parser.add_argument("--correction-delay-ms", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scans", type=int, default=1, help="scans to replay without restarting devices")
    mode.add_argument("--loop", action="store_true", help="keep replaying until Ctrl-C")
    parser.add_argument("--scan-period", type=float, help="minimum seconds between scan starts (default 1)")
    parser.add_argument("--live", action="store_true", help="serve a browser view on a latest subscription")
    parser.add_argument("--view-port", type=int, default=8765, help="HTTP port; 0 chooses a free port")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening it")
    parser.add_argument("--viewer-delay", type=float, default=0, help="slow the viewer to exercise latest delivery")
    parser.add_argument("--view-history-volumes", type=int, default=32, help="maximum buffered display volumes")
    parser.add_argument("--view-history-mib", type=int, default=64, help="maximum retained display volume MiB")
    args = parser.parse_args(argv)
    for role in ("decompress", "correct", "reconstruct"):
        if getattr(args, f"{role}_gpu") is None:
            setattr(args, f"{role}_gpu", args.gpu)
    if args.hdf5:
        if any(getattr(args, name) is not None for name in ("pixels", "slices", "angles")):
            parser.error("HDF5 dimensions come from the file; select rows with --sino and projections with --proj")
        try:
            dimensions = hdf5_dimensions(args.hdf5, data_path=args.data_path, sino=args.sino, proj=args.proj)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        args.pixels, args.slices, args.angles = dimensions["columns"], dimensions["rows"], dimensions["angles"]
    elif (args.sino is not None or args.proj is not None or args.theta_units != "auto" or
          args.data_path != "/exchange/data" or args.flat_path != "/exchange/data_white" or
          args.dark_path != "/exchange/data_dark" or args.theta_path != "/exchange/theta"):
        parser.error("HDF5 dataset and selection flags require --hdf5")
    if args.algorithm is None:
        args.algorithm = "gridrec" if args.live else "sirt"
    workload_defaults = (GPU_STRESS_WORKLOAD if args.gpu_stress else
                         STRESS_WORKLOAD if args.stress else DEFAULT_WORKLOAD)
    for name, value in workload_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    try:
        reconstruction_options(args)
        memory_plan(dict(rows=args.slices, columns=args.pixels, angles=args.angles,
                         buffering=buffering_options(args)), reconstruction_options(args))
        if args.output_mode == "blocks" and args.live and args.slices * args.pixels**2 * 4 * 3 > args.output_host_mib * 1024**2:
            raise ValueError("block viewer assembly/display snapshots exceed --output-host-mib (three float volumes)")
        if args.center is not None and args.center > args.pixels:
            raise ValueError("center must lie within the detector [0, pixels]")
        import math
        if not math.isfinite(args.max_phantom_error) or args.max_phantom_error <= 0:
            raise ValueError("max phantom error must be finite and positive")
        if args.display_max is not None and (not math.isfinite(args.display_max) or args.display_max <= 0):
            raise ValueError("display max must be finite and positive")
    except ValueError as error:
        parser.error(str(error))
    if args.scans < 1 or args.scan_period < 0 or args.viewer_delay < 0 or not 0 <= args.view_port <= 65535:
        parser.error("scans must be positive; timing and port must be nonnegative")
    if args.loop:
        args.scans = 0
    if min(args.pixels, args.slices, args.angles, args.iterations, args.budget, args.recon_threads,
           args.host_buffer_mib, args.pinned_buffer_mib, args.output_host_mib,
           args.view_history_volumes, args.view_history_mib) < 1 or min(
               args.gpu, args.decompress_gpu, args.correct_gpu, args.reconstruct_gpu) < 0:
        parser.error("dimensions, iterations and budget must be positive; GPU must be nonnegative")
    if args.live and args.slices * args.pixels**2 > args.view_history_mib * 1024**2:
        parser.error("display volume exceeds --view-history-mib; increase the history byte limit")
    if args.output and args.output.resolve().exists():
        parser.error("--output must name a new directory")
    return args


def main():
    args = parse_args()
    stop_requested = threading.Event()
    def finish_acquisition(*_):
        stop_requested.set()
    signal.signal(signal.SIGINT, finish_acquisition)
    signal.signal(signal.SIGTERM, finish_acquisition)
    inherited_devices = {key: os.environ.get(key) for key in
                         ("UCX_NET_DEVICES", "TANGO_UCX_UCX_NET_DEVICES")}

    def supervise(root):
        current = args
        previous = None
        run_id = 0
        if current.live:
            current._control_dir = root
            if current.view_port == 0:
                current.view_port = free_port()
            atomic_json(root / "pipeline-control.json", dict(
                requested=dict(options=pipeline_options(current), revision=0),
                active=None, phase="restarting", run_id=run_id,
                run_output=str(root), error=None))
        while True:
            if current.live and (root / "pipeline-stop.json").exists():
                stop_requested.set()
                (root / "pipeline-request.json").unlink(missing_ok=True)
            if stop_requested.is_set() and run_id:
                state_path = root / "pipeline-control.json"
                state = json.loads(state_path.read_text())
                state.update(phase="finished", requested=state.get("active"))
                atomic_json(state_path, state)
                break
            output = root if run_id == 0 else root / f"run-{run_id:04d}"
            if run_id:
                output.mkdir()
            for key, value in inherited_devices.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            env = launch_environment(current)
            os.environ.update(env)
            for location in env["PYTHONPATH"].split(":"):
                if location and location not in sys.path:
                    sys.path.insert(0, location)
            try:
                result = run(current, output, stop_requested, env)
            except Exception as failure:
                if not current.live:
                    raise
                state_path = root / "pipeline-control.json"
                state = json.loads(state_path.read_text())
                if (root / "pipeline-stop.json").exists():
                    stop_requested.set()
                if stop_requested.is_set():
                    state.update(phase="finished", requested=state.get("active"))
                    atomic_json(state_path, state)
                    break
                if previous is None or stop_requested.is_set():
                    state.update(phase="failed", error=str(failure))
                    atomic_json(state_path, state)
                    raise
                # Recover the last working settings if a new transport cannot start.
                current, previous = previous, None
                run_id += 1
                state.update(phase="restarting", error=f"Restart failed; restoring previous settings: {failure}",
                             requested=dict(options=pipeline_options(current),
                                            revision=state["requested"]["revision"] + 1), run_id=run_id)
                atomic_json(state_path, state)
                continue
            next_args = result.get("_restart_plan")
            if current.live and (root / "pipeline-stop.json").exists():
                stop_requested.set()
                next_args = None
            if next_args is None or stop_requested.is_set():
                if current.live:
                    state_path = root / "pipeline-control.json"
                    state = json.loads(state_path.read_text())
                    state.update(phase="finished")
                    atomic_json(state_path, state)
                break
            previous, current = current, next_args
            current.no_browser = True
            run_id += 1
            state_path = root / "pipeline-control.json"
            state = json.loads(state_path.read_text())
            state.update(phase="restarting", run_id=run_id,
                         run_output=str(root / f"run-{run_id:04d}"))
            atomic_json(state_path, state)
            (root / "pipeline-request.json").unlink(missing_ok=True)

    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True)
        supervise(output)
    else:
        with tempfile.TemporaryDirectory(prefix="tango-ucx-tomography-") as temporary:
            supervise(Path(temporary))


if __name__ == "__main__":
    main()
