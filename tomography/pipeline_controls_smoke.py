"""GPU integration check of GUI requests, graceful restarts and saved run integrity."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-server', required=True, type=Path)
    parser.add_argument('--stop-during-restart', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pipeline-controls-') as temp:
        root = Path(temp)
        output = root / 'run'
        with (root / 'demo.log').open('w+') as log:
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).with_name('demo.py')),
                '--device-server', str(args.device_server.resolve()), '--loop',
                '--algorithm', 'fbp', '--live', '--no-browser', '--view-port', '0',
                '--scan-period', '0', '--output', str(output)], stdout=log, stderr=log)
            try:
                def wait_for(operation, timeout=90):
                    until = time.monotonic() + timeout
                    while time.monotonic() < until:
                        if process.poll() is not None:
                            raise RuntimeError('demo exited unexpectedly')
                        try:
                            value = operation()
                            if value:
                                return value
                        except (OSError, ValueError):
                            pass  # Same-port viewer briefly disconnects during restart.
                        time.sleep(.1)
                    raise TimeoutError('pipeline controls did not reach expected state')

                url = wait_for(lambda: json.loads((output / 'live-ready.json').read_text())['url'])

                def get():
                    with urllib.request.urlopen(url + '/api/status', timeout=3) as response:
                        return json.load(response)

                def post(options, endpoint='/api/pipeline'):
                    request = urllib.request.Request(url + endpoint, data=json.dumps(options).encode(),
                                                     headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=3) as response:
                        return json.load(response)

                def wait_state(predicate):
                    def read():
                        state = get()
                        return state if predicate(state) else None
                    return wait_for(read)

                wait_for(lambda: get()['viewer']['received_volumes'] >= 2)
                recommended = post(dict(transport_batch=16, processing_batch=16,
                                        processing_mode='batched'), '/api/pipeline/recommend')
                assert recommended['information']['detector_shape'] == [8, 64]
                assert recommended['options']['receive_budget_mib'] >= 1
                assert get()['pipeline_control']['active']['revision'] == 0
                if args.stop_during_restart:
                    post(dict(network='auto', transport_batch=16, processing_batch=16,
                              processing_mode='batched', receive_budget_mib=1))
                    post({}, '/api/pipeline/stop')
                    assert process.wait(timeout=60) == 0
                    assert not (output / 'run-0001').exists(), 'Stop should cancel the queued restart'
                    summary = json.loads((output / 'summary.json').read_text())
                    assert summary['archived_frames'] == summary['completed_scans'] * 98
                    print(json.dumps(dict(gui_stop_cancelled_restart=True,
                                          completed_scans=summary['completed_scans'], archive_verified=True)))
                    return
                # This passes the value schema but exceeds the safe live restart budget.
                post(dict(receive_budget_mib=.001))
                rejected = wait_state(lambda s: s['pipeline_control'].get('error'))
                assert rejected['pipeline_control']['active']['revision'] == 0
                post(dict(algorithm='fbp', filter='parzen', scale_factor=1.1), '/api/reconstruction')
                wait_for(lambda: get()['reconstruction_control']['active']['options']['scale_factor'] == 1.1)

                requested = dict(network='auto', transport_batch=16, processing_batch=16,
                                 processing_mode='batched', receive_budget_mib=1,
                                 sinogram_memory='host', output_mode='blocks',
                                 host_buffer_mib=1, pinned_buffer_mib=1, output_host_mib=1)
                post(requested)
                first = wait_state(lambda s: (
                    s['pipeline_control']['run_id'] == 1 and
                    s['pipeline_control']['phase'] == 'running' and s['viewer']['received_volumes'] >= 3))
                for key, value in requested.items():
                    assert first['pipeline_control']['active']['options'][key] == value
                assert first['reconstruction_control']['active']['options']['scale_factor'] == 1.1
                assert first['workload']['network']['net_devices'] == 'all'
                assert first['workload']['processing']['processing_mode'] == 'batched'

                post(dict(network='tcp', transport_batch=4, processing_batch=4,
                          processing_mode='scalar', sinogram_memory='gpu', output_mode='volume',
                          reconstructors=2))
                second = wait_state(lambda s: (
                    s['pipeline_control']['run_id'] == 2 and
                    s['pipeline_control']['phase'] == 'running' and s['viewer']['received_volumes'] >= 3))
                assert second['workload']['network']['profile'] == 'tcp'
                assert second['workload']['buffering']['sinogram_memory'] == 'gpu'
                assert second['workload']['processing']['transport_batch'] == 4
                # A pull set: two reconstruction devices share the scans and both keep live settings.
                assert len(second['stages']['reconstruct']['reconstructors']) == 2
                assert second['reconstruction_control']['active']['options']['scale_factor'] == 1.1
                post(dict(algorithm='fbp', filter='hann', scale_factor=1.2), '/api/reconstruction')
                wait_for(lambda: get()['viewer']['volumes'][-1]['reconstruction']['options']['scale_factor'] == 1.2)
                post(dict(pixels=64, slices=4, angles=64))
                third = wait_state(lambda s: (
                    s['pipeline_control']['run_id'] == 3 and
                    s['pipeline_control']['phase'] == 'running' and s['viewer']['received_volumes'] >= 3))
                assert third['workload']['detector_shape'] == [4, 64]
                assert third['workload']['projections_per_volume'] == 64
                post({}, '/api/pipeline/stop')
                assert process.wait(timeout=60) == 0
                summaries = [json.loads((path / 'summary.json').read_text())
                             for path in (output, output / 'run-0001', output / 'run-0002', output / 'run-0003')]
                for summary in summaries:
                    assert summary['completed_scans'] >= 1
                    assert summary['archived_frames'] == summary['completed_scans'] * summary['workload']['input_frames_per_volume']
                    assert not any(stage['failure'] or stage['input_failure'] or stage['quarantined_bytes']
                                   for stage in summary['stages'].values())
                print(json.dumps(dict(same_viewer_url=url, restarts=3, unsafe_restart_rejected=True,
                                      live_reconstruction_preserved=True,
                                      gui_stop_verified=True, scan_based_buffers_verified=True,
                                      completed_scans=[s['completed_scans'] for s in summaries],
                                      all_archives_verified=True)), flush=True)
            except BaseException:
                log.seek(0)
                print(log.read()[-12000:], file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == '__main__':
    main()
