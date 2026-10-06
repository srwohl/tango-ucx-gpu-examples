"""Small unpaced opaque host publisher probe; run with the existing pixi Python."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from network import environment


def worker(args):
    import tango
    import tango_ucx
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    device = f'127.0.0.1:{port}/example/opaque/1#dbase=no'
    with tempfile.TemporaryFile(mode='w+') as log:
        server = subprocess.Popen([str(ROOT / 'build/opaque_publisher'), 'local', '-nodb',
            '-dlist', 'example/opaque/1', '-ORBendPoint', f'giop:tcp:127.0.0.1:{port}'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    proxy = tango.DeviceProxy(device)
                    proxy.set_timeout_millis(1000)
                    proxy.ping()
                    break
                except tango.DevFailed:
                    time.sleep(.05)
            else:
                raise RuntimeError('publisher readiness timeout')
            proxy.set_timeout_millis(30000)
            proxy.command_inout('Configure', json.dumps(dict(source='host', lengths=[args.size], budget=256 << 20)))
            connect_begin = time.perf_counter()
            sub = tango_ucx.every(proxy, memory='host', budget='256MiB', batch=args.batch, max_wait=.01)
            try:
                deadline = time.monotonic() + 30
                while not sub.health()['transport'] and time.monotonic() < deadline:
                    time.sleep(.005)
                connected = time.perf_counter()
                transport = sub.health()['transport']
                if not transport:
                    raise RuntimeError('missing transport')
                begin = time.perf_counter()
                proxy.command_inout('Start', args.frames)
                first_time = None
                last = None
                first_n = first_index = None
                count = batches = 0
                while True:
                    batch = sub.read(timeout=30)
                    now = time.perf_counter()
                    if batch is None:
                        if sub.outcome is None:
                            raise RuntimeError('read timeout')
                        break
                    n = len(batch.records)
                    if first_time is None:
                        first_time = now
                        first_n = n
                        first_index = int(batch.records[0]['index'])
                    count += n
                    batches += 1
                    last, last_time = batch, now
                    del batch
                end = time.perf_counter()
                health = sub.health()
                assert count == args.frames and sub.outcome == 'end', (count, sub.outcome)
                assert not health['failure'] and not health['quarantined_bytes'], health
                # Content checks occur after all timestamps; retain only the final batch.
                import numpy as np
                for sample in (last,):
                    for i in (0, len(sample.records) - 1):
                        row = sample.records[i]
                        index, seed = int(row['index']), int(row['seed'])
                        assert seed == (index * 17 + 3) & 255
                        payload = sample.payload(i)
                        assert payload.nbytes == args.size and np.all(payload == seed)
                assert first_index == 0
                assert int(last.records[-1]['index']) == args.frames - 1
                active = last_time - first_time
                result = dict(profile=args.profile, payload_size=args.size, requested_batch=args.batch,
                    frames=count, batches=batches, average_batch=count/batches, transport=transport,
                    connect_seconds=connected-connect_begin, first_frame_seconds=first_time-begin,
                    start_to_last_seconds=last_time-begin, start_to_end_seconds=end-begin,
                    active_seconds=active, active_MiB_s=(count-first_n)*args.size/(1<<20)/active if active > 0 else None,
                    active_frames_s=(count-first_n)/active if active > 0 else None, outcome=sub.outcome,
                    checked_frames=[int(last.records[0]['index']), count-1],
                    health=health, scope="Host source and receive, publisher memset plus UCX transfer and Python batch consumption; active interval excludes first batch and terminal end wait; final batch payload spot-check only")
                del payload, sample, last
                print(json.dumps(result), flush=True)
            finally:
                sub.close()
        except BaseException:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--profile', default='tcp', choices=('tcp', 'auto'))
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--frames', type=int, default=20000)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('transport-results.json'))
    args = parser.parse_args()
    if args.size < 1 or args.size > 16 << 20 or args.batch < 1 or args.frames < 2 or args.repeats < 1:
        parser.error("size must be 1..16MiB, batch/repeats positive, frames at least 2")
    if args.worker:
        worker(args)
        return
    results = []
    for repeat in range(args.repeats):
        for size, frames in ((2048,200000), (65536,32768), (1048576,2048)):
            for batch in (1,16):
                for profile in ('tcp','auto'):
                    env = environment(profile, 'lo' if profile == 'tcp' else None)
                    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--profile', profile, '--size', str(size), '--frames', str(frames), '--batch', str(batch)],
                        env=env, capture_output=True, text=True, timeout=90)
                    if run.returncode:
                        raise RuntimeError(run.stdout + run.stderr)
                    row = json.loads(next(line for line in run.stdout.splitlines() if line.startswith('{')))
                    row['repeat'] = repeat
                    results.append(row)
                    args.output.write_text(json.dumps(results, indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k != 'health'}), flush=True)

if __name__ == '__main__':
    main()
