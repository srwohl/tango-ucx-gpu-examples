"""Small FBP pipeline comparison using the unchanged demo and device server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

TOMOGRAPHY = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('pipeline-results.json'))
    args = parser.parse_args()
    if args.scans < 1 or args.repeats < 1:
        parser.error('scans and repeats must be positive')
    results = []
    for repeat in range(args.repeats):
        for memory in ('host', 'gpu'):
            for profile in ('tcp', 'auto'):
                with tempfile.TemporaryDirectory(prefix='tomography-probe-') as tmp:
                    output = Path(tmp) / 'run'
                    command = [sys.executable, str(TOMOGRAPHY / 'demo.py'),
                               '--network', profile, '--algorithm', 'fbp',
                               '--sinogram-memory', memory, '--output-mode', 'blocks',
                               '--slices-per-block', '8', '--scan-period', '0',
                               '--scans', str(args.scans), '--live', '--no-browser',
                               '--view-port', '0', '--output', str(output)]
                    with (Path(tmp) / 'demo.log').open('w+') as log:
                        run = subprocess.run(command, stdout=log, stderr=log, timeout=180)
                        if run.returncode:
                            log.seek(0)
                            raise RuntimeError(log.read()[-8000:])
                    summary = json.loads((output / 'summary.json').read_text())
                    assert summary['completed_scans'] == args.scans, summary
                    assert summary['archived_frames'] == args.scans * 98, summary
                    row = dict(repeat=repeat, profile=profile, sinogram_memory=memory,
                               command=command[:-1] + ['<temporary-output>'],
                               elapsed_seconds=summary['elapsed_seconds'],
                               completed_scans=summary['completed_scans'],
                               archived_frames=summary['archived_frames'],
                               throughput=summary['throughput'], stages=summary['stages'],
                               workload=summary['workload'])
                    results.append(row)
                    args.output.write_text(json.dumps(results, indent=2) + '\n')
                    print(json.dumps(dict(profile=profile, sinogram_memory=memory,
                                          elapsed_seconds=row['elapsed_seconds'],
                                          throughput=row['throughput'])), flush=True)


if __name__ == '__main__':
    main()
