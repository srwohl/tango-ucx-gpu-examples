"""Verify GPU-published opaque frames on the host and persist their bytes and records."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from common import arguments, connect, frames, records, Results


def main():
    parser = arguments(__doc__)
    parser.add_argument("--output", type=Path, help="new archive directory; existing paths are refused")
    args = parser.parse_args()
    results = Results(args)
    sub, meta = connect(args, "host")
    try:
        with ExitStack() as stack:
            data = index_file = None
            if args.output:
                args.output.mkdir(parents=True, exist_ok=False)
                (args.output / "description.json").write_text(json.dumps(sub.description, indent=2))
                data = stack.enter_context((args.output / "payloads.bin").open("xb"))
                index_file = stack.enter_context((args.output / "records.jsonl").open("x"))
            for batch in frames(sub, args):
                for i, row in enumerate(records(batch, meta)):
                    payload = batch.payload(i)
                    if payload.nbytes != row[2]:
                        raise ValueError("transport length and application field disagree")
                    results.accept(row, np.count_nonzero(payload != row[1]),
                                   int(payload.sum(dtype=np.uint64)))
                    if data:
                        offset = data.tell()
                        data.write(memoryview(payload))
                        index_file.write(json.dumps(dict(index=row[0], seed=row[1], bytes=row[2],
                                                        timestamp=row[3], offset=offset)) + "\n")
                del payload, batch
            results.finish(sub)
    finally:
        sub.close()


if __name__ == "__main__":
    main()
