"""Read a persisted example archive independently of Tango or CUDA."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(directory):
    count, offset, last = 0, 0, -1
    description = json.loads((directory / "description.json").read_text())
    meta = json.loads(description["application_text"])
    with (directory / "payloads.bin").open("rb") as data, (directory / "records.jsonl").open() as records:
        for line in records:
            row = json.loads(line)
            index, size, seed = row["index"], row["bytes"], row["seed"]
            if (index <= last or row["offset"] != offset or row["timestamp"] <= 0 or
                    size != meta["lengths"][index % len(meta["lengths"])] or seed != (index * 17 + 3) & 255):
                raise ValueError(f"invalid archive record: {row}")
            payload = data.read(size)
            if len(payload) != size or np.any(np.frombuffer(payload, dtype=np.uint8) != seed):
                raise ValueError(f"invalid archived bytes for frame {index}")
            count, offset, last = count + 1, offset + size, index
        if data.read(1):
            raise ValueError("unindexed trailing bytes")
    if not count:
        raise ValueError("empty archive")
    return {"archive_verified_frames": count, "payload_bytes": offset}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))
