#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def load_meta(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def load_tensor(path: Path, count: int) -> List[float]:
    import array

    data = array.array("f")
    with path.open("rb") as f:
        data.frombytes(f.read())
    if len(data) != count:
        raise ValueError(f"{path}: expected {count} float32 values, got {len(data)}")
    return data.tolist()


def compare(ref: List[float], cand: List[float]) -> Tuple[float, float, float, float, int, int]:
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    ref_norm = 0.0
    cand_norm = 0.0
    nan_count = 0
    inf_count = 0
    for a, b in zip(ref, cand):
        if math.isnan(b):
            nan_count += 1
        if math.isinf(b):
            inf_count += 1
        diff = a - b
        abs_diff = abs(diff)
        max_abs = max(max_abs, abs_diff)
        sum_abs += abs_diff
        sum_sq += diff * diff
        dot += a * b
        ref_norm += a * a
        cand_norm += b * b
    size = max(len(ref), 1)
    mean_abs = sum_abs / size
    rmse = math.sqrt(sum_sq / size)
    cosine = 0.0
    if ref_norm > 0.0 and cand_norm > 0.0:
        cosine = dot / math.sqrt(ref_norm * cand_norm)
    return max_abs, mean_abs, rmse, cosine, nan_count, inf_count


def index_dir(root: Path) -> Dict[str, Path]:
    result = {}
    for json_path in root.glob("*.json"):
        meta = load_meta(json_path)
        result[json_path.stem] = root / meta["file"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_dir")
    parser.add_argument("cand_dir")
    parser.add_argument("--names", default="", help="comma separated file stems to compare")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    cand_dir = Path(args.cand_dir)
    ref_index = index_dir(ref_dir)
    cand_index = index_dir(cand_dir)

    if args.names:
        names = [x for x in args.names.split(",") if x]
    else:
        names = sorted(set(ref_index.keys()) & set(cand_index.keys()))

    if not names:
        raise SystemExit("no matching tensor files found")

    for name in names:
        ref_meta = load_meta(ref_dir / f"{name}.json")
        cand_meta = load_meta(cand_dir / f"{name}.json")
        if ref_meta["shape"] != cand_meta["shape"]:
            print(f"{name}: shape mismatch ref={ref_meta['shape']} cand={cand_meta['shape']}")
            continue
        count = int(ref_meta["elements"])
        ref = load_tensor(ref_dir / ref_meta["file"], count)
        cand = load_tensor(cand_dir / cand_meta["file"], count)
        max_abs, mean_abs, rmse, cosine, nan_count, inf_count = compare(ref, cand)
        print(
            f"{name}: shape={ref_meta['shape']} "
            f"maxAbs={max_abs:.8g} meanAbs={mean_abs:.8g} rmse={rmse:.8g} "
            f"cosine={cosine:.8g} nan={nan_count} inf={inf_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
