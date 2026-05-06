#!/usr/bin/env python3
"""compare.py — compare C get_features output against Python features_from_parquet.py output."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

TOL = 1e-10


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def compare(c_path: str, py_path: str) -> int:
    c  = _read(c_path)
    py = _read(py_path)

    common = sorted(set(c.columns) & set(py.columns))
    ts_common = c.index.intersection(py.index)

    print(f"rows:   C={len(c)}  py={len(py)}  common={len(ts_common)}")
    print(f"cols:   C={len(c.columns)}  py={len(py.columns)}  common={len(common)}")
    print()

    diffs = []
    for col in common:
        a = py.loc[ts_common, col].astype(float)
        b = c.loc[ts_common, col].astype(float)
        ok = a.notna() & b.notna()
        n = ok.sum()
        if n == 0:
            diffs.append((col, "-", 0, 0, 0, 0))
            continue
        d = (a[ok] - b[ok]).abs()
        exact = (d < TOL).sum()
        close = ((d >= TOL) & (d < 1e-6)).sum()
        big   = (d >= 1e-6).sum()
        nan_mm = (a.isna() != b.isna()).sum()
        diffs.append((col, f"{d.max():.2e}", exact, close, big, nan_mm))

    print(f"{'column':<22s} {'max_diff':>10s} {'exact':>6s} {'~close':>6s} {'big':>6s} {'NaN≠':>5s}")
    print("-" * 60)
    for col, mx, ex, cl, bg, nm in diffs:
        print(f"{col:<22s} {mx:>10s} {ex:>6d} {cl:>6d} {bg:>6d} {nm:>5d}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <c_output> <python_output>")
        print("  Supports .parquet and .csv")
        sys.exit(1)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
