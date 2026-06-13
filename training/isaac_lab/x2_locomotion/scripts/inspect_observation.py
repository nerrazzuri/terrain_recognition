"""inspect_observation.py (P4-M4-T2).

Build a sample observation from the canonical layout and print its structure + per-part
slices. Runnable WITHOUT torch / Isaac Lab — a sanity check that the observation contract
(order + dims, total 168) is consistent. Deployment must match this exactly.

Usage: python -m x2_locomotion.scripts.inspect_observation
"""
from __future__ import annotations

import numpy as np

from x2_locomotion.tasks.common.observations import (
    OBSERVATION_LAYOUT, OBSERVATION_DIM, assemble, Normalizer)


def main() -> int:
    parts = {name: np.zeros(dim) for name, dim in OBSERVATION_LAYOUT}
    # mark each part with its index so slices are visible in the printout
    offset = 0
    for i, (name, dim) in enumerate(OBSERVATION_LAYOUT):
        parts[name] = np.full(dim, float(i))
        offset += dim
    v = assemble(parts)
    norm = Normalizer.identity().normalize(v)
    print(f"observation dim = {OBSERVATION_DIM} (assembled {v.shape[0]})")
    start = 0
    for name, dim in OBSERVATION_LAYOUT:
        print(f"  [{start:>3}:{start + dim:>3}] {name} (dim {dim})")
        start += dim
    assert v.shape[0] == OBSERVATION_DIM and norm.shape[0] == OBSERVATION_DIM
    print("OK: observation layout is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
