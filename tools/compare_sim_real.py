#!/usr/bin/env python3
"""compare_sim_real.py (P5-M3-T7).

Compare the filtered-action traces from a sim log and a real-robot log and report the maximum
per-step divergence — a quick sim-to-real sanity check. Read-only.

Usage: python tools/compare_sim_real.py logs/sim.jsonl logs/real.jsonl
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_policy_runtime"))
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_common"))

from x2_policy_runtime.core.log_analysis import load_jsonl, max_action_divergence  # noqa: E402


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print("usage: compare_sim_real.py <sim.jsonl> <real.jsonl>")
        return 2
    sim = [r.get("filtered_action") or [] for r in load_jsonl(args[0])]
    real = [r.get("filtered_action") or [] for r in load_jsonl(args[1])]
    n = min(len(sim), len(real))
    worst = 0.0
    for i in range(n):
        worst = max(worst, max_action_divergence([sim[i]], [real[i]]))
    print(f"compared {n} aligned cycles")
    print(f"max filtered-action divergence: {worst:.4f} rad")
    return 0


if __name__ == "__main__":
    sys.exit(main())
