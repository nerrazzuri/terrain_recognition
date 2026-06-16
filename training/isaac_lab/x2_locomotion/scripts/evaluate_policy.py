"""evaluate_policy.py (P4-M4-T1) — success-rate report against the graduation criteria.

Computes per-terrain success rates and checks them against the Phase 4 graduation gate
(roadmap §8.11): flat >95%, rough >90%, 5 cm step >90%, 10 cm step >80%, stair-up >80%.

The threshold-checking logic (``check_graduation``) is pure and unit-tested; running actual
rollouts is BLOCKED on Isaac Lab + a trained checkpoint.
"""
from __future__ import annotations

import argparse
import sys

GRADUATION_THRESHOLDS = {
    "flat": 0.95, "rough": 0.90, "step_5cm": 0.90, "step_10cm": 0.80, "stairs_up": 0.80,
}


def check_graduation(success_rates: dict[str, float]) -> tuple[bool, list[str]]:
    """Return ``(passed, failures)`` comparing measured rates to the graduation gate."""
    failures = []
    for name, thresh in GRADUATION_THRESHOLDS.items():
        rate = success_rates.get(name)
        if rate is None:
            failures.append(f"{name}: not measured")
        elif rate < thresh:
            failures.append(f"{name}: {rate:.2f} < {thresh:.2f}")
    return (len(failures) == 0), failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args(argv)
    print(f"[evaluate] checkpoint={args.checkpoint} episodes={args.episodes}")
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:
        print(f"[evaluate] BLOCKED: Isaac Lab not available ({exc}).")
        return 2
    raise NotImplementedError("run rollouts, then call check_graduation(success_rates)")


if __name__ == "__main__":
    sys.exit(main())
