#!/usr/bin/env python3
"""analyze_policy_log.py (P5-M3-T7).

Summarise a JSONL policy deployment log: cycle count, how many cycles would have commanded,
and a breakdown of stop reasons. Read-only.

Usage: python tools/analyze_policy_log.py logs/policy/run.jsonl
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_policy_runtime"))
sys.path.insert(0, str(_REPO / "ros2_ws/src/x2_common"))

from x2_policy_runtime.core.log_analysis import load_jsonl, summarize  # noqa: E402


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: analyze_policy_log.py <log.jsonl>")
        return 2
    summary = summarize(load_jsonl(args[0]))
    print(f"cycles:         {summary['cycles']}")
    print(f"would_command:  {summary['would_command']}")
    print("stop_reasons:")
    for reason, count in sorted(summary["stop_reasons"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>6}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
