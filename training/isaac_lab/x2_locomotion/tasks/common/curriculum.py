"""Curriculum manager (P4-M2-T3). Pure logic, no Isaac Lab.

Advances through the stages A-G (standing -> flat_walk -> rough -> single_step -> stairs_up
-> stairs_down -> mixed) when the success rate clears a threshold. Does not jump straight to
stairs (roadmap §8.8).
"""
from __future__ import annotations


class Curriculum:
    def __init__(self, stages: list[str], advance_threshold: float = 0.8):
        if not stages:
            raise ValueError("stages must be non-empty")
        self.stages = list(stages)
        self.advance_threshold = float(advance_threshold)
        self._idx = 0

    @property
    def current_stage(self) -> str:
        return self.stages[self._idx]

    @property
    def is_complete(self) -> bool:
        return self._idx >= len(self.stages) - 1

    def update(self, success_rate: float) -> bool:
        """If the success rate clears the threshold, advance one stage. Returns whether it did."""
        if success_rate >= self.advance_threshold and not self.is_complete:
            self._idx += 1
            return True
        return False
