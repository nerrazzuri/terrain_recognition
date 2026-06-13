"""onnx_policy_runner (P5-M1-T2).

Loads an ONNX policy and runs fixed-rate inference, validating output dims and timing. Bad or
late output triggers a safe stop (the caller cuts the command). Output validation +
timing-budget checks are the unit-tested ``onnx_runner_core`` helpers.

BLOCKED to run: requires onnxruntime + an exported policy.onnx (Phase 4). The class is usable
once both exist; loading is guarded so a missing runtime gives a clear error.
"""
from __future__ import annotations

import time

import numpy as np

from .core.onnx_runner_core import validate_output, within_period


class OnnxPolicyRunner:
    def __init__(self, model_path: str, action_dim: int = 12, policy_period_s: float = 0.02):
        self.model_path = model_path
        self.action_dim = action_dim
        self.policy_period_s = policy_period_s
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                f"onnxruntime not available — cannot load {model_path}: {exc}")
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def infer(self, observation: np.ndarray):
        """Return ``(action, ok, info)``. ``ok`` False ⇒ caller must safe-stop."""
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        t0 = time.perf_counter()
        out = self._sess.run(None, {self._input_name: obs})[0].reshape(-1)
        dt = time.perf_counter() - t0
        ok_dim, reason = validate_output(out, self.action_dim)
        ok_time = within_period(dt, self.policy_period_s)
        ok = ok_dim and ok_time
        if not ok_time:
            reason = f"inference {dt * 1e3:.1f} ms over budget {self.policy_period_s * 1e3:.1f} ms"
        return out, ok, {"inference_s": dt, "reason": reason}
