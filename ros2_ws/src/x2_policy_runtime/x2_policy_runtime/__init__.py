"""x2_policy_runtime — ONNX policy execution under safety supervision (Phases 5-6).

GATED: while REAL_ROBOT_LOW_LEVEL_LEG_POLICY_APPROVED is false, nodes run dry-run only and
publish debug topics only — never a leg joint command. Pure logic lives in ``core``.
"""
