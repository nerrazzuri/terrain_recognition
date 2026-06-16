"""Teacher-student distillation (P6-M3-T2). BLOCKED to run (torch).

Trains the raw-depth CReF student to imitate the Phase 4 height-map teacher: action imitation
+ value imitation + auxiliary terrain prediction (roadmap §10.5 Option A). Then fine-tune with
PPO (P6-M3-T3, scripts/train_cref.py). The loss *weights* live here as the single source.
"""
from __future__ import annotations

LOSS_WEIGHTS = {
    "action_imitation": 1.0,
    "value_imitation": 0.5,
    "terrain_aux": 0.2,
    "height_recon_aux": 0.1,
}


def distillation_loss(student_out, teacher_out, weights: dict | None = None):
    """Weighted distillation loss. BLOCKED to run (torch).

    student_out / teacher_out are dicts with action/value/aux tensors. Implemented once torch
    is available:
        L = w_a * MSE(student.action, teacher.action)
          + w_v * MSE(student.value,  teacher.value)
          + w_t * CE(student.terrain_logits, terrain_label)
          + w_h * MSE(student.height_recon, height_samples)
    """
    import torch
    import torch.nn.functional as F

    w = {**LOSS_WEIGHTS, **(weights or {})}
    loss = w["action_imitation"] * F.mse_loss(student_out["action"], teacher_out["action"])
    loss = loss + w["value_imitation"] * F.mse_loss(
        student_out["value"], teacher_out["value"])
    if "terrain_type_logits" in student_out and "terrain_label" in teacher_out:
        loss = loss + w["terrain_aux"] * F.cross_entropy(
            student_out["terrain_type_logits"], teacher_out["terrain_label"])
    if "height_reconstruction" in student_out and "height_samples" in teacher_out:
        loss = loss + w["height_recon_aux"] * F.mse_loss(
            student_out["height_reconstruction"], teacher_out["height_samples"])
    return loss
