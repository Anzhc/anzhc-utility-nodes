from __future__ import annotations

import torch


def resolve_alpha(alpha: float, task_count: int) -> float:
    """Return the effective SVC alpha, using 1/K when alpha is zero."""
    if task_count <= 0:
        raise ValueError("SVC merge requires at least one input")
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("SVC alpha must be between 0 and 1")
    if alpha == 0.0:
        return 1.0 / task_count
    return alpha


def _to_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32)


def _as_matrix(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    shape = tensor.shape
    return tensor.reshape(shape[0], -1), shape


def _svd_with_retry(matrix: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        noise = torch.zeros_like(matrix)
        diag_len = min(matrix.shape)
        if diag_len > 0:
            idx = torch.arange(diag_len, device=matrix.device)
            noise[idx, idx] = eps
        return torch.linalg.svd(matrix + noise, full_matrices=False)


def calibrate_matrix_delta(
    task_deltas: list[torch.Tensor],
    merged_delta: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not task_deltas:
        raise ValueError("SVC calibration requires at least one task delta")
    if merged_delta.numel() == 0:
        return merged_delta.clone()

    merged_matrix, original_shape = _as_matrix(_to_float32(merged_delta))
    task_matrices = torch.stack([_as_matrix(_to_float32(delta))[0] for delta in task_deltas], dim=0)

    u, singular_values, vh = _svd_with_retry(merged_matrix, eps)
    basis = u.T.contiguous()

    merged_response = basis @ merged_matrix
    task_response = torch.einsum("rm,kmn->rkn", basis, task_matrices)

    denom = (task_response * task_response).sum(dim=-1).clamp_min(eps)
    scale_coeffs = torch.einsum("rd,rkd->rk", merged_response, task_response) / denom
    alpha_tensor = torch.tensor(alpha, dtype=scale_coeffs.dtype, device=scale_coeffs.device)
    eta = torch.maximum(scale_coeffs, alpha_tensor).mean(dim=1).clamp_min(eps)

    calibrated_singular_values = singular_values / eta.to(singular_values.dtype)
    calibrated = (u * calibrated_singular_values.unsqueeze(0)) @ vh
    return calibrated.reshape(original_shape).to(device=merged_delta.device, dtype=merged_delta.dtype)


def calibrate_vector_delta(
    task_deltas: list[torch.Tensor],
    merged_delta: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not task_deltas:
        raise ValueError("SVC calibration requires at least one task delta")
    if merged_delta.numel() == 0:
        return merged_delta.clone()

    merged = _to_float32(merged_delta)
    tasks = torch.stack([_to_float32(delta) for delta in task_deltas], dim=0)
    denom = (tasks * tasks).sum(dim=-1).clamp_min(eps)
    scale_coeffs = (tasks * merged).sum(dim=-1) / denom
    alpha_tensor = torch.tensor(alpha, dtype=scale_coeffs.dtype, device=scale_coeffs.device)
    eta = torch.maximum(scale_coeffs, alpha_tensor).mean().clamp_min(eps)
    return (merged / eta).to(device=merged_delta.device, dtype=merged_delta.dtype)


def merge_svc_delta_tensors(
    task_deltas: list[torch.Tensor],
    alpha: float = 0.0,
) -> torch.Tensor:
    alpha_eff = resolve_alpha(alpha, len(task_deltas))
    ref = task_deltas[0]
    if ref.ndim == 0 or not torch.is_floating_point(ref):
        return ref.clone()

    normalized_deltas = []
    for delta in task_deltas:
        if not isinstance(delta, torch.Tensor):
            raise ValueError("SVC delta merge requires tensor inputs")
        if delta.shape != ref.shape:
            raise ValueError(
                f"SVC delta shape mismatch: expected {tuple(ref.shape)} vs input {tuple(delta.shape)}"
            )
        normalized_deltas.append(delta.to(device=ref.device, dtype=ref.dtype))

    merged_delta = torch.stack(normalized_deltas, dim=0).mean(dim=0)
    if ref.ndim == 1:
        return calibrate_vector_delta(normalized_deltas, merged_delta, alpha_eff)
    return calibrate_matrix_delta(normalized_deltas, merged_delta, alpha_eff)


def merge_svc_delta_state_dicts(
    delta_state_dicts: list[dict[str, torch.Tensor]],
    alpha: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Merge already-weighted LoRA delta tensors with SVC.

    Missing keys are treated as zero deltas so partially overlapping LoRAs can
    still be merged.
    """
    alpha_eff = resolve_alpha(alpha, len(delta_state_dicts))
    keys: set[str] = set()
    for state_dict in delta_state_dicts:
        keys.update(state_dict.keys())

    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in keys:
            ref = next((state_dict.get(key) for state_dict in delta_state_dicts if isinstance(state_dict.get(key), torch.Tensor)), None)
            if ref is None:
                continue

            if ref.ndim == 0 or not torch.is_floating_point(ref):
                result[key] = ref.clone()
                continue

            task_deltas = []
            for state_dict in delta_state_dicts:
                value = state_dict.get(key)
                if value is None:
                    task_deltas.append(torch.zeros_like(ref))
                    continue
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"SVC delta merge requires tensor value for key {key!r}")
                if value.shape != ref.shape:
                    raise ValueError(
                        f"SVC delta shape mismatch for key {key!r}: "
                        f"expected {tuple(ref.shape)} vs input {tuple(value.shape)}"
                    )
                task_deltas.append(value.to(device=ref.device, dtype=ref.dtype))

            result[key] = merge_svc_delta_tensors(task_deltas, alpha_eff)

    return result
