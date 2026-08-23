"""Numerical POD/DMD operations for two-dimensional cell-centred OpenFOAM states."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class DMDModel:
    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    amplitudes: np.ndarray
    singular_values: np.ndarray
    rank: int
    dt_s: float

def require_uniform_times(time_s: np.ndarray, rtol: float = 1e-8) -> float:
    if time_s.ndim != 1 or time_s.size < 3:
        raise ValueError("At least three snapshots are required for DMD")
    spacing = np.diff(time_s)
    if not np.all(spacing > 0):
        raise ValueError("Snapshot times must be strictly increasing")
    if not np.allclose(spacing, spacing.mean(), rtol=rtol, atol=1e-14):
        raise ValueError("Nonuniform time spacing: resampling is deliberately not automatic")
    return float(spacing.mean())

def choose_rank(s: np.ndarray, energy_target: float) -> int:
    if not (0.0 < energy_target <= 1.0):
        raise ValueError("energy_target must lie in (0, 1]")
    energy = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(energy, energy_target) + 1)

def pod(field_by_time: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = field_by_time.mean(axis=1, keepdims=True)
    centered = field_by_time - mean
    modes, singular, _ = np.linalg.svd(centered, full_matrices=False)
    r = min(rank, modes.shape[1])
    reconstruction = mean + modes[:, :r] @ (modes[:, :r].T @ centered)
    return reconstruction, singular, mean

def exact_dmd(field_by_time: np.ndarray, dt_s: float, rank: int) -> DMDModel:
    if field_by_time.ndim != 2 or field_by_time.shape[1] < 3:
        raise ValueError("DMD requires an (n_dof, n_snapshots>=3) matrix")
    mean = field_by_time.mean(axis=1, keepdims=True)
    centered = field_by_time - mean
    x1, x2 = centered[:, :-1], centered[:, 1:]
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    numerical_rank = int(np.count_nonzero(s > s[0] * 1e-12))
    r = min(rank, numerical_rank)
    if r < 1:
        raise ValueError("Snapshot matrix has no numerically resolvable dynamics")
    ur, sr, vhr = u[:, :r], s[:r], vh[:r]
    a_tilde = ur.conj().T @ x2 @ vhr.conj().T @ np.diag(1.0 / sr)
    eigenvalues, vectors = np.linalg.eig(a_tilde)
    modes = x2 @ vhr.conj().T @ np.diag(1.0 / sr) @ vectors
    amplitudes = np.linalg.lstsq(modes, centered[:, 0], rcond=None)[0]
    return DMDModel(mean, modes, eigenvalues, amplitudes, s, r, dt_s)

def reconstruct_dmd(model: DMDModel, n_times: int) -> np.ndarray:
    k = np.arange(n_times)
    dynamics = model.amplitudes[:, None] * model.eigenvalues[:, None]**k
    return model.mean + np.real(model.modes @ dynamics)

def standardize_and_stack(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, dict]]:
    """Center and scale each field before combined multi-variable ROM."""
    pieces, metadata, start = [], {}, 0
    for name, values in fields.items():
        mean = values.mean(axis=1, keepdims=True)
        scale = float(np.std(values - mean))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name}: zero or non-finite fluctuation scale")
        pieces.append((values - mean) / scale)
        metadata[name] = {"slice": [start, start + values.shape[0]], "mean": mean, "scale": scale}
        start += values.shape[0]
    return np.vstack(pieces), metadata

def metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if not valid.any():
        return {name: float("nan") for name in ("relative_frobenius", "l2", "rmse", "nrmse", "max_abs")}
    ref, error = reference[valid], estimate[valid] - reference[valid]
    rmse = float(np.sqrt(np.mean(error**2)))
    span = float(ref.max() - ref.min())
    return {"relative_frobenius": float(np.linalg.norm(error) / np.linalg.norm(ref)),
            "l2": float(np.linalg.norm(error)), "rmse": rmse,
            "nrmse": float(rmse / span) if span > 0 else float("nan"),
            "max_abs": float(np.max(np.abs(error)))}

def eigenspectrum(model: DMDModel) -> dict[str, np.ndarray]:
    omega = np.log(model.eigenvalues) / model.dt_s
    return {"eigenvalue_real": model.eigenvalues.real, "eigenvalue_imag": model.eigenvalues.imag,
            "modulus": np.abs(model.eigenvalues), "growth_rate_per_s": omega.real,
            "frequency_hz": omega.imag / (2.0 * np.pi), "amplitude_modulus": np.abs(model.amplitudes)}
