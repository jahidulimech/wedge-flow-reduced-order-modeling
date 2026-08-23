"""Data-driven screening of two-dimensional shock geometry.

This module deliberately does not register or shift 2D snapshots. Registration is
only enabled after a documented shock-curve definition and region of interest exist.
"""
from __future__ import annotations
import numpy as np

def neighbour_graph(n_cells: int, owner: np.ndarray, neighbour: np.ndarray) -> list[np.ndarray]:
    graph = [[] for _ in range(n_cells)]
    for left, right in zip(owner[:neighbour.size], neighbour):
        graph[int(left)].append(int(right))
        graph[int(right)].append(int(left))
    return [np.asarray(values, dtype=int) for values in graph]

def least_squares_gradient(xy: np.ndarray, scalar: np.ndarray, graph: list[np.ndarray]) -> np.ndarray:
    """Cell-neighbour least-squares scalar gradient; fails on inadequate stencils."""
    gradient = np.full((xy.shape[0], 2), np.nan)
    for i, neighbours in enumerate(graph):
        if neighbours.size < 2:
            continue
        offsets = xy[neighbours] - xy[i]
        delta = scalar[neighbours] - scalar[i]
        if np.linalg.matrix_rank(offsets) == 2:
            gradient[i] = np.linalg.lstsq(offsets, delta, rcond=None)[0]
    return gradient

def shock_indicator(xy: np.ndarray, pressure: np.ndarray, graph: list[np.ndarray], roi: dict | None = None) -> dict:
    """Return |grad p| and candidate support; no scalar shock location is assumed."""
    gradient = least_squares_gradient(xy, pressure, graph)
    indicator = np.linalg.norm(gradient, axis=1)
    support = np.isfinite(indicator)
    if roi is not None:
        required = {"x_min", "x_max", "y_min", "y_max"}
        if set(roi) != required:
            raise ValueError("2D shock ROI must have x_min, x_max, y_min, y_max")
        support &= ((xy[:, 0] >= roi["x_min"]) & (xy[:, 0] <= roi["x_max"]) &
                    (xy[:, 1] >= roi["y_min"]) & (xy[:, 1] <= roi["y_max"]))
    if not support.any():
        raise ValueError("No usable cells in shock screening support")
    threshold = np.nanpercentile(indicator[support], 99.0)
    candidates = support & (indicator >= threshold)
    return {"gradient_xy": gradient, "indicator": indicator, "candidate_mask": candidates,
            "threshold": float(threshold)}

def displacement_screen(indicators: np.ndarray, xy: np.ndarray) -> dict:
    """Weighted-centroid diagnostic only; not a physical shock-position definition."""
    if indicators.ndim != 2:
        raise ValueError("indicator snapshots must have shape (n_cells, n_times)")
    centres = []
    for values in indicators.T:
        weights = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        if weights.sum() == 0:
            centres.append([np.nan, np.nan])
        else:
            centres.append((weights[:, None] * xy).sum(axis=0) / weights.sum())
    centres = np.asarray(centres)
    return {"weighted_centroid_xy": centres, "centroid_displacement_m": np.linalg.norm(centres - centres[0], axis=1)}
