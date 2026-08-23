"""OpenFOAM ASCII reader and mesh mapper for the 2D ROM workflow.

The global state order is OpenFOAM's serial cell order (or cellProcAddressing order
for decomposed cases); it is never re-sorted geometrically before POD/DMD.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import re
import numpy as np

TIME_RE = re.compile(r"^[0-9.+-]+(?:[eE][+-]?[0-9]+)?$")


def numeric_time_dirs(root: Path) -> dict[float, str]:
    found: dict[float, str] = {}
    for item in root.iterdir():
        if item.is_dir() and TIME_RE.fullmatch(item.name):
            value = float(item.name)
            if value in found:
                raise ValueError(f"Duplicate numeric time {value}: {found[value]}, {item.name}")
            found[value] = item.name
    return dict(sorted(found.items()))


def _payload(text: str, source: Path) -> tuple[int, str]:
    match = re.search(r"\n\s*(\d+)\s*\n\s*\(", text)
    if not match:
        raise ValueError(f"{source}: expected an OpenFOAM counted list")
    count, start = int(match.group(1)), match.end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return count, text[start:i]
    raise ValueError(f"{source}: list closing parenthesis is missing")


def read_labels(path: Path) -> np.ndarray:
    count, data = _payload(path.read_text(encoding="utf-8", errors="replace"), path)
    values = np.fromstring(data, dtype=int, sep=" ")
    if values.size != count:
        raise ValueError(f"{path}: expected {count} labels, found {values.size}")
    return values


def read_points(path: Path) -> np.ndarray:
    count, data = _payload(path.read_text(encoding="utf-8", errors="replace"), path)
    values = np.array([np.fromstring(s, dtype=float, sep=" ") for s in re.findall(r"\(([^()]+)\)", data)])
    if values.shape != (count, 3):
        raise ValueError(f"{path}: expected {count} 3D points, found {values.shape}")
    return values


def read_faces(path: Path) -> list[np.ndarray]:
    count, data = _payload(path.read_text(encoding="utf-8", errors="replace"), path)
    faces = [np.fromstring(s, dtype=int, sep=" ") for s in re.findall(r"\d+\s*\(([^()]+)\)", data)]
    if len(faces) != count:
        raise ValueError(f"{path}: expected {count} faces, found {len(faces)}")
    return faces


def cell_centres_2d(poly_mesh: Path, n_cells: int | None = None) -> np.ndarray:
    """Return xy polygon centroids for extruded 2D cells from actual mesh topology."""
    points = read_points(poly_mesh / "points")
    faces = read_faces(poly_mesh / "faces")
    owner = read_labels(poly_mesh / "owner")
    neighbour = read_labels(poly_mesh / "neighbour")
    if owner.size != len(faces) or neighbour.size > owner.size:
        raise ValueError(f"{poly_mesh}: invalid owner/neighbour lists")
    count = int(owner.max()) + 1 if n_cells is None else n_cells
    members: list[list[int]] = [[] for _ in range(count)]
    for face_id, cell in enumerate(owner):
        members[int(cell)].append(face_id)
    for face_id, cell in enumerate(neighbour):
        members[int(cell)].append(face_id)
    centres = np.empty((count, 2), dtype=float)
    for cell, ids in enumerate(members):
        if not ids:
            raise ValueError(f"{poly_mesh}: cell {cell} has no faces")
        xy = points[np.unique(np.concatenate([faces[i] for i in ids]))][:, :2]
        xy = np.unique(xy, axis=0)
        if xy.shape[0] < 3:
            raise ValueError(f"{poly_mesh}: cell {cell} is not a 2D polygon")
        origin = xy.mean(axis=0)
        polygon = xy[np.argsort(np.arctan2(xy[:, 1] - origin[1], xy[:, 0] - origin[0]))]
        x, y = polygon[:, 0], polygon[:, 1]
        xp, yp = np.roll(x, -1), np.roll(y, -1)
        cross = x * yp - xp * y
        twice_area = cross.sum()
        centres[cell] = polygon.mean(axis=0) if abs(twice_area) < 1e-15 else np.array(
            [((x + xp) * cross).sum(), ((y + yp) * cross).sum()]) / (3.0 * twice_area)
    return centres


def read_internal_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"internalField\s+(.*?);", text, re.S)
    if not match:
        raise ValueError(f"{path}: no internalField")
    declaration = match.group(1).strip()
    if declaration.startswith("uniform"):
        token = declaration[len("uniform"):].strip()
        if token.startswith("("):
            return np.fromstring(token.strip("()"), dtype=float, sep=" ")[None, :]
        return np.array([float(token)])
    item = re.search(r"nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\(\s*(.*?)\s*\)\s*$", declaration, re.S)
    if not item:
        raise ValueError(f"{path}: unsupported internalField declaration")
    kind, count, body = item.group(1), int(item.group(2)), item.group(3)
    if kind == "scalar":
        values = np.fromstring(body, dtype=float, sep=" ")
        if values.size != count:
            raise ValueError(f"{path}: scalar count mismatch")
        return values
    values = np.array([np.fromstring(s, dtype=float, sep=" ") for s in re.findall(r"\(([^()]+)\)", body)])
    if values.shape != (count, 3):
        raise ValueError(f"{path}: vector count mismatch")
    return values


def _processors(case: Path) -> list[Path]:
    result = sorted((p for p in case.glob("processor*") if p.is_dir()), key=lambda p: int(p.name[9:]))
    numbers = [int(p.name[9:]) for p in result]
    if numbers != list(range(len(numbers))):
        raise ValueError(f"Non-contiguous processor identifiers: {numbers}")
    return result


def discover(case: Path, required_fields: Iterable[str]) -> dict:
    """Inspect serial or decomposed field availability and validate global mapping."""
    processors = _processors(case)
    roots = processors or [case]
    time_maps = [numeric_time_dirs(root) for root in roots]
    common = set(time_maps[0])
    for current in time_maps[1:]:
        common.intersection_update(current)
    fields_at_zero = sorted(p.name for p in (roots[0] / time_maps[0].get(0.0, "")).iterdir()
                            if p.is_file()) if 0.0 in time_maps[0] else []
    if processors:
        addresses, centre_blocks = [], []
        for root in processors:
            addressing = read_labels(root / "constant" / "polyMesh" / "cellProcAddressing")
            addresses.append(addressing)
            centre_blocks.append(cell_centres_2d(root / "constant" / "polyMesh", addressing.size))
        all_ids = np.concatenate(addresses)
        n_cells = int(all_ids.max()) + 1
        if all_ids.size != n_cells or set(all_ids.tolist()) != set(range(n_cells)):
            raise ValueError("cellProcAddressing does not cover global cells exactly once")
        xy = np.empty((n_cells, 2))
        for addressing, centres in zip(addresses, centre_blocks):
            xy[addressing] = centres
    else:
        xy = cell_centres_2d(case / "constant" / "polyMesh")
        n_cells = xy.shape[0]
        addresses = [np.arange(n_cells, dtype=int)]
    valid, missing = [], {}
    for time in sorted(common):
        absent = []
        for field in required_fields:
            for root, mapping in zip(roots, time_maps):
                if not (root / mapping[time] / field).is_file():
                    absent.append(f"{root.name}/{mapping[time]}/{field}")
        if absent:
            missing[time] = absent
        else:
            valid.append(time)
    return {"case": case, "processors": processors, "roots": roots, "time_maps": time_maps,
            "all_numeric_times": sorted(common), "valid_times": np.array(valid, dtype=float),
            "missing_fields": missing, "n_cells": n_cells, "xy_m": xy,
            "addresses": addresses, "initial_field_names": fields_at_zero}


def reconstruct(info: dict, time: float, field: str) -> np.ndarray:
    result = None
    for root, mapping, ids in zip(info["roots"], info["time_maps"], info["addresses"]):
        values = read_internal_field(root / mapping[float(time)] / field)
        if values.shape[0] == 1:
            values = np.repeat(values, ids.size, axis=0)
        if values.shape[0] != ids.size:
            raise ValueError(f"{root}/{mapping[float(time)]}/{field}: field/cell count mismatch")
        if result is None:
            result = np.empty((info["n_cells"],) + values.shape[1:], dtype=float)
        result[ids] = values
    if result is None or not np.isfinite(result).all():
        raise ValueError(f"{field} at {time}: reconstruction failed or contains non-finite values")
    return result
