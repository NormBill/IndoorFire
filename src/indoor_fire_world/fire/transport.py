"""Obstacle-aware explicit finite-volume transport utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def stability_number(
    dt: float,
    cell_size_m: float,
    diffusion_m2_per_s: float,
    velocity_mps: tuple[float, float],
    decay_per_s: float,
) -> float:
    """Return a conservative monotonicity bound for 2D upwind transport."""
    ux, uy = velocity_mps
    return dt * (
        4.0 * diffusion_m2_per_s / (cell_size_m * cell_size_m)
        + (abs(ux) + abs(uy)) / cell_size_m
        + decay_per_s
    )


def validate_stable_timestep(
    dt: float,
    cell_size_m: float,
    diffusion_m2_per_s: float,
    velocity_mps: tuple[float, float],
    decay_per_s: float,
    field_name: str,
) -> None:
    values = (dt, cell_size_m, diffusion_m2_per_s, *velocity_mps, decay_per_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite numerical parameter for {field_name}")
    if dt <= 0.0 or cell_size_m <= 0.0:
        raise ValueError("dt and cell_size_m must be positive")
    if diffusion_m2_per_s < 0.0 or decay_per_s < 0.0:
        raise ValueError("diffusion and decay must be non-negative")
    number = stability_number(dt, cell_size_m, diffusion_m2_per_s, velocity_mps, decay_per_s)
    if number > 1.0 + 1e-12:
        raise ValueError(
            f"Unstable explicit timestep for {field_name}: stability number "
            f"{number:.6g} exceeds 1. Reduce dt or transport coefficients."
        )


def _no_flux_laplacian(field: FloatArray, fluid: BoolArray, spacing: float) -> FloatArray:
    """Five-point Laplacian with zero normal flux at solid and domain faces."""
    laplacian = np.zeros_like(field)
    horizontal_faces = fluid[:, :-1] & fluid[:, 1:]
    horizontal_delta = np.where(horizontal_faces, field[:, 1:] - field[:, :-1], 0.0)
    laplacian[:, :-1] += horizontal_delta
    laplacian[:, 1:] -= horizontal_delta
    vertical_faces = fluid[:-1, :] & fluid[1:, :]
    vertical_delta = np.where(vertical_faces, field[1:, :] - field[:-1, :], 0.0)
    laplacian[:-1, :] += vertical_delta
    laplacian[1:, :] -= vertical_delta
    return laplacian / (spacing * spacing)


def _upwind_advection(
    field: FloatArray,
    fluid: BoolArray,
    spacing: float,
    velocity_mps: tuple[float, float],
) -> FloatArray:
    """Return conservative upwind ``-div(u*q)`` with closed boundary faces."""
    ux, uy = velocity_mps
    tendency = np.zeros_like(field)

    open_x_faces = fluid[:, :-1] & fluid[:, 1:]
    upwind_x = field[:, :-1] if ux >= 0.0 else field[:, 1:]
    flux_x = np.where(open_x_faces, ux * upwind_x, 0.0)
    tendency[:, :-1] -= flux_x / spacing
    tendency[:, 1:] += flux_x / spacing

    open_y_faces = fluid[:-1, :] & fluid[1:, :]
    upwind_y = field[:-1, :] if uy >= 0.0 else field[1:, :]
    flux_y = np.where(open_y_faces, uy * upwind_y, 0.0)
    tendency[:-1, :] -= flux_y / spacing
    tendency[1:, :] += flux_y / spacing
    return tendency


def advance_scalar(
    field: NDArray[np.floating],
    fluid_mask: NDArray[np.bool_],
    source_rate: NDArray[np.floating],
    *,
    dt: float,
    cell_size_m: float,
    diffusion_m2_per_s: float,
    velocity_mps: tuple[float, float],
    decay_per_s: float,
    solid_value: float,
) -> FloatArray:
    """Advance one scalar using explicit diffusion, upwind advection, source, and decay."""
    current = np.asarray(field, dtype=np.float64)
    fluid = np.asarray(fluid_mask, dtype=np.bool_)
    source = np.asarray(source_rate, dtype=np.float64)
    if current.ndim != 2 or current.shape != fluid.shape or current.shape != source.shape:
        raise ValueError("field, fluid_mask, and source_rate must be matching 2D arrays")
    validate_stable_timestep(
        dt,
        cell_size_m,
        diffusion_m2_per_s,
        velocity_mps,
        decay_per_s,
        "scalar",
    )
    tendency = (
        diffusion_m2_per_s * _no_flux_laplacian(current, fluid, cell_size_m)
        + _upwind_advection(current, fluid, cell_size_m, velocity_mps)
        + source
        - decay_per_s * current
    )
    updated = np.maximum(current + dt * tendency, 0.0)
    updated[~fluid] = solid_value
    return updated
