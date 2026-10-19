"""
Per-Rs undersaturated branches for live oil (`PVTO` / `PVCO`) and reciprocal (`1/B`) interpolation.

Keeps one undersaturated branch per Rs node and, for oil with an Rs between two nodes, blend the two
bounding branches linearly in Rs at the *same pressure*. Putting every branch on a common pressure
grid turns that into a plain trilinear lookup over `(pressure, temperature, Rs)`, which `PVTTable`
already does for the wet-gas tables.
"""

import typing

import numpy as np
import numpy.typing as npt

__all__ = [
    "build_live_oil_branch_tables",
    "evaluate_interpolator",
    "invert_interpolators",
    "linear_extrapolate",
]


def linear_extrapolate(
    x: npt.NDArray[np.floating],
    xp: npt.NDArray[np.floating],
    fp: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """
    Piecewise-linear interpolation of `(xp, fp)` at *x*, extrapolating linearly beyond both ends.

    Unlike `np.interp`, which holds the end values constant, the end segments' slopes are extended.
    That is what Eclipse does outside a branch's tabulated pressures.

    :param x: Query points.
    :param xp: Strictly increasing knots, at least two.
    :param fp: Values at the knots.
    :returns: Interpolated values, float64.
    """
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    result = np.interp(x, xp, fp)
    if len(xp) < 2:
        return result

    below = x < xp[0]
    if np.any(below):
        result[below] = fp[0] + (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x[below] - xp[0])

    above = x > xp[-1]
    if np.any(above):
        result[above] = fp[-1] + (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x[above] - xp[-1])
    return result


def build_live_oil_branch_tables(
    rs_nodes: npt.NDArray[np.floating],
    group_pressures: typing.Sequence[npt.NDArray[np.floating]],
    group_fvf: typing.Sequence[npt.NDArray[np.floating]],
    group_viscosity: typing.Sequence[npt.NDArray[np.floating]],
    pressures: npt.NDArray[np.floating],
    *,
    floor_fraction: float = 0.05,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Evaluate every Rs group's branch on a common pressure grid and build the saturated envelope.

    Each group is one Rs node: rows of `(P, Bo, μo)` whose first row is the saturated (bubble-point)
    state and whose later rows are the undersaturated branch at that Rs.

    - **Branch tables** `(n_p, n_rs)`: group `k`'s `Bo` / `μo` at every grid pressure, linearly
      extrapolated beyond its rows (below its bubble point as well, which Eclipse also does when it
      blends a branch at a pressure under that branch's own Pb). Extrapolated values are floored at
      `floor_fraction` of the group's smallest tabulated value so a far extrapolation cannot go
      non-positive.
    - **Saturated envelope** `(n_p,)`: `Bo`, `μo` and `Rs` along the bubble curve, interpolated
      linearly across the `(Pb, saturated value)` nodes for `P <= Pb_max`. Above the highest bubble
      point there are no saturated states, so the envelope continues on the highest-Rs group's branch
      (Rs held at its maximum).

    :param rs_nodes: Strictly increasing Rs value of each group.
    :param group_pressures: Per-group pressures, each strictly increasing; the first is `Pb`.
    :param group_fvf: Per-group `Bo` at those pressures.
    :param group_viscosity: Per-group `μo` at those pressures.
    :param pressures: Common pressure grid, strictly increasing.
    :returns: `(envelope_fvf, envelope_viscosity, envelope_rs, branch_fvf, branch_viscosity)`.
    """
    rs_nodes = np.asarray(rs_nodes, dtype=np.float64)
    pressures = np.asarray(pressures, dtype=np.float64)
    n_p, n_rs = len(pressures), len(rs_nodes)

    branch_fvf = np.empty((n_p, n_rs))
    branch_viscosity = np.empty((n_p, n_rs))
    for k in range(n_rs):
        oil_fvf = np.asarray(group_fvf[k], dtype=np.float64)
        viscosity = np.asarray(group_viscosity[k], dtype=np.float64)
        branch_fvf[:, k] = np.maximum(
            linear_extrapolate(pressures, group_pressures[k], oil_fvf),
            floor_fraction * oil_fvf.min(),
        )
        branch_viscosity[:, k] = np.maximum(
            linear_extrapolate(pressures, group_pressures[k], viscosity),
            floor_fraction * viscosity.min(),
        )

    bubble_points = np.array([
        np.asarray(pressure, dtype=np.float64)[0] for pressure in group_pressures
    ])
    saturated_fvf = np.array([np.asarray(fvf, dtype=np.float64)[0] for fvf in group_fvf])
    saturated_viscosity = np.array([
        np.asarray(viscosity, dtype=np.float64)[0] for viscosity in group_viscosity
    ])

    on_envelope = pressures <= bubble_points[-1]
    envelope_fvf = np.where(
        on_envelope, np.interp(pressures, bubble_points, saturated_fvf), branch_fvf[:, -1]
    )
    envelope_viscosity = np.where(
        on_envelope,
        np.interp(pressures, bubble_points, saturated_viscosity),
        branch_viscosity[:, -1],
    )
    envelope_rs = np.interp(pressures, bubble_points, rs_nodes)
    return envelope_fvf, envelope_viscosity, envelope_rs, branch_fvf, branch_viscosity


def evaluate_interpolator(interpolator: typing.Any, *args: typing.Any) -> typing.Any:
    """Call `interpolator.ev(*args)` (a `RectBivariateSpline`) or `interpolator(*args)` otherwise."""
    return interpolator.ev(*args) if hasattr(interpolator, "ev") else interpolator(*args)


def invert_interpolators(
    inverse_value: typing.Any, inverse_derivative: typing.Any
) -> tuple[typing.Callable[..., typing.Any], typing.Callable[..., typing.Any]]:
    """
    Build interpolators of `B` and `∂B/∂P` from interpolators of `1/B` and `∂(1/B)/∂P`.

    A formation volume factor is smoother in `1/B` than in `B` (ideal-gas `Bg ∝ 1/P`, so `1/Bg` is
    linear in `P`), which is why Eclipse tabulates and interpolates `1/B`. With `g = 1/B`:
    `B = 1/g` and `∂B/∂P = -g'/g²`.

    :param inverse_value: Interpolator of `1/B` (`.ev` or callable).
    :param inverse_derivative: Interpolator of `∂(1/B)/∂P`, same arguments.
    :returns: `(value, derivative)` callables taking the same arguments as the inputs.
    """

    def value(*args: typing.Any) -> typing.Any:
        return 1.0 / evaluate_interpolator(inverse_value, *args)

    def derivative(*args: typing.Any) -> typing.Any:
        inverse = evaluate_interpolator(inverse_value, *args)
        # Some derivative interpolators flatten their inputs; give the slope the value's shape
        slope = np.reshape(evaluate_interpolator(inverse_derivative, *args), np.shape(inverse))
        return -slope / (inverse * inverse)

    return value, derivative
