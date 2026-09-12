import warnings

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import NDimension, NDimensionalGrid

__all__ = ["build_pressure_grid"]


@numba.njit(cache=True, inline="always")
def _gradient_from_density(density_lbm_ft3: float) -> float:
    """Convert fluid density (lbm/ft³) to pressure gradient (psi/ft)."""
    return density_lbm_ft3 / 144.0


@numba.njit(cache=True)
def _gradient_from_specific_gravity(specific_gravity: float) -> float:
    """
    Convert specific gravity (relative to fresh water at 62.4 lbm/ft³) to
    pressure gradient (psi/ft).
    """
    return specific_gravity * 62.4 / 144.0


@numba.njit(cache=True)
def _gradient_from_api(api_gravity: float) -> float:
    """
    Convert oil API gravity to pressure gradient (psi/ft).
    API gravity -> specific gravity: SG = 141.5 / (API + 131.5)
    """
    sg = 141.5 / (api_gravity + 131.5)
    return _gradient_from_specific_gravity(sg)


def build_pressure_grid(
    depth_grid: NDimensionalGrid[NDimension],
    datum_depth: float,
    datum_pressure: float,
    oil_gradient: float | None = None,
    water_gradient: float | None = None,
    gas_gradient: float | None = None,
    oil_density: float | None = None,
    water_density: float | None = None,
    gas_density: float | None = None,
    oil_specific_gravity: float | None = None,
    water_specific_gravity: float | None = None,
    gas_specific_gravity: float | None = None,
    oil_api_gravity: float | None = None,
    gas_oil_contact: float | None = None,
    oil_water_contact: float | None = None,
    cap_pressure: float | None = None,
    dtype: npt.DTypeLike = None,
) -> NDimensionalGrid[NDimension]:
    """
    Build a hydrostatically equilibrated pressure grid from a datum pressure
    and fluid density/gradient information.

    This replicates the industry-standard "equilibration" step used in
    commercial simulators (ECLIPSE EQUIL, CMG INITS) to initialise the
    pressure field so that it is in capillary-gravity equilibrium at t = 0,
    eliminating the large pressure residuals that arise from flat (uniform)
    initialisation.

    The grid is divided into up to three hydrostatic columns based on the
    supplied fluid contacts, each integrated with its own pressure gradient:

    ```markdown
    ┌──────────────────────────────────┐
    │  GAS CAP  (depth < GOC)          │  gradient = gas_gradient
    ├──────────────────────────────────┤ ← Gas-Oil Contact (GOC)
    │  OIL ZONE (GOC ≤ depth < OWC)    │  gradient = oil_gradient   ← datum here
    ├──────────────────────────────────┤ ← Oil-Water Contact (OWC)
    │  WATER ZONE (depth >= OWC)       │  gradient = water_gradient
    └──────────────────────────────────┘
    ```

    Pressure at any depth D is computed as:

        P(D) = P_datum + gradient x (D - D_datum)

    where gradient is the appropriate fluid gradient for that zone and depth
    is positive downward (increases with depth).

    **Gradient specification (priority order per phase):**

    For each fluid phase, the gradient is resolved in this order:
    1. Direct gradient (psi/ft) — `oil_gradient`, `water_gradient`, `gas_gradient`
    2. Fluid density (lbm/ft³) — `oil_density`, `water_density`, `gas_density`
    3. Specific gravity (dimensionless) — `oil_specific_gravity` etc.
    4. API gravity (oil only) — `oil_api_gravity`

    Standard defaults are used for any phase whose gradient cannot be
    resolved from the supplied inputs (see below), with a warning issued.

    **Defaults when gradient is not supplied:**

    ```markdown
    +----------+----------------------------+-------------------+
    | Phase    | Default gradient (psi/ft)  | Equivalent SG     |
    +----------+----------------------------+-------------------+
    | Oil      | 0.3600                     | SG ≈ 0.831        |
    | Water    | 0.4335                     | SG = 1.000        |
    | Gas      | 0.0800                     | SG ≈ 0.185        |
    +----------+----------------------------+-------------------+
    ```

    **Single-phase / two-phase reservoirs:**

    - Oil-only (no gas cap, no free water): omit `gas_oil_contact` and `oil_water_contact`.
        The entire grid uses `oil_gradient`.
    - Oil + water aquifer: supply only `oil_water_contact`.
    - Gas + oil (no aquifer): supply only `gas_oil_contact`.
    - All three zones: supply both contacts.
    - Pure gas reservoir: supply `gas_gradient` (or equivalent) only; omit contacts.
        The datum pressure is applied to the gas column.

    :param depth_grid: 3-D grid of cell-centre depths (ft), positive downward.
        Typically built with `build_depth_grid`. Shape: (nx, ny, nz).
    :param datum_depth: Reference depth (ft) at which `datum_pressure` is
        known. Usually the centre of the oil zone, the reservoir mid-point,
        or the gas-oil contact. Must be within or close to the reservoir.
    :param datum_pressure: Pressure (psia) at `datum_depth`. Measured or estimated from well tests, DSTs, or gradient extrapolation.
    :param oil_gradient: Oil pressure gradient (psi/ft). Takes priority over density / SG / API inputs for the oil phase.
    :param water_gradient: Water pressure gradient (psi/ft).
    :param gas_gradient: Gas pressure gradient (psi/ft).
    :param oil_density: Dead-oil (or live-oil at reservoir conditions) density (lbm/ft³). Converted internally to psi/ft.
    :param water_density: Water density (lbm/ft³).
    :param gas_density: Gas density (lbm/ft³).
    :param oil_specific_gravity: Oil specific gravity relative to fresh water (dimensionless, 1.0 = 62.4 lbm/ft³).
    :param water_specific_gravity: Water specific gravity.
    :param gas_specific_gravity: Gas specific gravity.
    :param oil_api_gravity: Oil API gravity (°API). Converted to SG and then to gradient.
        Ignored if `oil_gradient`, `oil_density`, or `oil_specific_gravity` is provided.
    :param gas_oil_contact: Gas-oil contact depth (ft). Above this depth, the gas gradient is used.
        If None, no gas cap is assumed.
    :param oil_water_contact: Oil-water contact depth (ft). Below this depth, the water gradient is used.
        If None, no free water zone is assumed.
    :param cap_pressure: Optional fixed pressure (psia) to assign to cells
        shallower than the reservoir top (e.g. a sealing caprock layer with
        known overburden pressure). When provided, all cells with
        `depth < min(depth_grid)` are set to this value regardless of gradient.
        Rarely needed; most users should omit this.

    :return: Pressure grid (psia), same shape as `depth_grid`.

    :raises ValidationError: If contacts are in the wrong order (GOC >= OWC),
        datum depth is outside the depth grid range (with a warning, not an
        error, to handle datum at a reference surface), or if the grid is not
        3-D.
    :warns UserWarning: If the datum depth is outside the depth grid range,
        if default gradients are applied, or if contacts are outside the grid.

    Examples:

    **SPE1-style initialisation (oil + gas cap, no aquifer):**

    ```python
    pressure_grid = build_pressure_grid(
        depth_grid=depth_grid,
        datum_depth=8400.0,  # centre of Layer 3 (ft)
        datum_pressure=4800.0,  # psia
        oil_density=37.474,  # lbm/ft³ at initial conditions
        gas_oil_contact=8200.0,  # ft
        gas_specific_gravity=0.792,
    )
    ```

    **Simple oil reservoir, gradient from API:**

    ```python
    pressure_grid = build_pressure_grid(
        depth_grid=depth_grid,
        datum_depth=5000.0,
        datum_pressure=3000.0,
        oil_api_gravity=35.0,
        oil_water_contact=5300.0,
        water_specific_gravity=1.05,
    )
    ```

    **Three-phase equilibration matching spe1.py exactly:**

    ```python
    pressure_grid = build_pressure_grid(
        depth_grid=depth_grid,
        datum_depth=8400.0,
        datum_pressure=4800.0,
        oil_density=37.474,
        gas_specific_gravity=0.792,
        water_specific_gravity=1.0,
        gas_oil_contact=8200.0,
        oil_water_contact=8500.0,
    )
    ```

    References:
    - Craft, B.C., Hawkins, M.F. (1959). *Applied Petroleum Reservoir Engineering*.
    - Dake, L.P. (1978). *Fundamentals of Reservoir Engineering*. Elsevier.
    - Schlumberger (2010). *ECLIPSE Reservoir Simulation Reference Manual* — EQUIL keyword.
    """
    _validate_pressure_inputs(
        depth_grid=depth_grid,
        datum_depth=datum_depth,
        datum_pressure=datum_pressure,
        gas_oil_contact=gas_oil_contact,
        oil_water_contact=oil_water_contact,
    )
    dtype = dtype if dtype is not None else get_dtype()

    # Resolve gradients for each phase
    resolved_oil_gradient = _resolve_gradient(
        phase="oil",
        gradient=oil_gradient,
        density=oil_density,
        specific_gravity=oil_specific_gravity,
        api_gravity=oil_api_gravity,
        default=0.3600,
    )
    resolved_water_gradient = _resolve_gradient(
        phase="water",
        gradient=water_gradient,
        density=water_density,
        specific_gravity=water_specific_gravity,
        api_gravity=None,
        default=0.4335,
    )
    resolved_gas_gradient = _resolve_gradient(
        phase="gas",
        gradient=gas_gradient,
        density=gas_density,
        specific_gravity=gas_specific_gravity,
        api_gravity=None,
        default=0.0800,
    )

    # Compute contact pressures by integrating from datum. These become the inter-zone boundary conditions.
    # Pressure at GOC (propagated from datum using appropriate gradient)
    pressure_at_goc: float | None = None
    if gas_oil_contact is not None:
        # If datum is in oil zone (below GOC), integrate upward to GOC with oil gradient.
        # If datum is in gas zone (above GOC), integrate downward to GOC with gas gradient.
        if datum_depth >= gas_oil_contact:
            # datum is at or below GOC: oil gradient from datum up to GOC
            pressure_at_goc = datum_pressure + resolved_oil_gradient * (
                gas_oil_contact - datum_depth
            )
        else:
            # datum is above GOC (in gas cap): gas gradient from datum down to GOC
            pressure_at_goc = datum_pressure + resolved_gas_gradient * (
                gas_oil_contact - datum_depth
            )

    # Pressure at OWC
    pressure_at_owc: float | None = None
    if oil_water_contact is not None:
        if datum_depth <= oil_water_contact:
            # datum is at or above OWC: oil gradient from datum down to OWC
            pressure_at_owc = datum_pressure + resolved_oil_gradient * (
                oil_water_contact - datum_depth
            )
        else:
            # datum is below OWC (in water zone): water gradient from datum up to OWC
            pressure_at_owc = datum_pressure + resolved_water_gradient * (
                oil_water_contact - datum_depth
            )

    # Assign pressure per cell based on zone membership
    pressure_grid = np.empty_like(depth_grid, dtype=dtype)

    # Determine zone masks
    # Gas cap: above GOC
    if gas_oil_contact is not None:
        gas_mask = depth_grid < gas_oil_contact
    else:
        gas_mask = np.zeros_like(depth_grid, dtype=bool)

    # Water zone: at or below OWC
    if oil_water_contact is not None:
        water_mask = depth_grid >= oil_water_contact
    else:
        water_mask = np.zeros_like(depth_grid, dtype=bool)

    # Oil zone: everything else
    oil_mask = ~gas_mask & ~water_mask

    # Oil zone: P = P_datum + oil_gradient * (D - D_datum)
    # But if contacts exist, we anchor to the contact pressure to avoid
    # gradient discontinuities when datum is not in the oil zone.
    if np.any(oil_mask):
        if gas_oil_contact is not None and datum_depth < gas_oil_contact:
            # datum in gas zone; anchor oil zone to pressure_at_goc
            assert pressure_at_goc is not None
            pressure_grid[oil_mask] = pressure_at_goc + resolved_oil_gradient * (
                depth_grid[oil_mask] - gas_oil_contact
            )
        elif oil_water_contact is not None and datum_depth > oil_water_contact:
            # datum in water zone; anchor oil zone to pressure_at_owc
            assert pressure_at_owc is not None
            pressure_grid[oil_mask] = pressure_at_owc + resolved_oil_gradient * (
                depth_grid[oil_mask] - oil_water_contact
            )
        else:
            # datum is in oil zone (the typical case)
            pressure_grid[oil_mask] = datum_pressure + resolved_oil_gradient * (
                depth_grid[oil_mask] - datum_depth
            )

    # Gas cap: P = P_at_goc + gas_gradient * (D - GOC)
    if np.any(gas_mask):
        assert pressure_at_goc is not None
        pressure_grid[gas_mask] = pressure_at_goc + resolved_gas_gradient * (
            depth_grid[gas_mask] - gas_oil_contact  # type: ignore[operator]
        )

    # Water zone: P = P_at_owc + water_gradient * (D - OWC)
    if np.any(water_mask):
        assert pressure_at_owc is not None
        pressure_grid[water_mask] = pressure_at_owc + resolved_water_gradient * (
            depth_grid[water_mask] - oil_water_contact  # type: ignore[operator]
        )

    # Optional caprock override
    if cap_pressure is not None:
        cap_mask = depth_grid < float(np.min(depth_grid)) + 1e-6
        pressure_grid[cap_mask] = cap_pressure

    return pressure_grid  # type: ignore[return-value]


def _resolve_gradient(
    phase: str,
    gradient: float | None,
    density: float | None,
    specific_gravity: float | None,
    api_gravity: float | None,
    default: float,
) -> float:
    """
    Resolve the pressure gradient (psi/ft) for a fluid phase from the
    supplied inputs in priority order: gradient > density > SG > API > default.
    """
    if gradient is not None:
        return float(gradient)
    if density is not None:
        return _gradient_from_density(float(density))
    if specific_gravity is not None:
        return _gradient_from_specific_gravity(float(specific_gravity))
    if api_gravity is not None:
        return _gradient_from_api(float(api_gravity))

    warnings.warn(
        f"No {phase} gradient / density / SG supplied. "
        f"Using default gradient of {default:.4f} psi/ft.",
        UserWarning,
        stacklevel=4,
    )
    return default


def _validate_pressure_inputs(
    depth_grid: npt.NDArray,
    datum_depth: float,
    datum_pressure: float,
    gas_oil_contact: float | None,
    oil_water_contact: float | None,
) -> None:
    """Validate inputs for build_pressure_grid."""
    if depth_grid.ndim != 3:
        raise ValidationError(
            f"`depth_grid` must be 3-D (nx, ny, nz), got shape {depth_grid.shape}."
        )
    if datum_pressure <= 0:
        raise ValidationError(f"`datum_pressure` must be positive, got {datum_pressure} psia.")
    if datum_depth < 0:
        raise ValidationError(
            f"`datum_depth` must be non-negative (depth is positive downward), "
            f"got {datum_depth} ft."
        )
    if (
        gas_oil_contact is not None
        and oil_water_contact is not None
        and gas_oil_contact >= oil_water_contact
    ):
        raise ValidationError(
            f"`gas_oil_contact` ({gas_oil_contact} ft) must be shallower than "
            f"`oil_water_contact` ({oil_water_contact} ft). "
            "Depths are positive downward."
        )

    grid_min = float(np.min(depth_grid))
    grid_max = float(np.max(depth_grid))
    if not (grid_min <= datum_depth <= grid_max):
        warnings.warn(
            f"`datum_depth` ({datum_depth} ft) is outside the depth grid range "
            f"[{grid_min:.1f}, {grid_max:.1f}] ft. "
            "Pressure will be extrapolated from the datum.",
            UserWarning,
            stacklevel=4,
        )
