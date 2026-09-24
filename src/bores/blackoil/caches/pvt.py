"""
PVT (pressure-volume-temperature) property cache for black-oil simulation.

`PVTTable.formation_volume_factor`/`.viscosity`/`.db_dp`/`.dμ_dp`/
`.density`/`.dρ_dp` all correctly switch between the saturated and
undersaturated correlations for oil, given `solution_gor`/
`bubble_point_pressure` (which `compute_pvt_cache` passes to all six).

`.density`/`.dρ_dp`'s undersaturated branch only activates when the table
was built with stock-tank densities available (`pvt=` at `PVTTable`
construction). Falls back to the raw (potentially wrong above the bubble
point) saturated-curve table otherwise, since there's no way to recompute
density live without them.
"""

import typing

import numpy as np
import numpy.typing as npt

from bores.blackoil.pvt.regions import PVT
from bores.precision import get_dtype
from bores.types import BooleanCellArray, CellArray, IntCellArray

__all__ = ["PVTCache", "compute_pvt_cache"]


class PVTCache(typing.NamedTuple):
    """
    Per-cell PVT properties and their pressure derivatives.

    Every field is shape `(n_cells,)`. Absent quantities (a phase not
    present in the model, or a table not provided for it, e.g. no `Rv`/wet-gas
    table) are `numpy.nan`, never a silently-plausible zero: a stray `NaN`
    reaching arithmetic downstream is loud and fails fast; a stray zero
    viscosity or formation volume factor is not.
    """

    # Formation volume factors
    oil_formation_volume_factor: CellArray
    """Oil formation volume factor, $B_o$."""
    dbo_dp: CellArray
    """Pressure derivative of oil formation volume factor, $dB_o/dP$."""
    water_formation_volume_factor: CellArray
    """Water formation volume factor, $B_w$."""
    dbw_dp: CellArray
    """Pressure derivative of water formation volume factor, $dB_w/dP$."""
    gas_formation_volume_factor: CellArray
    """Gas formation volume factor, $B_g$."""
    dbg_dp: CellArray
    """Pressure derivative of gas formation volume factor, $dB_g/dP$."""

    # Viscosities
    oil_viscosity: CellArray
    r"""Oil viscosity, $\mu_o$."""
    dμo_dp: CellArray
    r"""Pressure derivative of oil viscosity, $d\mu_o/dP$."""
    water_viscosity: CellArray
    r"""Water viscosity, $\mu_w$."""
    dμw_dp: CellArray
    r"""Pressure derivative of water viscosity, $d\mu_w/dP$."""
    gas_viscosity: CellArray
    r"""Gas viscosity, $\mu_g$."""
    dμg_dp: CellArray
    r"""Pressure derivative of gas viscosity, $d\mu_g/dP$."""

    # Densities
    oil_density: CellArray
    r"""Oil density, $\rho_o$."""
    dρo_dp: CellArray
    r"""Pressure derivative of oil density, $d\rho_o/dP$."""
    water_density: CellArray
    r"""Water density, $\rho_w$."""
    dρw_dp: CellArray
    r"""Pressure derivative of water density, $d\rho_w/dP$."""
    gas_density: CellArray
    r"""Gas density, $\rho_g$."""
    dρg_dp: CellArray
    r"""Pressure derivative of gas density, $d\rho_g/dP$."""

    # Compressibilities (value only - c = -(1/B)(dB/dP) for oil/water,
    # 1/P - (1/z)(dz/dP) for gas; already what dB/dP-based Jacobian entries
    # would derive from, kept here mainly for output/diagnostics and timestep
    # heuristics rather than as a required Jacobian input)
    oil_compressibility: CellArray
    """Oil compressibility, $c_o$."""
    water_compressibility: CellArray
    """Water compressibility, $c_w$."""
    gas_compressibility: CellArray
    """Gas compressibility, $c_g$."""

    # Dissolved/vaporized composition
    solution_gas_oil_ratio: CellArray
    """
    `Rs`. For saturated cells (`is_saturated[i] == True`), this is the
    thermodynamically consistent value read off the bubble curve at the
    cell's current pressure - `PVTTable.solution_gas_to_oil_ratio` - not
    whatever raw estimate was passed in. For undersaturated cells, `Rs` is a
    genuine primary variable and this is that value, passed through as-is.
    """
    drs_dp: CellArray
    """
    Pressure derivative of solution gas-oil ratio, $dR_s/dP$. `dRs/dP` along
    the bubble curve. Only meaningful for saturated cells - `Rs` isn't a
    function of `P` alone in the undersaturated regime (it's the independent
    primary variable there), so this is `NaN` where `is_saturated[i] ==
    False`.
    """
    vaporized_oil_gas_ratio: CellArray
    """
    Vaporized oil-gas ratio, $R_v$. `Rv`, evaluated along the dew curve
    (`PVTTable.vaporized_oil_gas_ratio`). Unlike `Rs`, the underlying table has no
    separate undersaturated-wet-gas correction exposed for `Bg`/`mu_g` - this
    is always the dew-curve value, not a primary-variable override. `NaN` for
    dry-gas models with no `Rv` table.
    """
    drv_dp: CellArray
    """Pressure derivative of vaporized oil-gas ratio, $dR_v/dP$."""
    water_gas_solubility_ratio: CellArray
    """`Rsw` - gas solubility in water. `NaN` unless the deck models it."""
    drsw_dp: CellArray
    """Pressure derivative of water-gas solubility ratio, $dR_{sw}/dP$."""

    # Bubble/dew point
    bubble_point_pressure: CellArray
    """
    Bubble-point pressure, $P_b$. `Pb`, evaluated at each cell's current `Rs`
    (`solution_gor=`, not the reported `solution_gas_oil_ratio` output - see
    `compute_pvt_cache`). For a saturated cell this equals the cell's own
    pressure; for an undersaturated one it's below it.
    """
    dpb_drs: CellArray
    """
    Derivative of bubble-point pressure w.r.t. solution gas-oil ratio,
    $dP_b/dR_s$. `dPb/dRs`. Only defined for a 2-D (`Rs`-dependent)
    bubble-point table - `NaN` for a 1-D `Pb(T)` table or non-oil cells.
    """
    dew_point_pressure: CellArray
    """Dew-point pressure, $P_{dew}(T)$. `NaN` for dry-gas models."""

    # Gas compressibility factor
    gas_compressibility_factor: CellArray
    """Gas compressibility factor, $z$."""
    dz_dp: CellArray
    """Pressure derivative of gas compressibility factor, $dz/dP$."""

    # Regime / bookkeeping
    is_saturated: BooleanCellArray
    """
    Per-cell oil regime: `True` = saturated (free gas present, `Sg` is the
    natural primary variable, `Rs` is bubble-curve-determined), `False` =
    undersaturated (`Rs` is itself a primary variable). This is the
    primary-variable-switching indicator - computed from
    `PVTTable.is_saturated`, not supplied by the caller.
    """
    pvt_region: IntCellArray
    """The 1-based PVTNUM each cell was evaluated against."""


CACHE_NAN_FIELDS: tuple[str, ...] = tuple(
    name for name in PVTCache._fields if name not in ("is_saturated", "pvt_region")
)


def make_new_cache(n_cells: int, dtype: npt.DTypeLike) -> PVTCache:
    fields = {name: np.full(n_cells, np.nan, dtype=dtype) for name in CACHE_NAN_FIELDS}
    return PVTCache(
        **fields,
        is_saturated=np.zeros(n_cells, dtype=np.bool_),
        pvt_region=np.zeros(n_cells, dtype=np.int32),
    )


def compute_pvt_cache(
    pressure: CellArray,
    temperature: CellArray,
    solution_gas_oil_ratio: CellArray,
    pvt_region: IntCellArray,
    pvt: PVT,
    salinity: CellArray | None = None,
    out: PVTCache | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTCache:
    """
    Build (or refresh, in place) a `PVTCache` from the current cell state.

    Groups cells by `pvt_region` (typically a handful of regions, not
    `n_cells` of them) and evaluates each region's `PVTTable`s once per
    property, vectorised over that region's cells - never a per-cell Python
    loop. This function itself is not, and can't be, `@njit`-compiled: the
    underlying tables interpolate via `scipy`, which numba can't call.
    Derived, arithmetic-only quantities (mobility and its derivatives) belong
    in `caches.mobility`'s jitted kernel instead, consuming this cache's
    output as plain arrays.

    **Reuse**: pass the `PVTCache` returned by a previous call as `out` to
    overwrite its arrays in place rather than allocate new ones - the
    returned object is `out` itself in that case. Omit `out` (or pass `None`)
    only when no cache exists yet; every field is reset (`NaN`/`False`/`0`
    for absent quantities) before being written, so a cell that had a
    `gas_formation_volume_factor` last call but has none this call (e.g. a
    region reassignment) won't carry a stale value forward.

    :param pressure: Shape `(n_cells,)` current cell pressure.
    :param temperature: Shape `(n_cells,)` current cell temperature.
    :param solution_gas_oil_ratio: Shape `(n_cells,)` current `Rs` estimate -
        authoritative for undersaturated cells (where it's a primary
        variable), advisory for saturated ones (where `PVTCache.
        solution_gas_oil_ratio` reports the bubble-curve value instead - see
        that field's docstring). Pass whatever the solver currently tracks
        for every cell regardless of regime; regime is determined here, not
        by the caller.
    :param pvt_region: Shape `(n_cells,)` 1-based PVTNUM per cell, e.g.
        `reservoir.regions.pvt_region` (falls back to all-ones there when
        the deck had no `PVTNUM` keyword - pass that fallback through
        yourself if calling this directly without going through `Regions`).
    :param pvt: `PVT` to evaluate against - `bores.blackoil.
        fluid.BlackOil.pvt`.
    :param salinity: Optional shape `(n_cells,)` water salinity (ppm NaCl).
        `None` uses each table's own default salinity throughout.
    :param out: Previous `PVTCache` to overwrite in place, or `None` to
        allocate a new one.
    :return: The populated `PVTCache` - `out` itself if given, otherwise a
        newly allocated one. Always returned, never `None`.
    :raises ValueError: If `out` is given but sized for a different cell
        count than `pressure`.
    """
    n_cells = len(pressure)
    if out is not None:
        if len(out.oil_formation_volume_factor) != n_cells:
            raise ValueError(
                f"`out` was allocated for "
                f"{len(out.oil_formation_volume_factor)} cells; "
                f"`pressure` has {n_cells}."
            )
        cache = out
        for name in CACHE_NAN_FIELDS:
            getattr(cache, name).fill(np.nan)
        cache.is_saturated.fill(False)
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        cache = make_new_cache(n_cells, dtype=dtype)

    cache.pvt_region[:] = pvt_region
    for region in np.unique(pvt_region):
        mask = pvt_region == region
        tables = pvt[int(region)].tables

        p = pressure[mask]
        t = temperature[mask]
        rs_estimate = solution_gas_oil_ratio[mask]
        s = salinity[mask] if salinity is not None else None

        oil = tables.oil
        if oil is not None:
            is_saturated = oil.is_saturated(p, t, solution_gor=rs_estimate)
            if is_saturated is not None:
                cache.is_saturated[mask] = np.atleast_1d(is_saturated)

            bubble_point = oil.bubble_point_pressure(temperature=t, solution_gor=rs_estimate)

            cache.oil_formation_volume_factor[mask] = oil.formation_volume_factor(  # type: ignore[arg-type]
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )
            cache.dbo_dp[mask] = oil.db_dp(  # type: ignore[arg-type]
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )
            cache.oil_viscosity[mask] = oil.viscosity(  # type: ignore[arg-type]
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )
            cache.dμo_dp[mask] = oil.dμ_dp(  # type: ignore[arg-type]
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )
            cache.oil_density[mask] = oil.density(  # type: ignore[arg-type]
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )
            cache.dρo_dp[mask] = oil.dρ_dp(  # type: ignore[arg-type]
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )

            compressibility = oil.compressibility(p, t)
            if compressibility is not None:
                cache.oil_compressibility[mask] = compressibility

            solution_gor = oil.solution_gas_to_oil_ratio(
                p, t, solution_gor=rs_estimate, bubble_point_pressure=bubble_point
            )
            if solution_gor is not None:
                cache.solution_gas_oil_ratio[mask] = solution_gor
            solution_gor_dp = oil.drs_dp(p, t)
            if solution_gor_dp is not None:
                cache.drs_dp[mask] = solution_gor_dp

            if bubble_point is not None:
                cache.bubble_point_pressure[mask] = bubble_point
            bubble_point_drs = oil.dpb_drs(rs_estimate, t)
            if bubble_point_drs is not None:
                cache.dpb_drs[mask] = bubble_point_drs

        water = tables.water
        if water is not None:
            cache.water_formation_volume_factor[mask] = water.formation_volume_factor(  # type: ignore[arg-type]
                p, t, salinity=s
            )
            cache.dbw_dp[mask] = water.db_dp(p, t, salinity=s)  # type: ignore[arg-type]
            cache.water_viscosity[mask] = water.viscosity(p, t, salinity=s)  # type: ignore[arg-type]
            cache.dμw_dp[mask] = water.dμ_dp(p, t, salinity=s)  # type: ignore[arg-type]
            cache.water_density[mask] = water.density(p, t, salinity=s)  # type: ignore[arg-type]
            cache.dρw_dp[mask] = water.dρ_dp(p, t, salinity=s)  # type: ignore[arg-type]
            compressibility = water.compressibility(p, t, salinity=s)
            if compressibility is not None:
                cache.water_compressibility[mask] = compressibility

        gas = tables.gas
        if gas is not None:
            cache.gas_formation_volume_factor[mask] = gas.formation_volume_factor(p, t)  # type: ignore[arg-type]
            cache.dbg_dp[mask] = gas.db_dp(p, t)  # type: ignore[arg-type]
            cache.gas_viscosity[mask] = gas.viscosity(p, t)  # type: ignore[arg-type]
            cache.dμg_dp[mask] = gas.dμ_dp(p, t)  # type: ignore[arg-type]
            cache.gas_density[mask] = gas.density(p, t)  # type: ignore[arg-type]
            cache.dρg_dp[mask] = gas.dρ_dp(p, t)  # type: ignore[arg-type]

            compressibility = gas.compressibility(p, t)
            if compressibility is not None:
                cache.gas_compressibility[mask] = compressibility

            z = gas.compressibility_factor(p, t)
            if z is not None:
                cache.gas_compressibility_factor[mask] = z
            z_dp = gas.dz_dp(p, t)
            if z_dp is not None:
                cache.dz_dp[mask] = z_dp

            vaporized_oil_gas_ratio = gas.vaporized_oil_gas_ratio(p, t)
            if vaporized_oil_gas_ratio is not None:
                cache.vaporized_oil_gas_ratio[mask] = vaporized_oil_gas_ratio
            drv_dp = gas.drv_dp(p, t)
            if drv_dp is not None:
                cache.drv_dp[mask] = drv_dp

            dew_point = gas.dew_point_pressure(t)
            if dew_point is not None:
                cache.dew_point_pressure[mask] = dew_point

            solubility = gas.solubility_in_water(p, t, salinity=s)
            if solubility is not None:
                cache.water_gas_solubility_ratio[mask] = solubility
            solubility_dp = gas.drsw_dp(p, t, salinity=s)
            if solubility_dp is not None:
                cache.drsw_dp[mask] = solubility_dp

    return cache
