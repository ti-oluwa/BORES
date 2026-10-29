"""
Saturation-function (relative permeability / capillary pressure) property
cache for black-oil simulation.
"""

import typing

import numpy as np
import numpy.typing as npt

from bores.blackoil.satfunc.regions import SatFunc
from bores.precision import get_dtype
from bores.reservoir.regions import RegionGroups, group_cells_by_region
from bores.types import BooleanCellArray, CellArray, IntCellArray

__all__ = ["SatFuncCache", "compute_satfunc_cache"]


class SatFuncCache(typing.NamedTuple):
    """
    Per-cell relative permeabilities, capillary pressures, and their
    saturation derivatives. Every field is shape `(n_cells,)`.

    Relative permeability derivatives are the full nine-way cross set, e.g.
    `dkro_dsw` (`dkro/dsw`) is
    generally nonzero in a three-phase mixing-rule model, not just the three
    "own-saturation" diagonal terms so it's kept in full rather than reduced to
    whatever two-saturation primary-variable convention a solver ends up
    using, since that reduction is a solver-side decision this cache
    shouldn't presume.

    Capillary pressure derivatives are only the four combinations the
    underlying tables actually define (`Pcow` depends on `Sw`/`So`, `Pcgo` on
    `So`/`Sg`, not a full cross set like relative permeability).

    Capillary-pressure fields are `NaN` for any region with no capillary
    pressure table (`SatFuncTables.capillary_pressure is None`).
    """

    # Relative permeabilities
    water_relative_permeability: CellArray
    """Water relative permeability, krw."""
    oil_relative_permeability: CellArray
    """Oil relative permeability, kro."""
    gas_relative_permeability: CellArray
    """Gas relative permeability, krg."""

    # Relative permeability derivatives w.r.t. water saturation
    dkrw_dsw: CellArray
    """Derivative of water relative permeability w.r.t. water saturation, dkrw/dsw."""
    dkro_dsw: CellArray
    """Derivative of oil relative permeability w.r.t. water saturation, dkro/dsw."""
    dkrg_dsw: CellArray
    """Derivative of gas relative permeability w.r.t. water saturation, dkrg/dsw."""

    # ... w.r.t. oil saturation
    dkrw_dso: CellArray
    """Derivative of water relative permeability w.r.t. oil saturation, dkrw/dso."""
    dkro_dso: CellArray
    """Derivative of oil relative permeability w.r.t. oil saturation, dkro/dso."""
    dkrg_dso: CellArray
    """Derivative of gas relative permeability w.r.t. oil saturation, dkrg/dso."""

    # ... w.r.t. gas saturation
    dkrw_dsg: CellArray
    """Derivative of water relative permeability w.r.t. gas saturation, dkrw/dsg."""
    dkro_dsg: CellArray
    """Derivative of oil relative permeability w.r.t. gas saturation, dkro/dsg."""
    dkrg_dsg: CellArray
    """Derivative of gas relative permeability w.r.t. gas saturation, dkrg/dsg."""

    # Capillary pressures: Pcow = Po - Pw, Pcgo = Pg - Po
    oil_water_capillary_pressure: CellArray
    """Oil-water capillary pressure, Pcow = Po - Pw."""
    gas_oil_capillary_pressure: CellArray
    """Gas-oil capillary pressure, Pcgo = Pg - Po."""

    # Capillary pressure derivatives
    dpcow_dsw: CellArray
    """Derivative of oil-water capillary pressure w.r.t. water saturation, dPcow/dsw."""
    dpcow_dso: CellArray
    """Derivative of oil-water capillary pressure w.r.t. oil saturation, dPcow/dso."""
    dpcgo_dso: CellArray
    """Derivative of gas-oil capillary pressure w.r.t. oil saturation, dPcgo/dso."""
    dpcgo_dsg: CellArray
    """Derivative of gas-oil capillary pressure w.r.t. gas saturation, dPcgo/dsg."""

    saturation_region: IntCellArray
    """The 1-based SATNUM each cell was evaluated against."""


CACHE_NAN_FIELDS: tuple[str, ...] = tuple(
    name for name in SatFuncCache._fields if name != "saturation_region"
)

CAPILLARY_PRESSURE_FIELDS: tuple[str, ...] = (
    "oil_water_capillary_pressure",
    "gas_oil_capillary_pressure",
    "dpcow_dsw",
    "dpcow_dso",
    "dpcgo_dso",
    "dpcgo_dsg",
)


def make_new_cache(n_cells: int, dtype: npt.DTypeLike) -> SatFuncCache:
    fields = {name: np.full(n_cells, np.nan, dtype=dtype) for name in CACHE_NAN_FIELDS}
    return SatFuncCache(
        **fields,
        saturation_region=np.zeros(n_cells, dtype=np.int32),
    )


def compute_satfunc_cache(
    water_saturation: CellArray,
    oil_saturation: CellArray,
    gas_saturation: CellArray,
    saturation_region: IntCellArray,
    satfunc: SatFunc,
    irreducible_water_saturation: CellArray | None = None,
    residual_oil_saturation_water: CellArray | None = None,
    residual_oil_saturation_gas: CellArray | None = None,
    residual_gas_saturation: CellArray | None = None,
    max_water_saturation: CellArray | None = None,
    max_gas_saturation: CellArray | None = None,
    water_imbibition_flag: BooleanCellArray | None = None,
    gas_imbibition_flag: BooleanCellArray | None = None,
    water_reversal_saturation: CellArray | None = None,
    gas_reversal_saturation: CellArray | None = None,
    region_groups: RegionGroups | None = None,
    out: SatFuncCache | None = None,
    dtype: npt.DTypeLike = None,
) -> SatFuncCache:
    """
    Build (or refresh, in place) a `SatFuncCache` from current
    cell saturations.

    Evaluates each region's `SatFuncTables` once per property, vectorised over that
    region's cells. The cells of each region come from `region_groups`, which is built
    once for the run, so a timestep does no region search and builds no boolean masks. The relative permeabilities and their derivatives are
    computed together, which analytical models do in a single pass.

    **Reuse**: pass a previous call's returned `SatFuncCache` as
    `out` to overwrite its arrays in place; the return value is `out` itself
    in that case. Every field is reset before being written on each call, so
    a region change between calls can't leave a stale value behind.

    :param water_saturation: Shape `(n_cells,)` current `Sw`.
    :param oil_saturation: Shape `(n_cells,)` current `So`.
    :param gas_saturation: Shape `(n_cells,)` current `Sg`.
    :param saturation_region: Shape `(n_cells,)` 1-based SATNUM per
        cell, e.g. `reservoir.regions.saturation_region` (falls back to
        all-ones there when the deck had no `SATNUM` keyword - pass that
        fallback through yourself if calling this directly without going
        through `Regions`).
    :param satfunc: `SatFunc` to
        evaluate against usually `BlackOil.satfunc`.
    :param irreducible_water_saturation: Shape `(n_cells,)` current `swc` -
        dynamic, per-cell hysteresis state from reservoir state, not
        anything this cache computes. `None` (default) omits the keyword
        entirely, so each table falls back to its own value. Forwarded
        as-is to every `evaluate`/`derivatives` call.
    :param residual_oil_saturation_water: Shape `(n_cells,)` current `sorw`.
        Same fallback behaviour as `irreducible_water_saturation`.
    :param residual_oil_saturation_gas: Shape `(n_cells,)` current `sorg`.
        Same fallback behaviour.
    :param residual_gas_saturation: Shape `(n_cells,)` current `sgr`. Same
        fallback behaviour.
    :param max_water_saturation: Shape `(n_cells,)` historical maximum
        water saturation reached in each cell, for Killough/Land scanning-curve
        hysteresis. Comes from `ReservoirState.hysteresis`, not
        anything this cache computes. Same fallback behaviour: omitted
        entirely if not given, so a hysteresis-aware table falls back to
        primary drainage curves.
    :param max_gas_saturation: Shape `(n_cells,)` historical maximum gas
        saturation reached in each cell. Same fallback behaviour.
    :param water_imbibition_flag: Shape `(n_cells,)` whether water
        saturation is currently increasing in each cell. Same fallback behaviour.
    :param gas_imbibition_flag: Shape `(n_cells,)` whether gas saturation
        is currently increasing in each cell. Same fallback behaviour.
    :param water_reversal_saturation: Shape `(n_cells,)` water saturation
        at each cell's most recent drainage/imbibition reversal. Same
        fallback behaviour.
    :param gas_reversal_saturation: Shape `(n_cells,)` gas saturation at
        each cell's most recent reversal. Same fallback behaviour.
    :param region_groups: Cells grouped by region, from `group_cells_by_region` or
        `Regions.group_saturation_cells`. Region numbers do not change during a run, so build it once and
        pass it on every call. Built here from `saturation_region` when not given, which
        repeats the grouping on every call.
    :param out: Previous `SatFuncCache` to overwrite in place, or
        `None` to allocate a new one.
    :return: The populated `SatFuncCache`. `out` itself if given,
        otherwise a newly allocated one. Always returned, never `None`.
    :raises ValueError: If `out` is given but sized for a different cell
        count than `water_saturation`.
    """
    n_cells = len(water_saturation)
    if out is not None:
        if len(out.water_relative_permeability) != n_cells:
            raise ValueError(
                f"`out` was allocated for {len(out.water_relative_permeability)} "
                f"cells; `water_saturation` has {n_cells}."
            )
        cache = out
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        cache = make_new_cache(n_cells, dtype=dtype)

    cache.saturation_region[:] = saturation_region
    if region_groups is None:
        region_groups = group_cells_by_region(saturation_region=saturation_region)

    # Every cell belongs to exactly one group, so each relative permeability field is fully
    # rewritten below. Only the capillary pressure fields can keep an old value, and only when
    # some region has no capillary pressure table, so only then do they need resetting.
    if any(
        satfunc[region].capillary_pressure is None for region in region_groups.saturation_regions
    ):
        for name in CAPILLARY_PRESSURE_FIELDS:
            getattr(cache, name).fill(np.nan)

    for group, region in enumerate(region_groups.saturation_regions):
        # `None` means the group holds every cell, so views of the inputs and whole-array
        # assignment replace the gather and scatter that several regions need.
        cells = region_groups.cell_indices[group]
        mask = slice(None) if cells is None else cells
        tables = satfunc[region]

        sw = water_saturation[mask]
        so = oil_saturation[mask]
        sg = gas_saturation[mask]

        # Dynamic per-cell saturation-state kwargs (hysteresis, if tracked).
        # Values come from the caller's reservoir state, not this cache.
        # Only included when the caller actually supplied that array, so a
        # table's own fallback applies otherwise.
        state_kwargs = {}
        if irreducible_water_saturation is not None:
            state_kwargs["irreducible_water_saturation"] = irreducible_water_saturation[mask]
        if residual_oil_saturation_water is not None:
            state_kwargs["residual_oil_saturation_water"] = residual_oil_saturation_water[mask]
        if residual_oil_saturation_gas is not None:
            state_kwargs["residual_oil_saturation_gas"] = residual_oil_saturation_gas[mask]
        if residual_gas_saturation is not None:
            state_kwargs["residual_gas_saturation"] = residual_gas_saturation[mask]
        if max_water_saturation is not None:
            state_kwargs["max_water_saturation"] = max_water_saturation[mask]
        if max_gas_saturation is not None:
            state_kwargs["max_gas_saturation"] = max_gas_saturation[mask]
        if water_imbibition_flag is not None:
            state_kwargs["water_imbibition_flag"] = water_imbibition_flag[mask]
        if gas_imbibition_flag is not None:
            state_kwargs["gas_imbibition_flag"] = gas_imbibition_flag[mask]
        if water_reversal_saturation is not None:
            state_kwargs["water_reversal_saturation"] = water_reversal_saturation[mask]
        if gas_reversal_saturation is not None:
            state_kwargs["gas_reversal_saturation"] = gas_reversal_saturation[mask]

        relperm_table = tables.relative_permeability
        relperm, relperm_derivatives = relperm_table.evaluate_with_derivatives(
            water_saturation=sw, oil_saturation=so, gas_saturation=sg, **state_kwargs
        )
        cache.water_relative_permeability[mask] = relperm["water"]
        cache.oil_relative_permeability[mask] = relperm["oil"]
        cache.gas_relative_permeability[mask] = relperm["gas"]

        cache.dkrw_dsw[mask] = relperm_derivatives["dkrw_dsw"]
        cache.dkro_dsw[mask] = relperm_derivatives["dkro_dsw"]
        cache.dkrg_dsw[mask] = relperm_derivatives["dkrg_dsw"]
        cache.dkrw_dso[mask] = relperm_derivatives["dkrw_dso"]
        cache.dkro_dso[mask] = relperm_derivatives["dkro_dso"]
        cache.dkrg_dso[mask] = relperm_derivatives["dkrg_dso"]
        cache.dkrw_dsg[mask] = relperm_derivatives["dkrw_dsg"]
        cache.dkro_dsg[mask] = relperm_derivatives["dkro_dsg"]
        cache.dkrg_dsg[mask] = relperm_derivatives["dkrg_dsg"]

        capillary_pressure_table = tables.capillary_pressure
        if capillary_pressure_table is not None:
            capillary_pressure, capillary_pressure_derivatives = (
                capillary_pressure_table.evaluate_with_derivatives(
                    water_saturation=sw, oil_saturation=so, gas_saturation=sg, **state_kwargs
                )
            )
            cache.oil_water_capillary_pressure[mask] = capillary_pressure["oil_water"]
            cache.gas_oil_capillary_pressure[mask] = capillary_pressure["gas_oil"]
            cache.dpcow_dsw[mask] = capillary_pressure_derivatives["dpcow_dsw"]
            cache.dpcow_dso[mask] = capillary_pressure_derivatives["dpcow_dso"]
            cache.dpcgo_dso[mask] = capillary_pressure_derivatives["dpcgo_dso"]
            cache.dpcgo_dsg[mask] = capillary_pressure_derivatives["dpcgo_dsg"]

    return cache
