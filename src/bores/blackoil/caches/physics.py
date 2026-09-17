"""
Composed physics property cache for black-oil simulation.

Bundles `PVTCache`, `SatFuncCache`, and `MobilityCache` behind a single
call: the three property groups a per-Newton-iteration cell update always
needs together, computed in the correct dependency order (`MobilityCache`
consumes both of the other two) without the caller wiring that dependency
itself each time.
"""

import typing

import numpy.typing as npt

from bores.blackoil.caches.mobility import MobilityCache, compute_mobility_cache
from bores.blackoil.caches.pvt import PVTCache, compute_pvt_cache
from bores.blackoil.caches.satfunc import SatFuncCache, compute_satfunc_cache
from bores.blackoil.fluids.model import BlackOil
from bores.reservoir.state.base import Hysteresis
from bores.reservoir.workspace import HysteresisWorkspace
from bores.types import CellArray, IntCellArray

__all__ = ["PhysicsCache", "compute_physics_cache"]


class PhysicsCache(typing.NamedTuple):
    """
    Every per-cell physics property a black-oil cell update needs, for one
    point in the simulation: PVT properties, saturation-function properties,
    and the phase mobilities/fractional-flows derived from both.

    A thin composition, not a flattened merge - `pvt`/`satfunc`/`mobility`
    stay as their own `PVTCache`/`SatFuncCache`/`MobilityCache` so each
    keeps its own field names, docstrings, and NaN-for-absent-data
    convention rather than colliding in one combined namespace.
    """

    pvt: PVTCache
    """This cell state's PVT properties - `caches.pvt.compute_pvt_cache`'s output."""

    satfunc: SatFuncCache
    """This cell state's saturation-function properties -
    `caches.satfunc.compute_satfunc_cache`'s output."""

    mobility: MobilityCache
    """Phase mobilities/fractional-flows derived from `pvt` and `satfunc` -
    `caches.mobility.compute_mobility_cache`'s output."""


def compute_physics_cache(
    pressure: CellArray,
    temperature: CellArray,
    solution_gas_oil_ratio: CellArray,
    water_saturation: CellArray,
    oil_saturation: CellArray,
    gas_saturation: CellArray,
    pvt_region: IntCellArray,
    saturation_region: IntCellArray,
    fluid: BlackOil,
    salinity: CellArray | None = None,
    irreducible_water_saturation: CellArray | None = None,
    residual_oil_saturation_water: CellArray | None = None,
    residual_oil_saturation_gas: CellArray | None = None,
    residual_gas_saturation: CellArray | None = None,
    hysteresis: Hysteresis | HysteresisWorkspace | None = None,
    out: PhysicsCache | None = None,
    dtype: npt.DTypeLike = None,
) -> PhysicsCache:
    """
    Build (or refresh, in place) a `PhysicsCache` from the current cell
    state: `compute_pvt_cache` and `compute_satfunc_cache`, then
    `compute_mobility_cache` from their two outputs - one call in place of
    wiring all three yourself.

    **Reuse**: pass a previous call's returned `PhysicsCache` as `out` to
    overwrite its three sub-caches' arrays in place - each sub-cache is
    forwarded to its own compute function as that function's own `out`, so
    no sub-cache is ever reallocated across calls. The return value is
    `out` itself in that case, same convention as every other cache in this
    package.

    :param pressure: Shape `(n_cells,)` current cell pressure.
    :param temperature: Shape `(n_cells,)` current cell temperature.
    :param solution_gas_oil_ratio: Shape `(n_cells,)` current `Rs`
        estimate. Forwarded to `compute_pvt_cache` - see that function's
        docstring for the saturated/undersaturated distinction.
    :param water_saturation: Shape `(n_cells,)` current `Sw`.
    :param oil_saturation: Shape `(n_cells,)` current `So`.
    :param gas_saturation: Shape `(n_cells,)` current `Sg`.
    :param pvt_region: Shape `(n_cells,)` 1-based PVTNUM per cell.
    :param saturation_region: Shape `(n_cells,)` 1-based SATNUM per cell.
    :param fluid: `BlackOil` supplying both `pvt` and `satfunc` region tables.
    :param salinity: Optional shape `(n_cells,)` water salinity (ppm NaCl).
        Forwarded to `compute_pvt_cache`.
    :param irreducible_water_saturation: Optional shape `(n_cells,)` current
        `swc`. Forwarded to `compute_satfunc_cache`.
    :param residual_oil_saturation_water: Optional shape `(n_cells,)`
        current `sorw`. Forwarded to `compute_satfunc_cache`.
    :param residual_oil_saturation_gas: Optional shape `(n_cells,)` current
        `sorg`. Forwarded to `compute_satfunc_cache`.
    :param residual_gas_saturation: Optional shape `(n_cells,)` current
        `sgr`. Forwarded to `compute_satfunc_cache`.
    :param hysteresis: Optional current `ReservoirState.hysteresis`. If
        given, its six tracked fields are unpacked and forwarded to
        `compute_satfunc_cache`, enabling Killough/Land scanning-curve
        hysteresis for any saturation-function table that supports it.
        Tables that don't fall back to their normal drainage curves, same
        as when this isn't given at all.
    :param out: Previous `PhysicsCache` to overwrite in place, or `None` to
        allocate a new one.
    :param dtype: Output array dtype for any newly allocated sub-cache.
        `bores.precision.get_dtype()` if not given. Ignored for a sub-cache
        supplied via `out` - that sub-cache keeps its existing dtype.
    :return: The populated `PhysicsCache`. `out` itself if given, otherwise
        a newly allocated one. Always returned, never `None`.
    :raises ValueError: If `pressure`/`water_saturation` disagree on cell
        count, or a sub-cache inside `out` is sized for a different cell
        count - raised by that sub-cache's own compute function.
    """
    n_cells = len(pressure)
    if len(water_saturation) != n_cells:
        raise ValueError(
            f"`pressure` has {n_cells} cells; `water_saturation` has {len(water_saturation)}."
        )

    pvt_cache = compute_pvt_cache(
        pressure=pressure,
        temperature=temperature,
        solution_gas_oil_ratio=solution_gas_oil_ratio,
        pvt_region=pvt_region,
        pvt=fluid.pvt,
        salinity=salinity,
        out=out.pvt if out is not None else None,
        dtype=dtype,
    )
    satfunc_cache = compute_satfunc_cache(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        saturation_region=saturation_region,
        satfunc=fluid.satfunc,
        irreducible_water_saturation=irreducible_water_saturation,
        residual_oil_saturation_water=residual_oil_saturation_water,
        residual_oil_saturation_gas=residual_oil_saturation_gas,
        residual_gas_saturation=residual_gas_saturation,
        max_water_saturation=hysteresis.max_water_saturation if hysteresis is not None else None,
        max_gas_saturation=hysteresis.max_gas_saturation if hysteresis is not None else None,
        water_imbibition_flag=(
            hysteresis.water_imbibition_flag if hysteresis is not None else None
        ),
        gas_imbibition_flag=hysteresis.gas_imbibition_flag if hysteresis is not None else None,
        water_reversal_saturation=(
            hysteresis.water_reversal_saturation if hysteresis is not None else None
        ),
        gas_reversal_saturation=(
            hysteresis.gas_reversal_saturation if hysteresis is not None else None
        ),
        out=out.satfunc if out is not None else None,
        dtype=dtype,
    )
    mobility_cache = compute_mobility_cache(
        pvt_cache=pvt_cache,
        satfunc_cache=satfunc_cache,
        out=out.mobility if out is not None else None,
        dtype=dtype,
    )

    if out is not None:
        return out
    return PhysicsCache(pvt=pvt_cache, satfunc=satfunc_cache, mobility=mobility_cache)
