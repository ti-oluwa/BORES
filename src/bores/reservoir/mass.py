"""Phase masses derived from saturations, pore volume, and PVT properties."""

import typing

import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import CellArray, IntCellArray, UnitSystem

if typing.TYPE_CHECKING:
    from bores.blackoil.pvt.regions import PVT

__all__ = ["Masses", "compute_masses"]


class Masses(typing.NamedTuple):
    """Per-cell phase masses, one row per cell."""

    oil_mass: CellArray
    """Shape `(n_cells,)`. Stock-tank oil mass in place, including dissolved gas."""

    water_mass: CellArray
    """Shape `(n_cells,)`. Stock-tank water mass in place."""

    free_gas_mass: CellArray
    """Shape `(n_cells,)`. Free (non-dissolved) gas mass in place."""

    dissolved_gas_mass_in_oil: CellArray
    """Shape `(n_cells,)`. Gas mass dissolved in the oil phase, `Rs`-derived."""

    dissolved_gas_mass_in_water: CellArray
    """
    Shape `(n_cells,)`. Gas mass dissolved in the water phase. Always
    zero - gas solubility in water (`Rsw`) isn't accounted for in mass
    terms anywhere in this codebase yet, matching `initialize_reservoir_state`.
    """

    vaporized_oil_mass_in_gas: CellArray
    """Shape `(n_cells,)`. Oil mass vaporized into the gas phase, `Rv`-derived."""


def compute_masses(
    *,
    pressure: CellArray,
    temperature: CellArray,
    oil_saturation: CellArray,
    water_saturation: CellArray,
    gas_saturation: CellArray,
    solution_gor: CellArray,
    vaporized_oil_gas_ratio: CellArray,
    pore_volumes: CellArray,
    pvt: "PVT",
    pvt_region_index: IntCellArray | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    dtype: npt.DTypeLike = None,
) -> Masses:
    """
    Compute per-cell phase masses from saturations, pore volume, and PVT
    properties.

    Masses are not primary unknowns anywhere in this codebase - they're
    derived from pressure, saturations, `Rs`/`Rv`, and each `PVTNUM`
    region's own PVT tables, the same way `initialize_reservoir_state`
    derives them for the initial state. This function is that same
    derivation, shared so it isn't duplicated.

    :param pressure: Oil-phase reference pressure, shape `(n_cells,)`.
    :param temperature: Reservoir temperature, shape `(n_cells,)`.
    :param oil_saturation: Shape `(n_cells,)`.
    :param water_saturation: Shape `(n_cells,)`.
    :param gas_saturation: Shape `(n_cells,)`.
    :param solution_gor: `Rs`, shape `(n_cells,)`.
    :param vaporized_oil_gas_ratio: `Rv`, shape `(n_cells,)`.
    :param pore_volumes: Pore volume at `pressure` (i.e. already scaled by
        `rock.compressibility_table.pore_volume_multiplier(pressure)` where
        rock compressibility applies), shape `(n_cells,)`.
    :param pvt: PVT tables to evaluate `Bo`/`Bg`/`Bw` and stock-tank
        densities from, by `PVTNUM` region.
    :param pvt_region_index: `PVTNUM` per cell. All region `1` if not given.
    :param unit_system: Needed to apply the `FIELD`-only ft3<->bbl
        correction between `Rs`/`Rv` (SCF/STB) and oil/water volumes (STB).
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `Masses` for every cell.
    :raises ValidationError: If a `PVTNUM` region needed by at least one
        cell has no oil PVT table, no `stock_tank_oil_density`, or is
        missing a gas/water PVT table for a cell with nonzero gas/water
        saturation.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n_cells = len(pressure)
    if pvt_region_index is None:
        pvt_region_index = typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))

    oil_mass = np.zeros(n_cells, dtype=dtype)
    water_mass = np.zeros(n_cells, dtype=dtype)
    free_gas_mass = np.zeros(n_cells, dtype=dtype)
    dissolved_gas_mass_in_oil = np.zeros(n_cells, dtype=dtype)
    vaporized_oil_mass_in_gas = np.zeros(n_cells, dtype=dtype)

    # `UnitSystem.FIELD` mixes two volume "families": oil/water are barrels (STB), gas is
    # cubic feet (SCF), and 1 barrel = 5.614583 ft3 (`c.BARRELS_TO_CUBIC_FEET`).
    # `solution_gor` (Rs) and `vaporized_oil_gas_ratio` (Rv) are both normalized
    # to SCF/STB by this codebase, so each crosses those two families the same
    # way: converting `Rs * rho_g_sc` into an oil-mass-basis term (or
    # `Rv * rho_o_sc` into a gas-mass-basis term) needs an explicit ft3<->bbl
    # correction; skipping it overstates the mass by ~5.615x. `METRIC`/`SI`/
    # `LAB` use a single volume unit throughout (m3/m3, cc/cc) so no such
    # correction applies for them.
    volume_correction = c.CUBIC_FEET_TO_STB if unit_system is UnitSystem.FIELD else 1.0

    for pvtnum in np.unique(pvt_region_index):
        mask = pvt_region_index == pvtnum
        pvt_region = pvt.region(pvtnum)
        static = pvt_region.static
        if static.stock_tank_oil_density is None:
            raise ValidationError(
                f"`PVTNUM` {pvtnum}: `stock_tank_oil_density` (`DENSITY` "
                "keyword) is required to compute masses."
            )

        rho_o_sc = static.stock_tank_oil_density
        rho_g_sc = static.stock_tank_gas_density
        rho_w_sc = static.stock_tank_water_density

        p = pressure[mask]
        t = temperature[mask]
        so = oil_saturation[mask]
        sw = water_saturation[mask]
        sg = gas_saturation[mask]
        rs = solution_gor[mask]
        rv = vaporized_oil_gas_ratio[mask]
        pv = pore_volumes[mask]

        if pvt_region.tables.oil is None:
            raise ValidationError(f"`PVTNUM` {pvtnum}: oil PVT table is unavailable.")
        bo = pvt_region.tables.oil.formation_volume_factor(
            pressure=p, temperature=t, solution_gor=rs
        )
        if bo is None:
            raise ValidationError(
                f"`PVTNUM` {pvtnum}: oil formation volume factor table is unavailable."
            )
        oil_mass[mask] = so * pv / bo * rho_o_sc

        if np.any(sg > 0.0):
            if pvt_region.tables.gas is None:
                raise ValidationError(
                    f"`PVTNUM` {pvtnum}: free gas saturation is present "
                    "but no gas PVT table is available."
                )

            bg = pvt_region.tables.gas.formation_volume_factor(pressure=p, temperature=t)
            if bg is None or rho_g_sc is None:
                raise ValidationError(
                    f"`PVTNUM` {pvtnum}: gas FVF table or "
                    "`stock_tank_gas_density` is unavailable but Sg > 0."
                )
            free_gas_mass[mask] = sg * pv / bg * rho_g_sc
            vaporized_oil_mass_in_gas[mask] = (
                rv * free_gas_mass[mask] * (rho_o_sc / rho_g_sc) * volume_correction
            )

        if np.any(sw > 0.0):
            if pvt_region.tables.water is None:
                raise ValidationError(
                    f"`PVTNUM` {pvtnum}: water saturation is present "
                    "but no water PVT table is available."
                )

            bw = pvt_region.tables.water.formation_volume_factor(pressure=p, temperature=t)
            if bw is None or rho_w_sc is None:
                raise ValidationError(
                    f"`PVTNUM` {pvtnum}: water FVF table or "
                    "`stock_tank_water_density` is unavailable."
                )
            water_mass[mask] = sw * pv / bw * rho_w_sc

        if rho_g_sc is not None:
            dissolved_gas_mass_in_oil[mask] = (
                rs * oil_mass[mask] * (rho_g_sc / rho_o_sc) * volume_correction
            )

    return Masses(
        oil_mass=typing.cast(CellArray, oil_mass),
        water_mass=typing.cast(CellArray, water_mass),
        free_gas_mass=typing.cast(CellArray, free_gas_mass),
        dissolved_gas_mass_in_oil=typing.cast(CellArray, dissolved_gas_mass_in_oil),
        dissolved_gas_mass_in_water=typing.cast(CellArray, np.zeros(n_cells, dtype=dtype)),
        vaporized_oil_mass_in_gas=typing.cast(CellArray, vaporized_oil_mass_in_gas),
    )
