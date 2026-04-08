import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.constants import c
from bores.datastructures import Rates
from bores.models import FluidProperties, RockProperties
from bores.types import NDimensionalGrid, ThreeDimensions

__all__ = ["MaterialBalanceErrors", "compute_material_balance_errors"]


@attrs.frozen(slots=True)
class MaterialBalanceErrors:
    """
    Material balance errors computed at the end of a time step.

    All volumetric quantities are in reservoir cubic feet (ft³).
    Relative errors are dimensionless fractions; multiply by 100 for percent.

    The gas phase accounts for free gas plus solution gas dissolved in oil
    (via Rs, SCF/STB) and solution gas dissolved in water (via Rsw, SCF/STB),
    converted to a common ft³ basis using the mean cell gas FVF (Bg).
    """

    absolute_oil_mbe: float
    """
    Absolute oil material balance error (ft³).

    Defined as ΔVp·So - (Q_inj_o - Q_prod_o)·Δt, where all rates are in
    ft³/day and Δt is in days. Positive means more oil accumulated in
    the reservoir than the net inflow accounts for; negative means a deficit.
    """

    absolute_water_mbe: float
    """
    Absolute water material balance error (ft³).

    Analogous to `absolute_oil_mbe` for the water phase.
    """

    absolute_gas_mbe: float
    """
    Absolute gas material balance error (ft³).

    Computed in surface SCF (free gas + Rs·oil + Rsw·water) then converted
    to ft³ via the grid-mean gas FVF (Bg) for unit consistency with the
    oil and water MBEs. Positive means more gas remains in place than the
    net surface inflow accounts for.
    """

    total_absolute_mbe: float
    """
    Sum of absolute oil, water, and gas material balance errors (ft³).

    Provides a single scalar to assess overall volumetric closure.
    """

    relative_oil_mbe: float
    """
    Oil MBE normalised by the previous-step oil pore volume (dimensionless fraction).

    `relative_oil_mbe = absolute_oil_mbe / max(|previous_oil_pore_volume|, 1)`.
    Multiply by 100 for percent. Typical acceptance threshold: |value| < 0.01 (1 %).
    """

    relative_water_mbe: float
    """
    Water MBE normalised by the previous-step water pore volume (dimensionless fraction).

    Analogous to `relative_oil_mbe` for the water phase.
    """

    relative_gas_mbe: float
    """
    Gas MBE normalised by the previous-step gas-in-place equivalent volume (dimensionless fraction).

    Reference volume is `|previous_gas_SCF| x mean_Bg` so the denominator is
    in the same ft³ basis as `absolute_gas_mbe`.
    """

    total_relative_mbe: float
    """
    Total MBE normalised by the sum of previous-step phase reference volumes (dimensionless fraction).

    `total_relative_mbe = total_absolute_mbe / max(|Vo_prev| + |Vw_prev| + |Vg_ref_prev|, 1)`.
    """

    @classmethod
    def null(cls) -> Self:
        return cls(
            absolute_oil_mbe=0,
            absolute_gas_mbe=0,
            absolute_water_mbe=0,
            total_absolute_mbe=0,
            relative_oil_mbe=0,
            relative_gas_mbe=0,
            relative_water_mbe=0,
            total_relative_mbe=0,
        )


def compute_material_balance_errors(
    current_fluid_properties: FluidProperties[ThreeDimensions],
    previous_fluid_properties: FluidProperties[ThreeDimensions],
    rock: RockProperties[ThreeDimensions],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    injection_rates: Rates[float, ThreeDimensions],
    production_rates: Rates[float, ThreeDimensions],
    time_step_size: float,
) -> MaterialBalanceErrors:
    """
    Compute per-phase material balance errors at reservoir conditions (ft³) for a timestep.

    All rate tensors hold values in ft³/day (reservoir condition).

    MBE = ΔV_reservoir - (Q_inj - Q_prod) * Δt

    Gas MBE includes free gas plus solution gas dissolved in oil (Rs) and water (Rsw).

    :param current_fluid_properties: Fluid properties at end of time step.
    :param previous_fluid_properties: Fluid properties at start of time step.
    :param rock: Rock properties (porosity, NTG, etc.).
    :param thickness_grid: Un-padded cell thickness grid (ft).
    :param cell_dimension: (dx, dy) in ft.
    :param injection_rates: Injection rate sparse tensors (ft³/day, res cond).
    :param production_rates: Production rate sparse tensors (ft³/day, res cond).
    :param time_step_size: Time step size in seconds.
    :return: `MaterialBalanceErrors` instance.
    """
    cell_size_x, cell_size_y = cell_dimension
    # Cell pore volume in ft³
    pore_volume = (
        rock.porosity_grid
        * rock.net_to_gross_ratio_grid
        * thickness_grid
        * cell_size_x
        * cell_size_y
    )
    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND

    # OIL
    current_oil_volume = float(
        np.sum(pore_volume * current_fluid_properties.oil_saturation_grid)
    )
    previous_oil_volume = float(
        np.sum(pore_volume * previous_fluid_properties.oil_saturation_grid)
    )
    oil_volume_change = current_oil_volume - previous_oil_volume

    net_oil_inflow = 0.0
    for key in injection_rates.oil:
        net_oil_inflow += float(injection_rates.oil[key])
    for key in production_rates.oil:
        net_oil_inflow -= float(production_rates.oil[key])
    net_oil_inflow *= time_step_size_in_days  # ft³

    absolute_oil_mbe = oil_volume_change - net_oil_inflow
    reference_oil = max(abs(previous_oil_volume), 1.0)
    relative_oil_mbe = absolute_oil_mbe / reference_oil

    # WATER
    current_water_volume = float(
        np.sum(pore_volume * current_fluid_properties.water_saturation_grid)
    )
    previous_water_volume = float(
        np.sum(pore_volume * previous_fluid_properties.water_saturation_grid)
    )
    water_volume_change = current_water_volume - previous_water_volume

    net_water_inflow = 0.0
    for key in injection_rates.water:
        net_water_inflow += float(injection_rates.water[key])
    for key in production_rates.water:
        net_water_inflow -= float(production_rates.water[key])
    net_water_inflow *= time_step_size_in_days

    absolute_water_mbe = water_volume_change - net_water_inflow
    reference_water = max(abs(previous_water_volume), 1.0)
    relative_water_mbe = absolute_water_mbe / reference_water

    # GAS
    # Total gas in place (ft³) = free gas + solution gas in oil + solution gas in water
    # Free gas (ft³):           Vp * Sg
    # Solution gas in oil (ft³): Vp * So * Rs * Bg   where Bg = Bo_g / Bo  ... complex
    # Simpler consistent approach: convert everything to surface SCF then back.
    #
    # Gas in place at surface (SCF):
    #   G = Vp * Sg / Bg  +  Vp * So * Rs / Bo  +  Vp * Sw * Rsw / Bw
    # where Bg [ft³/SCF], Bo [res bbl/STB→ ft³ via *5.614583], Bw similar.
    #
    # To stay in ft³:  multiply G_surface by Bg_ref (use current Bg field mean).
    # But that introduces a reference Bg choice. The cleanest approach:
    # track G in surface SCF and report MBE in surface SCF, then convert to ft³
    # using the mean Bg for reporting alongside oil and water in ft³.
    #
    # We'll report all MBEs in ft³ using phase-specific FVFs for gas.

    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET  # 5.614583 ft³/bbl

    # Bg in ft³/SCF  (gas_formation_volume_factor_grid is in ft³/SCF already per models.py)
    current_gas_fvf = current_fluid_properties.gas_formation_volume_factor_grid
    previous_gas_fvf = previous_fluid_properties.gas_formation_volume_factor_grid

    # Bo in ft³/STB  (oil_formation_volume_factor_grid is bbl/STB → x 5.614583)
    current_oil_fvf = (
        current_fluid_properties.oil_formation_volume_factor_grid * bbl_to_ft3
    )
    previous_oil_fvf = (
        previous_fluid_properties.oil_formation_volume_factor_grid * bbl_to_ft3
    )

    # Bw in ft³/STB
    current_water_fvf = (
        current_fluid_properties.water_formation_volume_factor_grid * bbl_to_ft3
    )
    previous_water_fvf = (
        previous_fluid_properties.water_formation_volume_factor_grid * bbl_to_ft3
    )

    # Rs in SCF/STB, Rsw in SCF/STB
    current_solution_gor = current_fluid_properties.solution_gas_to_oil_ratio_grid
    previous_solution_gor = previous_fluid_properties.solution_gas_to_oil_ratio_grid
    current_gas_solubility_in_water = (
        current_fluid_properties.gas_solubility_in_water_grid
    )
    previous_gas_solubility_in_water = (
        previous_fluid_properties.gas_solubility_in_water_grid
    )

    # Total gas in place in surface SCF
    # free gas: (Vp * Sg) / Bg
    # dissolved in oil: (Vp * So / Bo) * Rs
    # dissolved in water: (Vp * Sw / Bw) * Rsw
    current_gas_volume_scf = float(
        np.sum(
            (
                pore_volume
                * current_fluid_properties.gas_saturation_grid
                / np.maximum(current_gas_fvf, 1e-30)
            )
            + (
                pore_volume
                * current_fluid_properties.oil_saturation_grid
                / np.maximum(current_oil_fvf, 1e-30)
                * current_solution_gor
            )
            + (
                pore_volume
                * current_fluid_properties.water_saturation_grid
                / np.maximum(current_water_fvf, 1e-30)
                * current_gas_solubility_in_water
            )
        )
    )
    previous_gas_volume_scf = float(
        np.sum(
            pore_volume
            * previous_fluid_properties.gas_saturation_grid
            / np.maximum(previous_gas_fvf, 1e-30)
            + pore_volume
            * previous_fluid_properties.oil_saturation_grid
            / np.maximum(previous_oil_fvf, 1e-30)
            * previous_solution_gor
            + pore_volume
            * previous_fluid_properties.water_saturation_grid
            / np.maximum(previous_water_fvf, 1e-30)
            * previous_gas_solubility_in_water
        )
    )
    gas_volume_change_scf = current_gas_volume_scf - previous_gas_volume_scf

    # Net gas inflow: injection_rates.gas and production_rates.gas are in ft³/day.
    # Convert to SCF/day using Bg at current conditions (cell-averaged).
    # For simplicity use grid-mean Bg; or sum cell-by-cell for accuracy.
    net_gas_inflow_scf = 0.0
    for key in injection_rates.gas:
        cell_idx = key
        gas_fvf = (
            float(current_gas_fvf[cell_idx])
            if current_gas_fvf[cell_idx] > 0
            else float(np.mean(current_gas_fvf))
        )
        net_gas_inflow_scf += float(injection_rates.gas[key]) / gas_fvf
    for key in production_rates.gas:
        cell_idx = key
        gas_fvf = (
            float(current_gas_fvf[cell_idx])
            if current_gas_fvf[cell_idx] > 0
            else float(np.mean(current_gas_fvf))
        )
        net_gas_inflow_scf -= float(production_rates.gas[key]) / gas_fvf

    net_gas_inflow_scf *= time_step_size_in_days  # SCF

    absolute_gas_mbe_scf = gas_volume_change_scf - net_gas_inflow_scf  # SCF
    # Convert to ft³ using mean current Bg for consistent reporting units
    mean_gas_fvf = float(np.mean(current_gas_fvf))
    absolute_gas_mbe = absolute_gas_mbe_scf * mean_gas_fvf  # ft³
    reference_gas = max(abs(previous_gas_volume_scf) * mean_gas_fvf, 1.0)
    relative_gas_mbe = absolute_gas_mbe / reference_gas

    # TOTAL
    total_absolute_mbe = absolute_oil_mbe + absolute_water_mbe + absolute_gas_mbe
    total_reference = max(
        abs(previous_oil_volume) + abs(previous_water_volume) + reference_gas, 1.0
    )
    total_relative_mbe = total_absolute_mbe / total_reference

    return MaterialBalanceErrors(
        absolute_oil_mbe=absolute_oil_mbe,
        absolute_water_mbe=absolute_water_mbe,
        absolute_gas_mbe=absolute_gas_mbe,
        total_absolute_mbe=total_absolute_mbe,
        relative_oil_mbe=relative_oil_mbe,
        relative_water_mbe=relative_water_mbe,
        relative_gas_mbe=relative_gas_mbe,
        total_relative_mbe=total_relative_mbe,
    )
