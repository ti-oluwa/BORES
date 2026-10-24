"""
Hand-built oil and wet-gas PVT tables from smooth analytic functions, for testing the state
(`Rs`/`Rv`) derivatives and the saturation-pressure modes without needing correlations.
"""

import types

import numpy as np

from bores.blackoil.pvt.data import PVTData
from bores.blackoil.pvt.static import StaticPVT
from bores.blackoil.pvt.tables import PVTTable, PVTTables
from bores.types import FluidPhase

PRESSURES = np.linspace(500.0, 6000.0, 23)
TEMPERATURES = np.array([150.0, 200.0, 250.0])
RS_AXIS = np.linspace(20.0, 800.0, 14)
RV_AXIS = np.linspace(0.0, 1.0e-4, 9)


def _solubility_coefficient(temperature):
    return 0.02 * (1.0 - 0.002 * (temperature - 150.0))


def solution_gor_at_saturation(pressure, temperature):
    return _solubility_coefficient(temperature) * pressure**1.15


def bubble_point(solution_gor, temperature):
    return (solution_gor / _solubility_coefficient(temperature)) ** (1.0 / 1.15)


def _oil_fvf_at_saturation(pressure, temperature):
    return (
        1.05
        + 5e-4 * solution_gor_at_saturation(pressure, temperature)
        + 2e-4 * (temperature - 150.0) / 50.0
    )


def _oil_viscosity_at_saturation(pressure, temperature):
    return (
        2.0
        * np.exp(-0.0011 * solution_gor_at_saturation(pressure, temperature))
        * (1.0 - 0.003 * (temperature - 150.0))
    )


def _static_pvt() -> StaticPVT:
    """Stock-tank densities, which enable the undersaturated density correction."""
    return StaticPVT(stock_tank_oil_density=53.0, stock_tank_gas_density=0.0494)


def _oil_density_at_saturation(pressure, temperature):
    return (
        56.0
        - 0.012 * solution_gor_at_saturation(pressure, temperature)
        - 0.01 * (temperature - 150.0)
        + 4e-4 * pressure
    )


def build_oil_table(with_undersaturated_tables: bool = False, dtype=None) -> PVTTable:
    """
    An oil table. With `with_undersaturated_tables` the undersaturated `Bo` and `μo` are read
    from 3-D `(P, T, Rs)` tables, otherwise they come from the bubble-point value and the
    compressibility.
    """
    pressure_grid, temperature_grid = np.meshgrid(PRESSURES, TEMPERATURES, indexing="ij")
    data = {
        "phase": FluidPhase.OIL,
        "pressures": PRESSURES,
        "temperatures": TEMPERATURES,
        "solution_gas_to_oil_ratios": RS_AXIS,
        "bubble_point_pressures": np.array([
            [bubble_point(rs, t) for t in TEMPERATURES] for rs in RS_AXIS
        ]),
        "viscosity_table": _oil_viscosity_at_saturation(pressure_grid, temperature_grid),
        "formation_volume_factor_table": _oil_fvf_at_saturation(pressure_grid, temperature_grid),
        "solution_gor_table": solution_gor_at_saturation(pressure_grid, temperature_grid),
        "compressibility_table": np.full_like(pressure_grid, 1.5e-5),
        "density_table": _oil_density_at_saturation(pressure_grid, temperature_grid),
        "dtype": dtype,
    }
    if with_undersaturated_tables:
        fvf = np.zeros((len(PRESSURES), len(TEMPERATURES), len(RS_AXIS)))
        viscosity = np.zeros_like(fvf)
        for k, rs in enumerate(RS_AXIS):
            for j, t in enumerate(TEMPERATURES):
                pb = bubble_point(rs, t)
                excess = np.maximum(PRESSURES - pb, 0.0)
                fvf[:, j, k] = _oil_fvf_at_saturation(pb, t) * np.exp(-1.5e-5 * excess)
                viscosity[:, j, k] = _oil_viscosity_at_saturation(pb, t) * (1.0 + 4e-5 * excess)
        data["undersaturated_formation_volume_factor_table"] = fvf
        data["undersaturated_viscosity_table"] = viscosity
    return PVTTable(data=PVTData(**data), pvt=_static_pvt(), validate=False)


def _vaporization_coefficient(temperature):
    return 1.5e-8 * (1.0 + 0.001 * (temperature - 150.0))


def vaporized_oil_ratio_at_saturation(pressure, temperature):
    return _vaporization_coefficient(temperature) * pressure


def dew_point(vaporized_oil_ratio, temperature):
    return vaporized_oil_ratio / _vaporization_coefficient(temperature)


def _gas_fvf_dry(pressure, temperature):
    return 0.0283 * 0.9 * (temperature + 460.0) / pressure


def _gas_viscosity_dry(pressure, temperature):
    return 0.012 + 2e-6 * pressure + 1e-5 * (temperature - 150.0)


def _gas_density_at_saturation(pressure, temperature):
    return 0.0283 * 0 + 0.0763 * pressure / (0.9 * (temperature + 460.0)) * (
        1.0 + 800.0 * vaporized_oil_ratio_at_saturation(pressure, temperature)
    )


def build_gas_table(dtype=None) -> PVTTable:
    """A wet-gas table whose undersaturated `Bg` and `μg` come from 3-D `(P, T, Rv)` tables."""
    pressure_grid, temperature_grid = np.meshgrid(PRESSURES, TEMPERATURES, indexing="ij")
    fvf = np.zeros((len(PRESSURES), len(TEMPERATURES), len(RV_AXIS)))
    viscosity = np.zeros_like(fvf)
    for k, rv in enumerate(RV_AXIS):
        fvf[:, :, k] = _gas_fvf_dry(pressure_grid, temperature_grid) * (1.0 + 2000.0 * rv)
        viscosity[:, :, k] = _gas_viscosity_dry(pressure_grid, temperature_grid) * (
            1.0 + 500.0 * rv
        )
    saturated_rv = vaporized_oil_ratio_at_saturation(pressure_grid, temperature_grid)
    data = {
        "phase": FluidPhase.GAS,
        "pressures": PRESSURES,
        "temperatures": TEMPERATURES,
        "vaporized_oil_ratios": RV_AXIS,
        "dew_point_pressures": np.array([
            [dew_point(rv, t) for t in TEMPERATURES] for rv in RV_AXIS
        ]),
        "vaporized_oil_ratio_table": saturated_rv,
        "viscosity_table": _gas_viscosity_dry(pressure_grid, temperature_grid)
        * (1.0 + 500.0 * saturated_rv),
        "formation_volume_factor_table": _gas_fvf_dry(pressure_grid, temperature_grid)
        * (1.0 + 2000.0 * saturated_rv),
        "compressibility_factor_table": 0.95 - 4e-5 * pressure_grid,
        "density_table": _gas_density_at_saturation(pressure_grid, temperature_grid),
        "undersaturated_formation_volume_factor_table": fvf,
        "undersaturated_viscosity_table": viscosity,
        "dtype": dtype,
    }
    return PVTTable(data=PVTData(**data), pvt=_static_pvt(), validate=False)


class SingleRegionPVT:
    """Stands in for `PVT`: every region has the same tables (`compute_pvt_cache` only indexes it)."""

    def __init__(self, oil: PVTTable | None = None, gas: PVTTable | None = None):
        self._region = types.SimpleNamespace(tables=PVTTables(oil=oil, gas=gas))

    def region(self, pvtnum: int):
        return self._region

    def __getitem__(self, key: int):
        return self.region(key)
