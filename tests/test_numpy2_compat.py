import numpy as np

from bores.blackoil.satfunc.capillary_pressure.tables import TwoPhaseCapillaryPressureTable
from bores.blackoil.satfunc.relperm.tables import TwoPhaseRelPermTable
from bores.types import FluidPhase


def test_relperm_table_accepts_lists_and_evaluates_arrays():
    table = TwoPhaseRelPermTable(
        wetting_phase=FluidPhase.WATER,
        non_wetting_phase=FluidPhase.OIL,
        reference_saturation=[0.2, 0.5, 0.8],
        wetting_phase_relative_permeability=[0.0, 0.3, 1.0],
        non_wetting_phase_relative_permeability=[1.0, 0.4, 0.0],
        reference_phase="wetting",
    )
    values = table.get_wetting_phase_relative_permeability(np.array([[0.3, 0.6], [0.4, 0.7]]))
    assert values.shape == (2, 2)


def test_capillary_pressure_table_accepts_lists_and_evaluates_arrays():
    table = TwoPhaseCapillaryPressureTable(
        wetting_phase=FluidPhase.WATER,
        non_wetting_phase=FluidPhase.OIL,
        reference_saturation=[0.2, 0.5, 0.8],
        capillary_pressure=[5.0, 1.0, 0.0],
        reference_phase="wetting",
    )
    values = table.get_capillary_pressure(np.array([[0.3, 0.6], [0.4, 0.7]]))
    assert values.shape == (2, 2)
