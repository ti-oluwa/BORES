"""
A well with no BHP, rate or THP limit has nothing to re-solve for, so applying its limits must
leave the governing resolution exactly as the nominal one, whether it has no limits at all or
only economic ones.
"""

import numpy as np
import pytest

from bores.precision import with_precision
from bores.wells.controls import EconomicLimit, EconomicQuantity, WorkoverAction
from bores.wells.hydraulics.beggs_and_brill import beggs_and_brill_wellbore
from bores.wells.resolution.spec import ConnectionPressureMode

from .test_well_derivatives import NORMAL, Well

FIELDS = (
    "bhps",
    "oil_rates",
    "water_rates",
    "gas_rates",
    "surface_oil_rates",
    "surface_water_rates",
    "surface_gas_rates",
    "connection_pressures",
    "connection_oil_rates",
    "connection_water_rates",
    "connection_gas_rates",
)


@pytest.fixture(autouse=True)
def double_precision():
    with with_precision(np.float64):
        yield


def make_well(*, limits):
    return Well(
        kind="producer",
        n=3,
        bhp=1000.0,
        mode=ConnectionPressureMode.HYDRAULIC,
        tubing=beggs_and_brill_wellbore(tubing_inner_diameter=0.25),
        limits=limits,
    )


def test_economic_limits_that_are_not_violated_leave_the_resolution_untouched():
    reference = make_well(limits=None)
    reference.resolve(*NORMAL)
    well = make_well(
        limits=[
            EconomicLimit(
                quantity=EconomicQuantity.OIL_RATE,
                min_value=1e-9,
                workover_action=WorkoverAction.WELL,
            )
        ]
    )
    well.resolve(*NORMAL)

    assert well.workspace.economic_shutins[0] == 0
    for field in FIELDS:
        np.testing.assert_array_equal(
            getattr(well.workspace, field), getattr(reference.workspace, field), err_msg=field
        )
    assert (well.workspace.connection_oil_rates > 0.0).all()


def test_a_violated_economic_limit_zeroes_the_rates_and_keeps_the_pressure_profile():
    reference = make_well(limits=None)
    reference.resolve(*NORMAL)
    well = make_well(
        limits=[
            EconomicLimit(
                quantity=EconomicQuantity.OIL_RATE,
                min_value=1e12,
                workover_action=WorkoverAction.WELL,
            )
        ]
    )
    well.resolve(*NORMAL)

    assert well.workspace.economic_shutins[0] == 1
    np.testing.assert_array_equal(well.workspace.oil_rates[0], 0.0)
    np.testing.assert_array_equal(well.workspace.surface_oil_rates[0], 0.0)
    np.testing.assert_array_equal(
        well.workspace.connection_pressures, reference.workspace.connection_pressures
    )
