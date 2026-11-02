"""
A well under THP control must find the bottom-hole pressure that gives its target tubing head
pressure. The answer is known exactly by resolving the same well under BHP control first, so
each case asks a THP-controlled well for the THP a known BHP produced and checks that it
recovers that BHP.
"""

import numpy as np
import pytest

from bores.precision import with_precision
from bores.wells.compile import InjectorControlModeTag, ProducerControlModeTag
from bores.wells.hydraulics.beggs_and_brill import beggs_and_brill_wellbore
from bores.wells.resolution.spec import ConnectionPressureMode

from .test_well_derivatives import NORMAL, SURFACE, Well


@pytest.fixture(autouse=True)
def double_precision():
    with with_precision(np.float64):
        yield


def make_well(*, kind, bhp):
    return Well(
        kind=kind,
        n=3,
        bhp=bhp,
        mode=ConnectionPressureMode.HYDRAULIC,
        tubing=beggs_and_brill_wellbore(tubing_inner_diameter=0.25),
        surface_sample=SURFACE,
    )


def make_thp_well(*, kind, target_thp):
    well = make_well(kind=kind, bhp=1000.0)
    tag = ProducerControlModeTag.THP if kind == "producer" else InjectorControlModeTag.THP
    well.system.controls.control_modes[0] = tag
    well.system.controls.target_thps[0] = target_thp
    return well


REACHABLE_BHPS = {
    "producer": (900.0, 1500.0, 2200.0, 2800.0),
    "injector": (3300.0, 4000.0, 5000.0, 6000.0),
}
"""Bottom-hole pressures each kind of well can reach: below the reservoir for a producer, above it for an injector."""


@pytest.mark.parametrize(
    ("kind", "bhp"), [(kind, bhp) for kind, values in REACHABLE_BHPS.items() for bhp in values]
)
def test_a_thp_controlled_well_recovers_the_bhp_that_gave_its_target(kind, bhp):
    reference = make_well(kind=kind, bhp=bhp)
    reference.resolve(*NORMAL)
    target_thp = float(reference.workspace.thps[0])

    well = make_thp_well(kind=kind, target_thp=target_thp)
    well.resolve(*NORMAL)

    assert well.workspace.bhps[0] == pytest.approx(bhp, rel=2e-3)
    assert well.workspace.thps[0] == pytest.approx(target_thp, rel=2e-3)
    np.testing.assert_allclose(
        well.workspace.connection_oil_rates, reference.workspace.connection_oil_rates, rtol=5e-3
    )


@pytest.mark.parametrize("kind", ["producer", "injector"])
def test_a_higher_thp_target_gives_a_higher_bhp(kind):
    bhps = []
    for bhp in REACHABLE_BHPS[kind][1:4]:
        reference = make_well(kind=kind, bhp=bhp)
        reference.resolve(*NORMAL)
        well = make_thp_well(kind=kind, target_thp=float(reference.workspace.thps[0]))
        well.resolve(*NORMAL)
        bhps.append(float(well.workspace.bhps[0]))
    assert bhps == sorted(bhps)
    assert len(set(np.round(bhps, 1))) == 3
