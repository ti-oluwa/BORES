"""
`resolve_wells` resolves the wells it can in one compiled pass over all of them. For every well
that pass takes, it must give what resolving the well on its own gives, so each case here runs
both and compares every field a resolution writes. A spy on the one-at-a-time path shows which
wells each pass really took.
"""

import attrs
import numpy as np
import pytest

from bores.errors import ValidationError
from bores.precision import get_dtype, with_precision
from bores.types import FluidPhase
from bores.wells.compile import (
    InjectorControlModeTag,
    ProducerControlModeTag,
)
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    RateLimit,
    RateQuantity,
    THPLimit,
    WorkoverAction,
)
from bores.wells.hydraulics.base import compile_connection_fluids
from bores.wells.hydraulics.beggs_and_brill import beggs_and_brill_wellbore
from bores.wells.hydraulics.gray import gray_wellbore
from bores.wells.hydraulics.hagedorn_brown import hagedorn_brown_wellbore
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
from bores.wells.hydraulics.woldesemayat_ghajar import woldesemayat_ghajar_wellbore
from bores.wells.resolution import system as system_module
from bores.wells.resolution.spec import ConnectionPressureMode
from bores.wells.resolution.system import resolve_wells

from .test_well_derivatives import NORMAL, SURFACE, Well, make_samples

FIELDS = (
    "bhps",
    "oil_rates",
    "water_rates",
    "gas_rates",
    "surface_oil_rates",
    "surface_water_rates",
    "surface_gas_rates",
    "thps",
    "active_limit_rows",
    "economic_shutins",
    "connection_pressures",
    "connection_oil_rates",
    "connection_water_rates",
    "connection_gas_rates",
)

CORRELATIONS = {
    "homogeneous": homogeneous_wellbore,
    "beggs_and_brill": beggs_and_brill_wellbore,
    "hagedorn_brown": hagedorn_brown_wellbore,
    "gray": gray_wellbore,
    "woldesemayat_ghajar": woldesemayat_ghajar_wellbore,
}


@pytest.fixture(autouse=True, params=[np.float64, np.float32], ids=["float64", "float32"])
def precision(request):
    with with_precision(request.param):
        yield request.param


@pytest.fixture
def spy(monkeypatch):
    """The wells `resolve_wells` resolved one at a time."""
    rows = []
    original = system_module.resolve_well_control

    def record(**kwargs):
        rows.append(int(kwargs["well_row"]))
        return original(**kwargs)

    monkeypatch.setattr(system_module, "resolve_well_control", record)
    return rows


def fluids_for(*, count=3, dtype=None):
    dtype = np.dtype(get_dtype()) if dtype is None else dtype
    pressures, water, gas = (values[:count] for values in NORMAL)
    return compile_connection_fluids(
        connection_samples=make_samples(pressures, water, gas), dtype=dtype
    )


def resolve(well, *, compiled, tubing=None):
    spec = attrs.evolve(well.spec, use_compiled_resolution=compiled)
    fluids = fluids_for(count=well.n)
    resolve_wells(
        compiled_system=well.system,
        workspace=well.workspace,
        spec=spec,
        get_tubing_performance=lambda row: well.tubing,
        get_connection_fluids=lambda row: fluids,
        get_surface_sample=(lambda row: well.surface_sample) if well.surface_sample else None,
    )
    return well.workspace


def assert_same(actual, expected):
    tolerance = 1e-10 if np.dtype(get_dtype()) == np.float64 else 2e-5
    for field in FIELDS:
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
            err_msg=field,
        )


def set_control(well, mode, target_rate=None):
    well.system.controls.control_modes[0] = mode
    if target_rate is not None:
        well.system.controls.target_rates[0] = target_rate


def make_well(*, kind, correlation, mode=ConnectionPressureMode.HYDRAULIC, **options):
    tubing = CORRELATIONS[correlation](tubing_inner_diameter=0.25) if correlation else None
    bhp = options.pop("bhp", 1000.0 if kind == "producer" else 4500.0)
    return Well(
        kind=kind,
        n=3,
        bhp=bhp,
        mode=mode,
        tubing=tubing,
        dtype=get_dtype(),
        **options,
    )


def check_same_as_one_at_a_time(build, spy):
    reference = build()
    expected = resolve(reference, compiled=False)
    well = build()
    spy.clear()
    actual = resolve(well, compiled=True)
    assert_same(actual, expected)
    return spy.copy()


@pytest.mark.parametrize("correlation", list(CORRELATIONS))
@pytest.mark.parametrize("kind", ["producer", "injector"])
def test_a_bhp_controlled_well_matches_for_every_correlation(correlation, kind, spy):
    taken_by_python = check_same_as_one_at_a_time(
        lambda: make_well(kind=kind, correlation=correlation), spy
    )
    assert taken_by_python == []


@pytest.mark.parametrize("mode", ["hydrostatic", "uniform"])
@pytest.mark.parametrize("kind", ["producer", "injector"])
def test_a_well_without_a_wellbore_model_matches_for_each_pressure_mode(mode, kind, spy):
    pressure_mode = {
        "hydrostatic": ConnectionPressureMode.HYDROSTATIC,
        "uniform": ConnectionPressureMode.UNIFORM_BHP,
    }[mode]
    assert (
        check_same_as_one_at_a_time(
            lambda: make_well(kind=kind, correlation=None, mode=pressure_mode), spy
        )
        == []
    )


PRODUCER_MODES = [
    (ProducerControlModeTag.OIL_RATE, ("surface_oil_rates",)),
    (ProducerControlModeTag.WATER_RATE, ("surface_water_rates",)),
    (ProducerControlModeTag.GAS_RATE, ("surface_gas_rates",)),
    (ProducerControlModeTag.LIQUID_RATE, ("surface_oil_rates", "surface_water_rates")),
    (ProducerControlModeTag.RESERVOIR_VOLUME_RATE, ("oil_rates", "water_rates", "gas_rates")),
]


@pytest.mark.parametrize("correlation", ["beggs_and_brill", "gray", "homogeneous"])
@pytest.mark.parametrize(("mode", "fields"), PRODUCER_MODES)
def test_a_rate_controlled_producer_matches_and_hits_its_target(mode, fields, correlation, spy):
    reference = make_well(kind="producer", correlation=correlation)
    workspace = resolve(reference, compiled=False)
    target = sum(float(getattr(workspace, field)[0]) for field in fields)

    def build():
        well = make_well(kind="producer", correlation=correlation)
        set_control(well, mode, target_rate=target)
        return well

    assert check_same_as_one_at_a_time(build, spy) == []
    resolved = resolve(build(), compiled=True)
    assert float(resolved.bhps[0]) == pytest.approx(1000.0, rel=5e-3)


@pytest.mark.parametrize(
    "correlation", ["beggs_and_brill", "hagedorn_brown", "woldesemayat_ghajar"]
)
@pytest.mark.parametrize(
    ("mode", "phase", "field"),
    [
        (InjectorControlModeTag.RATE, FluidPhase.WATER, "surface_water_rates"),
        (InjectorControlModeTag.RATE, FluidPhase.GAS, "surface_gas_rates"),
        (InjectorControlModeTag.RESERVOIR_VOLUME_RATE, FluidPhase.WATER, "water_rates"),
    ],
)
def test_a_rate_controlled_injector_matches_and_hits_its_target(
    mode, phase, field, correlation, spy
):
    reference = make_well(kind="injector", correlation=correlation, injected=phase)
    target = float(getattr(resolve(reference, compiled=False), field)[0])

    def build():
        well = make_well(kind="injector", correlation=correlation, injected=phase)
        set_control(well, mode, target_rate=target)
        return well

    assert check_same_as_one_at_a_time(build, spy) == []
    assert float(resolve(build(), compiled=True).bhps[0]) == pytest.approx(4500.0, rel=5e-3)


def rate_of(kind, field, **options):
    well = make_well(kind=kind, correlation="beggs_and_brill", **options)
    return float(getattr(resolve(well, compiled=False), field)[0])


@pytest.mark.parametrize(
    "limits_for",
    [
        lambda: [BHPLimit(min_value=1800.0)],
        lambda: [BHPLimit(max_value=900.0, min_value=100.0)],
        lambda: [
            RateLimit(
                quantity=RateQuantity.OIL, max_value=0.4 * rate_of("producer", "surface_oil_rates")
            )
        ],
        lambda: [
            BHPLimit(min_value=1200.0),
            RateLimit(
                quantity=RateQuantity.LIQUID,
                max_value=0.5 * rate_of("producer", "surface_oil_rates"),
            ),
        ],
        lambda: [
            EconomicLimit(
                quantity=EconomicQuantity.OIL_RATE,
                min_value=1e12,
                workover_action=WorkoverAction.WELL,
            )
        ],
        lambda: [
            BHPLimit(min_value=1500.0),
            EconomicLimit(
                quantity=EconomicQuantity.OIL_RATE,
                min_value=1e12,
                workover_action=WorkoverAction.WELL,
            ),
        ],
        lambda: [
            RateLimit(quantity=RateQuantity.OIL, max_value=1e15),
            BHPLimit(min_value=1.0),
        ],
    ],
    ids=[
        "bhp_floor_binds",
        "bhp_cap_not_binding",
        "rate_cap_binds",
        "bhp_and_rate_both",
        "economic_violated",
        "bhp_binds_then_economic",
        "limits_not_binding",
    ],
)
def test_a_producer_with_limits_matches(limits_for, spy):
    assert (
        check_same_as_one_at_a_time(
            lambda: make_well(kind="producer", correlation="beggs_and_brill", limits=limits_for()),
            spy,
        )
        == []
    )


@pytest.mark.parametrize(
    "limits_for",
    [
        lambda: [BHPLimit(max_value=4200.0)],
        lambda: [
            RateLimit(
                quantity=RateQuantity.WATER,
                max_value=0.5
                * rate_of("injector", "surface_water_rates", injected=FluidPhase.WATER),
            )
        ],
    ],
    ids=["bhp_cap_binds", "rate_cap_binds"],
)
def test_an_injector_with_limits_matches(limits_for, spy):
    assert (
        check_same_as_one_at_a_time(
            lambda: make_well(
                kind="injector",
                correlation="beggs_and_brill",
                injected=FluidPhase.WATER,
                limits=limits_for(),
            ),
            spy,
        )
        == []
    )


@pytest.mark.parametrize("d_factor", [None, 3e-6])
def test_a_non_darcy_well_matches(d_factor, spy):
    assert (
        check_same_as_one_at_a_time(
            lambda: make_well(kind="producer", correlation="gray", d_factor=d_factor), spy
        )
        == []
    )


def test_a_well_with_a_surface_sample_reports_the_same_tubing_head_pressure(spy):
    assert (
        check_same_as_one_at_a_time(
            lambda: make_well(
                kind="producer", correlation="beggs_and_brill", surface_sample=SURFACE
            ),
            spy,
        )
        == []
    )
    well = make_well(kind="producer", correlation="beggs_and_brill", surface_sample=SURFACE)
    assert np.isfinite(resolve(well, compiled=True).thps[0])


def test_a_crossflowing_injector_is_reported_as_one_at_a_time_reports_it(spy):
    def build():
        well = make_well(kind="injector", correlation="beggs_and_brill", surface_sample=SURFACE)
        return well

    check_same_as_one_at_a_time(build, spy)


def thp_of(*, kind, correlation, **options):
    """The tubing head pressure a well under BHP control reports."""
    well = make_well(kind=kind, correlation=correlation, surface_sample=SURFACE, **options)
    return float(resolve(well, compiled=False).thps[0])


def make_thp_well(*, kind, correlation, target_thp, **options):
    well = make_well(kind=kind, correlation=correlation, surface_sample=SURFACE, **options)
    tag = ProducerControlModeTag.THP if kind == "producer" else InjectorControlModeTag.THP
    well.system.controls.control_modes[0] = tag
    well.system.controls.target_thps[0] = target_thp
    return well


@pytest.mark.parametrize("correlation", list(CORRELATIONS))
@pytest.mark.parametrize("kind", ["producer", "injector"])
def test_a_thp_controlled_well_matches_and_recovers_its_bhp(correlation, kind, spy):
    target = thp_of(kind=kind, correlation=correlation)
    expected_bhp = 1000.0 if kind == "producer" else 4500.0

    def build():
        return make_thp_well(kind=kind, correlation=correlation, target_thp=target)

    assert check_same_as_one_at_a_time(build, spy) == []
    assert float(resolve(build(), compiled=True).bhps[0]) == pytest.approx(expected_bhp, rel=5e-3)


@pytest.mark.parametrize("correlation", ["beggs_and_brill", "gray", "homogeneous"])
@pytest.mark.parametrize(
    "case",
    [
        "producer_floor_binds",
        "producer_floor_not_binding",
        "injector_cap_binds",
        "injector_cap_loose",
    ],
)
def test_a_well_with_a_thp_limit_matches(correlation, case, spy):
    kind = "producer" if case.startswith("producer") else "injector"
    nominal = thp_of(kind=kind, correlation=correlation)
    limit = {
        "producer_floor_binds": THPLimit(min_value=nominal + 150.0),
        "producer_floor_not_binding": THPLimit(min_value=nominal - 150.0),
        "injector_cap_binds": THPLimit(max_value=nominal - 200.0),
        "injector_cap_loose": THPLimit(max_value=nominal + 200.0),
    }[case]

    def build():
        return make_well(
            kind=kind, correlation=correlation, surface_sample=SURFACE, limits=[limit]
        )

    assert check_same_as_one_at_a_time(build, spy) == []


@pytest.mark.parametrize("kind", ["producer", "injector"])
def test_a_well_held_by_thp_bhp_and_rate_limits_matches(kind, spy):
    nominal = thp_of(kind=kind, correlation="beggs_and_brill")
    if kind == "producer":
        limits = [
            THPLimit(min_value=nominal + 100.0),
            BHPLimit(min_value=1300.0),
            RateLimit(
                quantity=RateQuantity.OIL, max_value=0.8 * rate_of("producer", "surface_oil_rates")
            ),
        ]
    else:
        limits = [
            THPLimit(max_value=nominal - 150.0),
            BHPLimit(max_value=4300.0),
        ]

    def build():
        return make_well(
            kind=kind, correlation="beggs_and_brill", surface_sample=SURFACE, limits=limits
        )

    assert check_same_as_one_at_a_time(build, spy) == []


@pytest.mark.parametrize("kind", ["producer", "injector"])
def test_a_thp_controlled_well_without_a_surface_sample_raises_as_one_at_a_time_does(kind, spy):
    def build():
        well = make_well(kind=kind, correlation="beggs_and_brill")
        tag = ProducerControlModeTag.THP if kind == "producer" else InjectorControlModeTag.THP
        well.system.controls.control_modes[0] = tag
        well.system.controls.target_thps[0] = 500.0
        return well

    for compiled in (False, True):
        spy.clear()
        with pytest.raises(ValidationError, match="surface_sample"):
            resolve(build(), compiled=compiled)
    assert spy == [0]


def test_a_well_with_a_thp_limit_and_no_surface_sample_raises_as_one_at_a_time_does(spy):
    def build():
        return make_well(
            kind="producer", correlation="beggs_and_brill", limits=[THPLimit(min_value=300.0)]
        )

    for compiled in (False, True):
        with pytest.raises(ValidationError):
            resolve(build(), compiled=compiled)


def test_the_tubing_head_pressure_of_a_stopped_flow_is_reported_as_one_at_a_time_reports_it(spy):
    def build():
        well = make_well(kind="producer", correlation="beggs_and_brill", surface_sample=SURFACE)
        set_control(well, ProducerControlModeTag.OIL_RATE, target_rate=0.0)
        return well

    check_same_as_one_at_a_time(build, spy)


def test_a_well_with_a_shut_connection_is_left_to_the_one_at_a_time_path(spy):
    def build():
        well = make_well(kind="producer", correlation="beggs_and_brill")
        well.system.perforations.completion_statuses[1] = 0
        return well

    reference = build()
    fluids = fluids_for(count=2)
    expected = build()
    resolve_wells(
        compiled_system=expected.system,
        workspace=expected.workspace,
        spec=attrs.evolve(expected.spec, use_compiled_resolution=False),
        get_tubing_performance=lambda row: expected.tubing,
        get_connection_fluids=lambda row: fluids,
    )
    spy.clear()
    resolve_wells(
        compiled_system=reference.system,
        workspace=reference.workspace,
        spec=reference.spec,
        get_tubing_performance=lambda row: reference.tubing,
        get_connection_fluids=lambda row: fluids,
    )
    assert spy == [0]
    assert_same(reference.workspace, expected.workspace)


def test_a_pass_that_cannot_converge_falls_back_and_raises_the_usual_error(spy):
    from bores.errors import WellConvergenceError

    well = make_well(kind="producer", correlation="beggs_and_brill")
    well.spec = attrs.evolve(
        well.spec, max_fixed_point_iterations=1, rate_convergence_tolerance=1e-14
    )
    with pytest.raises(WellConvergenceError):
        resolve(well, compiled=True)
    assert spy == [0]


def test_the_compiled_pass_can_be_turned_off(spy):
    well = make_well(kind="producer", correlation="beggs_and_brill")
    resolve(well, compiled=False)
    assert spy == [0]


@pytest.mark.parametrize(
    ("bhp_cap", "rate_fraction"),
    [(4200.0, 0.5), (4400.0, 0.9), (4100.0, 0.3)],
    ids=["both_bind", "bhp_cap_looser", "rate_cap_tighter"],
)
def test_an_injector_held_by_two_limits_follows_the_one_that_allows_less(
    bhp_cap, rate_fraction, spy
):
    water_rate = rate_of("injector", "surface_water_rates", injected=FluidPhase.WATER)

    def build():
        return make_well(
            kind="injector",
            correlation="beggs_and_brill",
            injected=FluidPhase.WATER,
            limits=[
                BHPLimit(max_value=bhp_cap),
                RateLimit(quantity=RateQuantity.WATER, max_value=rate_fraction * water_rate),
            ],
        )

    assert check_same_as_one_at_a_time(build, spy) == []
    workspace = resolve(build(), compiled=True)
    assert float(workspace.bhps[0]) <= bhp_cap + 1e-6
    assert float(workspace.surface_water_rates[0]) <= rate_fraction * water_rate * 1.001


@pytest.mark.parametrize(
    "quantity", [RateQuantity.GAS, RateQuantity.WATER, RateQuantity.RESERVOIR], ids=str
)
def test_a_rate_limit_that_does_not_bind_leaves_the_nominal_resolution(quantity, spy):
    def build():
        return make_well(
            kind="producer",
            correlation="beggs_and_brill",
            d_factor=3e-6,
            limits=[RateLimit(quantity=quantity, max_value=1e15)],
        )

    assert check_same_as_one_at_a_time(build, spy) == []
