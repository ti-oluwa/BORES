"""
Well derivatives are checked against central finite differences of the resolver itself, run in
double precision, on fluids whose properties have exact analytic derivatives. The resolver
is solved at a fixed bottom-hole pressure, so a difference of its output is the total derivative
the implicit solve claims to give.
"""

import numpy as np
import pytest

from bores.constants import c
from bores.errors import ValidationError
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.precision import with_precision
from bores.types import Orientation, PhaseValues
from bores.wells.base import FluidPhase
from bores.wells.compile import compile_well_system
from bores.wells.controls import EconomicLimit, EconomicQuantity, WorkoverAction
from bores.wells.factories import make_injector, make_producer, make_well_controls, make_wells
from bores.wells.hydraulics.base import (
    SurfaceSample,
    compile_connection_derivatives,
    compile_connection_fluids,
)
from bores.wells.hydraulics.beggs_and_brill import beggs_and_brill_wellbore
from bores.wells.hydraulics.gray import gray_wellbore
from bores.wells.hydraulics.hagedorn_brown import hagedorn_brown_wellbore
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
from bores.wells.hydraulics.woldesemayat_ghajar import woldesemayat_ghajar_wellbore
from bores.wells.resolution.derivatives import compute_well_derivatives, get_well_derivatives
from bores.wells.resolution.engine import resolve_well_control
from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
from bores.wells.state import ConnectionSample, ConnectionSampleDerivatives
from bores.wells.workspace import build_wells_workspace

F64 = np.float64
STB = c.STB_TO_CUBIC_FEET
HYDRAULIC = ConnectionPressureMode.HYDRAULIC
SURFACE = SurfaceSample(
    density=40.0,
    viscosity=1.0,
    phase_densities=PhaseValues(50.0, 62.4, 0.06),
    phase_viscosities=PhaseValues(1.0, 0.5, 0.02),
    gas_liquid_surface_tension=20.0,
)
NORMAL = ([3000.0, 3010.0, 3020.0], [0.2, 0.25, 0.3], [0.1, 0.05, 0.12])
CROSSFLOW = ([1500.0, 2500.0, 3500.0], [0.2, 0.25, 0.3], [0.1, 0.05, 0.12])


@pytest.fixture(autouse=True)
def double_precision():
    with with_precision(F64):
        yield


def get_properties(pressure, water_saturation, gas_saturation):
    """Fluid properties with their exact derivatives, for one cell."""
    so = 1.0 - water_saturation - gas_saturation
    mu_o, d_mu_o = 1.0 + 1e-4 * (pressure - 3000.0), 1e-4
    mu_g, d_mu_g = 0.02 * (1.0 + 2e-4 * (pressure - 3000.0)), 0.02 * 2e-4
    derivatives = ConnectionSampleDerivatives(
        oil_mobility_pressure=-(so**2) * d_mu_o / mu_o**2,
        oil_mobility_water_saturation=-2.0 * so / mu_o,
        oil_mobility_gas_saturation=-2.0 * so / mu_o,
        water_mobility_pressure=0.0,
        water_mobility_water_saturation=2.0 * water_saturation / 0.5,
        water_mobility_gas_saturation=0.0,
        gas_mobility_pressure=-(gas_saturation**2) * d_mu_g / mu_g**2,
        gas_mobility_water_saturation=0.0,
        gas_mobility_gas_saturation=2.0 * gas_saturation / mu_g,
        oil_fvf_pressure=-1e-5,
        water_fvf_pressure=-1e-6,
        gas_fvf_pressure=-0.01 * 3000.0 / pressure**2,
        solution_gas_oil_ratio_pressure=0.05,
        vaporized_oil_gas_ratio_pressure=1e-10,
    )
    values = {
        "mobilities": PhaseValues(
            so**2 / mu_o, water_saturation**2 / 0.5, gas_saturation**2 / mu_g
        ),
        "fvfs": PhaseValues(
            1.2 - 1e-5 * (pressure - 3000.0),
            1.0 - 1e-6 * (pressure - 3000.0),
            0.01 * 3000.0 / pressure,
        ),
        "rs": 300.0 + 0.05 * (pressure - 3000.0),
        "rv": 1e-7 + 1e-10 * (pressure - 3000.0),
    }
    return values, derivatives


def make_samples(pressures, water_saturations, gas_saturations):
    samples = []
    for index, state in enumerate(zip(pressures, water_saturations, gas_saturations, strict=True)):
        values, derivatives = get_properties(*state)
        samples.append(
            ConnectionSample(
                cell_index=index,
                pressure=state[0],
                temperature=200.0,
                phase_saturations=PhaseValues(1.0 - state[1] - state[2], state[1], state[2]),
                phase_mobilities=values["mobilities"],
                phase_densities=PhaseValues(50.0 + index, 62.4, 5.0),
                phase_viscosities=PhaseValues(1.0, 0.5, 0.02),
                gas_liquid_surface_tension=20.0,
                phase_fvfs=values["fvfs"],
                solution_gas_oil_ratio=values["rs"],
                vaporized_oil_gas_ratio=values["rv"],
                derivatives=derivatives,
            )
        )
    return samples


class Well:
    """One compiled well that can be resolved at chosen cell conditions."""

    def __init__(
        self,
        *,
        kind="producer",
        n=3,
        bhp=1000.0,
        mode=ConnectionPressureMode.UNIFORM_BHP,
        tubing=None,
        surface_sample=None,
        allows_crossflow=True,
        d_factor=None,
        limits=None,
        injected=FluidPhase.WATER,
    ):
        grid = make_cartesian_grid(nx=1, ny=1, nz=n, dx=50.0, dy=50.0, dz=50.0)
        options = dict(
            surface_location=(25.0, 25.0),
            perforation_depths=(0.0, 50.0 * n),
            reference_depth=25.0 * n,
            target_bhp=bhp,
            allows_crossflow=allows_crossflow,
            limits=limits,
        )
        if kind == "producer":
            well, control = make_producer("W", **options)
        else:
            well, control = make_injector("W", injected_phase=injected, **options)
        self.system = compile_well_system(
            make_wells(well),
            make_well_controls(W=control),
            grid,
            permeabilities={orientation: np.full(n, 100.0) for orientation in Orientation},
        )
        if d_factor is not None:
            self.system.d_factors[0] = d_factor
        self.n = n
        self.tubing = tubing
        self.surface_sample = surface_sample
        self.spec = WellControlSpec(connection_pressure_mode=mode)
        self.workspace = build_wells_workspace(
            n_wells=1,
            n_connections=n,
            well_offsets=self.system.perforations.well_offsets,
            dtype=F64,
        )

    def resolve(
        self, pressures, water_saturations, gas_saturations, *, bhp=None, derivatives=False
    ):
        if bhp is not None:
            self.system.controls.target_bhps[0] = bhp
        samples = make_samples(pressures, water_saturations, gas_saturations)
        fluids = compile_connection_fluids(connection_samples=samples, dtype=F64)
        resolve_well_control(
            compiled_system=self.system,
            well_row=0,
            tubing_performance=self.tubing,
            connection_fluids=fluids,
            workspace=self.workspace,
            spec=self.spec,
            surface_sample=self.surface_sample,
        )
        if derivatives:
            compute_well_derivatives(
                compiled_system=self.system,
                well_row=0,
                tubing_performance=self.tubing,
                connection_fluids=fluids,
                connection_derivatives=compile_connection_derivatives(
                    connection_samples=samples, dtype=F64
                ),
                workspace=self.workspace,
                spec=self.spec,
                surface_sample=self.surface_sample,
            )

    def get_rates(self, pressures):
        """Connection reservoir rates and surface component rates, from the last resolve."""
        workspace = self.workspace
        rates = np.stack(
            [
                workspace.connection_oil_rates,
                workspace.connection_water_rates,
                workspace.connection_gas_rates,
            ],
            axis=1,
        ).astype(F64)
        is_injector = bool(self.system.controls.well_kinds[0] == 1)
        surface = np.zeros_like(rates)
        for index in range(self.n):
            values, _ = get_properties(pressures[index], 0.2, 0.1)
            oil_volume, water_volume, gas_volume = values["fvfs"]
            carries = (not is_injector) or bool((rates[index] < 0.0).any())
            oil, water, gas = rates[index]
            surface[index] = [
                oil / (oil_volume * STB) + carries * values["rv"] * gas / gas_volume,
                water / (water_volume * STB),
                gas / gas_volume + carries * values["rs"] * oil / (oil_volume * STB),
            ]
        return rates, surface


def assert_derivatives_match(well, state, bhp, *, tolerance=2e-4):
    """Compares every derivative of `well` at `state` with a central finite difference."""
    pressures, _water_saturations, _gas_saturations = state
    well.resolve(*state, bhp=bhp, derivatives=True)
    derivatives = get_well_derivatives(
        compiled_system=well.system, workspace=well.workspace, well_row=0
    )
    steps = (0.5, 1e-4, 1e-4)
    for cell in range(well.n):
        for unknown in range(3):

            def solve(shift):
                shifted = [np.array(values, dtype=F64) for values in state]
                shifted[unknown][cell] += shift
                well.resolve(*shifted, bhp=bhp)
                return well.get_rates(shifted[0])

            rates_up, surface_up = solve(steps[unknown])
            rates_down, surface_down = solve(-steps[unknown])
            difference = (rates_up - rates_down) / (2.0 * steps[unknown])
            surface_difference = (surface_up - surface_down) / (2.0 * steps[unknown])
            for connection in range(well.n):
                for phase in range(3):
                    row = 3 * connection + phase
                    column = 3 * cell + unknown
                    np.testing.assert_allclose(
                        derivatives.rates_cell[row, column],
                        difference[connection, phase],
                        rtol=tolerance,
                        atol=1e-5 * np.abs(difference).max(),
                    )
                    np.testing.assert_allclose(
                        derivatives.surface_rates_cell[row, column],
                        surface_difference[connection, phase],
                        rtol=tolerance,
                        atol=1e-5 * np.abs(surface_difference).max(),
                    )

    step = 1e-5 * bhp
    well.resolve(*state, bhp=bhp + step)
    rates_up, surface_up = well.get_rates(pressures)
    well.resolve(*state, bhp=bhp - step)
    rates_down, surface_down = well.get_rates(pressures)
    difference = (rates_up - rates_down) / (2.0 * step)
    surface_difference = (surface_up - surface_down) / (2.0 * step)
    np.testing.assert_allclose(
        derivatives.rates_bhp, difference, rtol=tolerance, atol=1e-5 * np.abs(difference).max()
    )
    np.testing.assert_allclose(
        derivatives.surface_rates_bhp,
        surface_difference,
        rtol=tolerance,
        atol=1e-5 * np.abs(surface_difference).max(),
    )
    np.testing.assert_allclose(
        derivatives.well_rates_bhp, difference.sum(axis=0), rtol=tolerance, atol=1e-6
    )
    np.testing.assert_allclose(
        derivatives.well_surface_rates_bhp,
        surface_difference.sum(axis=0),
        rtol=tolerance,
        atol=1e-6,
    )


WELLBORES = {
    "homogeneous": homogeneous_wellbore,
    "beggs_brill": beggs_and_brill_wellbore,
    "hagedorn_brown": hagedorn_brown_wellbore,
    "gray": gray_wellbore,
    "woldesemayat_ghajar": woldesemayat_ghajar_wellbore,
}


@pytest.mark.parametrize("name", sorted(WELLBORES))
def test_producer_derivatives_include_the_wellbore_coupling_of_every_correlation(name):
    wellbore = WELLBORES[name](tubing_inner_diameter=0.25)
    well = Well(kind="producer", bhp=1500.0, mode=HYDRAULIC, tubing=wellbore)
    assert_derivatives_match(well, NORMAL, 1500.0, tolerance=5e-4)


@pytest.mark.parametrize(
    "scenario",
    [
        dict(kind="producer", bhp=1000.0),
        dict(kind="producer", bhp=1500.0, mode=ConnectionPressureMode.HYDROSTATIC),
        dict(kind="injector", bhp=4000.0),
        dict(kind="injector", bhp=4000.0, injected=FluidPhase.GAS),
        dict(kind="producer", bhp=1500.0, d_factor=2e-6),
    ],
    ids=["producer", "hydrostatic", "water_injector", "gas_injector", "non_darcy"],
)
def test_derivatives_match_finite_differences(scenario):
    well = Well(**scenario)
    assert_derivatives_match(well, NORMAL, scenario["bhp"])


@pytest.mark.parametrize("kind,bhp", [("producer", 2000.0), ("injector", 3000.0)])
@pytest.mark.parametrize("hydraulic", [False, True])
def test_crossflowing_connections_have_derivatives(kind, bhp, hydraulic):
    options = dict(kind=kind, bhp=bhp)
    if hydraulic:
        options.update(mode=HYDRAULIC, tubing=homogeneous_wellbore(tubing_inner_diameter=0.25))
    assert_derivatives_match(Well(**options), CROSSFLOW, bhp)


def test_producer_without_crossflow_has_no_derivative_on_a_reversed_connection():
    well = Well(kind="producer", bhp=2000.0, allows_crossflow=False)
    assert_derivatives_match(well, CROSSFLOW, 2000.0)
    derivatives = get_well_derivatives(
        compiled_system=well.system, workspace=well.workspace, well_row=0
    )
    assert not derivatives.rates_cell[0:3].any()


@pytest.mark.parametrize(
    "kind,bhp,density", [("producer", 1500.0, 40.0), ("injector", 4000.0, 62.4)]
)
def test_tubing_head_pressure_derivatives_match_finite_differences(kind, bhp, density):
    sample = SurfaceSample(
        density=density,
        viscosity=1.0,
        phase_densities=PhaseValues(50.0, 62.4, 0.06),
        phase_viscosities=PhaseValues(1.0, 0.5, 0.02),
        gas_liquid_surface_tension=20.0,
    )
    well = Well(
        kind=kind,
        bhp=bhp,
        mode=HYDRAULIC,
        tubing=homogeneous_wellbore(tubing_inner_diameter=0.25),
        surface_sample=sample,
    )
    well.resolve(*NORMAL, bhp=bhp, derivatives=True)
    derivatives = get_well_derivatives(
        compiled_system=well.system, workspace=well.workspace, well_row=0
    )

    def get_thp(state=NORMAL, bhp=bhp):
        well.resolve(*state, bhp=bhp)
        return float(well.workspace.thps[0])

    step = 1e-5 * bhp
    np.testing.assert_allclose(
        derivatives.thp_bhp,
        (get_thp(bhp=bhp + step) - get_thp(bhp=bhp - step)) / (2 * step),
        rtol=1e-4,
    )
    steps = (0.5, 1e-4, 1e-4)
    for cell in range(3):
        for unknown in range(3):

            def at(shift):
                shifted = [np.array(values, dtype=F64) for values in NORMAL]
                shifted[unknown][cell] += shift
                return get_thp(shifted)

            expected = (at(steps[unknown]) - at(-steps[unknown])) / (2 * steps[unknown])
            np.testing.assert_allclose(
                derivatives.thp_cell[cell, unknown], expected, rtol=1e-3, atol=1e-6
            )


def test_well_totals_are_the_sum_of_its_connections():
    well = Well(kind="producer", bhp=1500.0)
    well.resolve(*NORMAL, derivatives=True)
    derivatives = get_well_derivatives(
        compiled_system=well.system, workspace=well.workspace, well_row=0
    )
    expected = derivatives.rates_cell.reshape(3, 3, 3, 3).sum(axis=0)
    np.testing.assert_allclose(derivatives.well_rates_cell, expected.transpose(1, 0, 2))
    np.testing.assert_allclose(
        derivatives.well_surface_rates_cell,
        derivatives.surface_rates_cell.reshape(3, 3, 3, 3).sum(axis=0).transpose(1, 0, 2),
    )


def test_a_well_shut_in_by_an_economic_limit_has_zero_derivatives():
    limit = EconomicLimit(
        quantity=EconomicQuantity.WATER_CUT, max_value=1e-4, workover_action=WorkoverAction.WELL
    )
    well = Well(kind="producer", bhp=1000.0, limits=[limit])
    well.resolve(*NORMAL, derivatives=True)
    derivatives = get_well_derivatives(
        compiled_system=well.system, workspace=well.workspace, well_row=0
    )
    assert well.workspace.economic_shutins[0] == 1
    assert not derivatives.rates_cell.any()
    assert not derivatives.surface_rates_bhp.any()


def test_a_workspace_without_room_for_derivatives_is_rejected():
    well = Well(kind="producer")
    well.workspace = build_wells_workspace(n_wells=1, n_connections=3, dtype=F64)
    with pytest.raises(ValidationError, match="no room"):
        well.resolve(*NORMAL, derivatives=True)


def test_derivatives_for_the_wrong_number_of_connections_are_rejected():
    well = Well(kind="producer")
    well.resolve(*NORMAL)
    samples = make_samples(*NORMAL)
    with pytest.raises(ValidationError, match="must match"):
        compute_well_derivatives(
            compiled_system=well.system,
            well_row=0,
            tubing_performance=None,
            connection_fluids=compile_connection_fluids(connection_samples=samples, dtype=F64),
            connection_derivatives=compile_connection_derivatives(
                connection_samples=samples[:2], dtype=F64
            ),
            workspace=well.workspace,
            spec=well.spec,
        )


def test_samples_without_derivatives_give_nan_derivatives():
    well = Well(kind="producer")
    samples = [sample._replace(derivatives=None) for sample in make_samples(*NORMAL)]
    well.resolve(*NORMAL)
    compute_well_derivatives(
        compiled_system=well.system,
        well_row=0,
        tubing_performance=None,
        connection_fluids=compile_connection_fluids(connection_samples=samples, dtype=F64),
        connection_derivatives=compile_connection_derivatives(
            connection_samples=samples, dtype=F64
        ),
        workspace=well.workspace,
        spec=well.spec,
    )
    derivatives = get_well_derivatives(
        compiled_system=well.system, workspace=well.workspace, well_row=0
    )
    assert np.isnan(derivatives.rates_cell).all()


def test_injector_crossflow_reports_the_formation_fluid_it_produces():
    well = Well(kind="injector", bhp=3000.0)
    well.resolve(*CROSSFLOW)
    workspace = well.workspace
    assert workspace.connection_oil_rates[2] < 0.0
    np.testing.assert_allclose(workspace.oil_rates[0], workspace.connection_oil_rates.sum())
    np.testing.assert_allclose(workspace.gas_rates[0], workspace.connection_gas_rates.sum())
    _, surface = well.get_rates(CROSSFLOW[0])
    np.testing.assert_allclose(
        [
            workspace.surface_oil_rates[0],
            workspace.surface_water_rates[0],
            workspace.surface_gas_rates[0],
        ],
        surface.sum(axis=0),
        rtol=1e-9,
    )


@pytest.mark.parametrize("action", [WorkoverAction.NONE, WorkoverAction.WELL])
def test_a_well_economic_limit_with_no_workover_action_does_not_shut_the_well(action):
    limit = EconomicLimit(
        quantity=EconomicQuantity.WATER_CUT, max_value=0.2, workover_action=action
    )
    well = Well(kind="producer", bhp=1000.0, limits=[limit])
    well.resolve(*NORMAL)
    shut_in = action == WorkoverAction.WELL
    assert bool(well.workspace.economic_shutins[0]) is shut_in
    assert bool(well.workspace.surface_oil_rates[0] == 0.0) is shut_in
