import re
from pathlib import Path

import numpy as np
import pytest

import bores.initialization as initialization
from bores.blackoil.satfunc.capillary_pressure.models import (
    BrooksCoreyCapillaryPressureTable,
    Wettability,
)
from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.initialization import (
    get_hydrostatic_gradient_factor,
    initialize_horizontal_subdivision_equilibrium,
    initialize_reservoir_state,
    integrate_hydrostatic_pressure,
    invert_capillary_pressure_curves,
)
from bores.reservoir.equilibrium import Equilibrium
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem

SPE1_DECK = Path(__file__).resolve().parents[2] / "data" / "SPE1CASE1.DATA"

WATER_PRESSURE_BY_SATURATION = {
    0.12: 12, 0.18: 9, 0.24: 7, 0.3: 5.5, 0.36: 4.5, 0.42: 3.6, 0.48: 2.8,
    0.54: 2.1, 0.6: 1.5, 0.66: 1.0, 0.72: 0.6, 0.78: 0.3, 0.84: 0.15, 0.91: 0.05, 1.0: 0.0,
}  # fmt: skip


def make_capillary_deck(path: Path) -> Path:
    """SPE1 with real capillary pressure curves and a gas cap above and aquifer below the oil."""
    lines = SPE1_DECK.read_text().split("\n")
    out: list[str] = []
    mode = None
    for line in lines:
        stripped = line.strip()
        if stripped == "SWOF":
            mode = "water"
        elif stripped == "SGOF":
            mode = "gas"
        elif stripped.startswith("EQUIL"):
            mode = "equil"
        elif (
            stripped
            and not stripped.startswith("--")
            and mode in ("water", "gas")
            and re.match(r"^[0-9.]", stripped)
        ):
            ends = stripped.endswith("/")
            tokens = stripped.replace("/", " ").split()
            if len(tokens) >= 4:
                saturation = float(tokens[0])
                tokens[3] = (
                    str(WATER_PRESSURE_BY_SATURATION.get(round(saturation, 2), 0))
                    if mode == "water"
                    else str(round(saturation * 10, 3))
                )
                line = "\t".join(tokens[:4]) + (" /" if ends else "")
            if ends:
                mode = None
        elif mode == "equil" and stripped.startswith("8400"):
            line = "8365 4800 8380 0 8350 0 1 0 0 /"
            mode = None
        out.append(line)
    path.write_text("\n".join(out))
    return path


def load(path: Path):
    deck_file = DeckFile(str(path), encoding="utf-8", unit_system=UnitSystem.FIELD)
    case = SimulationCase.from_deck(deck_file, temperature=200.0)
    return deck_file, case


@pytest.fixture(scope="module")
def spe1():
    return load(SPE1_DECK)


@pytest.fixture(scope="module")
def spe1_capillary(tmp_path_factory):
    return load(make_capillary_deck(tmp_path_factory.mktemp("deck") / "spe1_pc.DATA"))


class TestInvertCapillaryPressureCurves:
    saturation = np.tile(np.linspace(0.12, 1.0, 11), (3, 1))

    def invert(self, pressure_grid, target, *, falls, prefer_highest):
        return invert_capillary_pressure_curves(
            saturation_grid=self.saturation,
            capillary_pressure_grid=pressure_grid,
            target_capillary_pressure=np.asarray(target, dtype=float),
            pressure_falls_with_saturation=falls,
            prefer_highest_pressure_end=prefer_highest,
        )

    def test_flat_curve_is_a_sharp_contact(self):
        flat = np.zeros((3, 11))
        saturation, pressure = self.invert(flat, [5.0, -5.0, 0.0], falls=True, prefer_highest=True)
        # Above the contact is connate, below it is water-filled, on it is oil-bearing
        np.testing.assert_allclose(saturation, [0.12, 1.0, 0.12])
        np.testing.assert_allclose(pressure, 0.0)

    def test_interior_target_is_reproduced_for_either_direction(self):
        rising = np.tile(np.linspace(0.0, 10.0, 11), (3, 1))
        for grid, falls in ((rising, False), (rising[:, ::-1], True)):
            saturation, pressure = self.invert(
                grid, [2.5, 5.0, 7.5], falls=falls, prefer_highest=False
            )
            reproduced = np.array([
                np.interp(s, self.saturation[0], g) for s, g in zip(saturation, grid, strict=False)
            ])
            np.testing.assert_allclose(reproduced, [2.5, 5.0, 7.5], atol=1e-9)
            np.testing.assert_allclose(pressure, [2.5, 5.0, 7.5])

    def test_unreachable_target_is_clamped(self):
        falling = np.tile(np.linspace(10.0, 0.0, 11), (3, 1))
        saturation, pressure = self.invert(
            falling, [50.0, -50.0, 5.0], falls=True, prefer_highest=True
        )
        assert saturation[0] == pytest.approx(0.12)
        assert saturation[1] == pytest.approx(1.0)
        np.testing.assert_allclose(pressure, [10.0, 0.0, 5.0])


class TestCapillaryEquilibrium:
    def test_zero_capillary_pressure_gives_connate_water_in_the_oil_zone(self, spe1):
        _, case = spe1
        state = case.initial_state
        np.testing.assert_allclose(state.water_saturation, 0.12, atol=1e-6)
        np.testing.assert_allclose(state.gas_saturation, 0.0, atol=1e-6)
        assert np.all(state.solution_gor > 0.0)

    def test_gas_cap_oil_zone_and_aquifer(self, spe1_capillary):
        _, case = spe1_capillary
        state = case.initial_state
        depth = case.model.reservoir.depth
        gas_cap, oil_zone, aquifer = depth < 8350, (depth > 8350) & (depth < 8380), depth > 8380
        assert state.gas_saturation[gas_cap].min() > 0.1  # was exactly zero before
        np.testing.assert_allclose(state.gas_saturation[oil_zone], 0.0, atol=1e-6)
        np.testing.assert_allclose(state.water_saturation[aquifer], 1.0, atol=1e-6)

    def test_pressure_increases_with_depth(self, spe1_capillary):
        _, case = spe1_capillary
        order = np.argsort(case.model.reservoir.depth, kind="stable")
        assert np.all(np.diff(case.initial_state.pressure[order]) >= -1e-3)


class TestBrooksCoreyDtype:
    @pytest.mark.parametrize("wettability", list(Wettability))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_array_evaluation_keeps_the_input_dtype(self, wettability, dtype):
        table = BrooksCoreyCapillaryPressureTable(
            unit_system=UnitSystem.FIELD, wettability=wettability
        )
        sw = np.linspace(0.2, 0.9, 7).astype(dtype)
        full = lambda value: np.full(7, value, dtype=dtype)  # noqa: E731
        result = table.evaluate(
            sw, 1 - sw, np.zeros(7, dtype=dtype),
            irreducible_water_saturation=full(0.2), residual_oil_saturation_water=full(0.1),
            residual_oil_saturation_gas=full(0.1), residual_gas_saturation=full(0.05),
        )  # fmt: skip
        assert result["oil_water"].dtype == dtype
        assert result["gas_oil"].dtype == dtype


class TestSubdivisionAveraging:
    def test_cell_straddling_the_goc_keeps_the_dissolved_gas_of_its_oil(self, spe1_capillary):
        deck_file, case = spe1_capillary
        equilibrium = Equilibrium.from_deck(deck_file)
        region = equilibrium.region(1)
        pvt_tables = case.model.fluid.pvt.region(1).tables
        n = 2
        arrays = initialize_horizontal_subdivision_equilibrium(
            region=region,
            cell_top=np.array([8340.0, 8360.0]),
            cell_bottom=np.array([8360.0, 8370.0]),  # the first cell straddles the GOC at 8350
            n_subdivisions=10,
            connate_water_saturation=np.full(n, 0.12),
            oil_table=pvt_tables.oil,
            gas_table=pvt_tables.gas,
            water_table=pvt_tables.water,
            rsvd_table=equilibrium.rsvd_tables.get(region.rsvd_table),
            rvvd_table=None,
            temperature=np.full(n, 200.0),
            unit_system=UnitSystem.FIELD,
            capillary_pressure=None,
            residual_oil_saturation_water=None,
            residual_oil_saturation_gas=None,
            residual_gas_saturation=None,
            dtype=np.float64,
        )  # fmt: skip
        assert 0.0 < arrays.gas_saturation[0] < 0.88
        assert arrays.solution_gor[0] == pytest.approx(arrays.solution_gor[1])
        assert arrays.oil_bubble_point_pressure[0] == pytest.approx(
            arrays.oil_bubble_point_pressure[1]
        )


class TestStateConsistency:
    def test_explicit_solution_gor_gets_its_own_bubble_point(self, spe1):
        deck_file, case = spe1
        reservoir, pvt = case.model.reservoir, case.model.fluid.pvt
        with pytest.warns(UserWarning):
            state = initialize_reservoir_state(
                reservoir, pvt, deck_file=deck_file, satfunc=None, temperature=200.0,
                solution_gor=np.full(reservoir.n_cells, 600.0, dtype=np.float32),
            )  # fmt: skip
        expected = np.ravel(
            pvt.region(1).tables.oil.bubble_point_pressure(
                temperature=np.array([200.0]), solution_gor=np.array([600.0])
            )
        )[0]
        np.testing.assert_allclose(state.oil_bubble_point_pressure, expected, rtol=1e-4)

    @pytest.mark.parametrize("bad", [-0.2, 1.3, np.nan])
    def test_invalid_explicit_saturation_is_rejected(self, spe1, bad):
        deck_file, case = spe1
        reservoir, pvt = case.model.reservoir, case.model.fluid.pvt
        water = np.full(reservoir.n_cells, 0.12, dtype=np.float32)
        water[0] = bad
        with pytest.raises(ValidationError):
            initialize_reservoir_state(
                reservoir, pvt, deck_file=deck_file, satfunc=None, temperature=200.0,
                water_saturation=water,
            )  # fmt: skip


class TestIntegrateHydrostaticPressure:
    compressibility = 2.0e-6
    gradient_factor = get_hydrostatic_gradient_factor(UnitSystem.FIELD)

    def density(self, pressure, depth):
        # Density proportional to pressure, which has the exact solution `P0 * exp(k * dz)`
        return self.compressibility * np.asarray(pressure, dtype=np.float64)

    @pytest.mark.parametrize("endpoint", [8000.0, 11000.0])
    def test_matches_the_analytic_solution_in_both_directions(self, endpoint):
        depths, pressures = integrate_hydrostatic_pressure(
            self.density,
            9000.0,
            4800.0,
            endpoint,
            self.gradient_factor,
            step=1.0,
            dtype=np.float64,
        )
        assert np.all(np.diff(depths) > 0.0)
        exact = 4800.0 * np.exp(self.gradient_factor * self.compressibility * (depths - 9000.0))
        np.testing.assert_allclose(pressures, exact, rtol=1e-10)

    def test_visits_the_same_depths_as_the_step_by_step_march(self):
        reference = initialization._rk4_march(
            self.density, 9000.0, 4800.0, 8333.3, self.gradient_factor, step=1.0, dtype=np.float64
        )
        result = integrate_hydrostatic_pressure(
            self.density, 9000.0, 4800.0, 8333.3, self.gradient_factor, step=1.0, dtype=np.float64
        )
        np.testing.assert_allclose(result[0], reference[0])
        np.testing.assert_allclose(result[1], reference[1], rtol=1e-9)

    def test_falls_back_to_the_step_by_step_march_when_the_solve_does_not_settle(
        self, monkeypatch
    ):
        monkeypatch.setattr(initialization, "COLLOCATION_MAX_ITERATIONS", 1)
        reference = initialization._rk4_march(
            self.density, 9000.0, 4800.0, 8000.0, self.gradient_factor, step=1.0, dtype=np.float64
        )
        result = integrate_hydrostatic_pressure(
            self.density, 9000.0, 4800.0, 8000.0, self.gradient_factor, step=1.0, dtype=np.float64
        )
        np.testing.assert_array_equal(result[1], reference[1])


class TestSaturatedDissolvedRatiosOutsideTheOilZone:
    def test_gas_cap_and_aquifer_cells_carry_the_oil_properties_of_their_depth(
        self, spe1_capillary
    ):
        _, case = spe1_capillary
        state = case.initial_state
        depth = case.model.reservoir.depth
        oil_zone = (depth > 8350) & (depth < 8380)
        for outside in (depth < 8350, depth > 8380):
            assert np.all(state.solution_gor[outside] > 0.0)
            assert np.all(state.oil_bubble_point_pressure[outside] > 0.0)
        np.testing.assert_allclose(state.solution_gor, state.solution_gor[oil_zone][0], rtol=1e-5)

    def test_sharp_contact_gas_cap_carries_them_too(self, spe1_capillary):
        deck_file, case = spe1_capillary
        reservoir, pvt = case.model.reservoir, case.model.fluid.pvt
        state = initialize_reservoir_state(
            reservoir, pvt, deck_file=deck_file, satfunc=None, temperature=200.0
        )
        gas_cap = reservoir.depth < 8350
        assert np.all(state.gas_saturation[gas_cap] > 0.8)
        assert np.all(state.solution_gor[gas_cap] > 0.0)


class TestTiltedSubdivision:
    def test_a_flat_grid_gives_the_same_answer_as_horizontal_subdivision(self, spe1_capillary):
        deck_file, case = spe1_capillary
        reservoir = case.model.reservoir
        equilibrium = Equilibrium.from_deck(deck_file)
        region = equilibrium.region(1)
        tables = case.model.fluid.pvt.region(1).tables
        n = reservoir.n_cells
        shared = dict(
            region=region,
            connate_water_saturation=np.full(n, 0.12),
            oil_table=tables.oil,
            gas_table=tables.gas,
            water_table=tables.water,
            rsvd_table=equilibrium.rsvd_tables.get(region.rsvd_table),
            rvvd_table=None,
            temperature=np.full(n, 200.0),
            unit_system=UnitSystem.FIELD,
            capillary_pressure=None,
            residual_oil_saturation_water=None,
            residual_oil_saturation_gas=None,
            residual_gas_saturation=None,
            dtype=np.float64,
            n_subdivisions=6,
        )  # fmt: skip
        grid = reservoir.grid
        tilted = initialization.initialize_tilted_subdivision_equilibrium(
            grid=grid, cell_indices=np.arange(n), **shared
        )
        horizontal = initialize_horizontal_subdivision_equilibrium(
            cell_top=grid.cell_min_xyz[:, 2], cell_bottom=grid.cell_max_xyz[:, 2], **shared
        )
        for name in tilted._fields:
            np.testing.assert_allclose(
                getattr(tilted, name), getattr(horizontal, name), rtol=1e-6, atol=1e-6
            )
