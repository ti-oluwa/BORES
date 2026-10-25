import types

import numpy as np
import pytest

from bores.constants import c
from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.types import UnitSystem
from bores.units import get_conversion_factors
from bores.wells import summary as well_summary
from bores.wells.base import WellType
from bores.wells.deck import (
    get_fluid_in_place_regions,
    load_producer_control_from_record,
    load_summary_items,
)
from bores.wells.hydraulics.base import compile_connection_fluids
from bores.wells.hydraulics.vfp import (
    VFPData,
    VFPFlowType,
    VFPGasFractionType,
    VFPTable,
    VFPTables,
    VFPWaterFractionType,
    load_vfp_data,
    load_vfp_table,
    load_vfp_tables,
)

VFPPROD_BODY = """VFPPROD
 3 2000 LIQ {wfr} {gfr} THP GRAT {units} BHP /
 100 500 /
 20 40 /
 0 1 /
 {gor0} {gor1} /
 0 1000 /
""" + "".join(
    f" {t} {w} {g} {a}  {10 * t + w + g + a} {100 + 10 * t + w + g + a} /\n"
    for t in (1, 2)
    for w in (1, 2)
    for g in (1, 2)
    for a in (1, 2)
)

WCONPROD_AND_WLIFT = """WCONPROD
 'P1' OPEN LRAT 1* 1* 1* 500 1* 80 1* 3 250 /
/
WLIFT
 'P1' 100 OIL 2 400 /
/
"""


def deck_from(tmp_path, text, unit="FIELD"):
    path = tmp_path / "case.DATA"
    path.write_text(f"RUNSPEC\n{unit}\nSCHEDULE\n{text}")
    return DeckFile.from_path(path) if hasattr(DeckFile, "from_path") else DeckFile(path)


def producer_table(tmp_path, *, wfr="WCT", gfr="GOR", units="FIELD", unit="FIELD", flo="LIQ"):
    body = VFPPROD_BODY.format(wfr=wfr, gfr=gfr, units=units, gor0=0.1, gor1=0.5).replace(
        "LIQ", flo, 1
    )
    return deck_from(tmp_path, body, unit=unit)


def test_wconprod_reads_alq_and_wlift_reads_every_item(tmp_path):
    deck = deck_from(tmp_path, WCONPROD_AND_WLIFT)
    record = deck.get("WCONPROD")[0]
    assert record["alq"] == pytest.approx(250.0) and record["vfp_table"] == 3
    lift = deck.get("WLIFT")[0]
    assert (lift["well"], lift["trigger_limit"], lift["trigger_phase"]) == ("P1", 100.0, "OIL")
    assert (lift["new_vfp_table"], lift["new_alq"], lift["alq_shift"]) == (2, 400.0, 1e20)


def test_producer_control_carries_the_vfp_table_and_alq(tmp_path):
    record = deck_from(tmp_path, WCONPROD_AND_WLIFT).get("WCONPROD")[0]
    control = load_producer_control_from_record(record, UnitSystem.FIELD)
    assert control.vfp_table == 3 and control.artificial_lift_quantity == pytest.approx(250.0)


def test_producer_control_defaults_to_no_table_and_no_lift(tmp_path):
    deck = deck_from(tmp_path, "WCONPROD\n 'P1' OPEN LRAT 1* 1* 1* 500 1* 80 /\n/\n")
    control = load_producer_control_from_record(deck.get("WCONPROD")[0], UnitSystem.FIELD)
    assert control.vfp_table is None and control.artificial_lift_quantity == pytest.approx(0.0)


def test_negative_alq_is_rejected(tmp_path):
    record = dict(deck_from(tmp_path, WCONPROD_AND_WLIFT).get("WCONPROD")[0])
    record["alq"] = -1.0
    with pytest.raises(ValidationError):
        load_producer_control_from_record(record, UnitSystem.FIELD)


def test_vfpprod_and_vfpinj_tables_parse_with_the_right_layout(tmp_path):
    deck = deck_from(
        tmp_path,
        VFPPROD_BODY.format(wfr="WCT", gfr="GOR", units="FIELD", gor0=0.1, gor1=0.5)
        + "VFPINJ\n 5 2000 WAT THP FIELD BHP /\n 100 200 300 /\n 20 40 /\n 1  50 60 70 /\n 2  80 90 100 /\n",
    )
    producer = deck.get("VFPPROD")[0]
    assert producer["bhps"].shape == (2, 2, 2, 2, 2)
    assert (
        producer["bhps"][0, 1, 0, 1, 1] == 10 * 2 + 1 + 2 + 2
    )  # row `2 1 2 2` -> first flow value
    injector = deck.get("VFPINJ")[0]
    assert injector["bhps"].shape == (3, 2) and injector["bhps"][2, 1] == pytest.approx(100.0)


def test_vfp_table_with_missing_rows_is_rejected(tmp_path):
    text = VFPPROD_BODY.format(wfr="WCT", gfr="GOR", units="FIELD", gor0=0.1, gor1=0.5)
    truncated = "\n".join(text.splitlines()[:-1]) + "\n"
    with pytest.raises(Exception, match="cover every combination"):
        deck_from(tmp_path, truncated).get("VFPPROD")


def test_field_vfp_table_rescales_gas_oil_ratio_and_queries_exactly(tmp_path):
    deck = producer_table(tmp_path)
    data = load_vfp_data(deck, table_number=3)
    assert data.unit_system == UnitSystem.FIELD
    assert np.allclose(data.gas_fractions, [100.0, 500.0])  # Mscf/STB -> scf/STB
    table = VFPTable.from_deck(deck, table_number=3)
    bhp = table.query(
        flow_rate=500.0,
        thp=40.0,
        water_fraction=1.0,
        gas_fraction=500.0,
        artificial_lift_quantity=1000.0,
    )
    assert float(bhp) == pytest.approx(100 + 10 * 2 + 2 + 2 + 2)


def test_water_oil_ratio_axis_is_kept_as_a_water_oil_ratio(tmp_path):
    data = load_vfp_data(producer_table(tmp_path, wfr="WOR"), table_number=3)
    assert data.water_fraction_type == VFPWaterFractionType.WATER_OIL_RATIO
    assert np.allclose(data.water_fractions, [0.0, 1.0])


def test_metric_tables_keep_their_gas_oil_ratio(tmp_path):
    deck = producer_table(tmp_path, units="METRIC", unit="METRIC")
    assert np.allclose(load_vfp_data(deck, table_number=3).gas_fractions, [0.1, 0.5])


@pytest.mark.parametrize("flow", ["WG", "TM"])
def test_flow_definitions_with_no_single_rate_are_rejected_clearly(tmp_path, flow):
    with pytest.raises(ValidationError, match="flow axis"):
        load_vfp_data(producer_table(tmp_path, flo=flow), table_number=3)


def test_table_keeps_the_definitions_its_deck_declares(tmp_path):
    deck = producer_table(tmp_path, wfr="WOR", gfr="GLR", flo="OIL")
    data = load_vfp_data(deck, table_number=3)
    assert data.flow_type == VFPFlowType.OIL
    assert data.water_fraction_type == VFPWaterFractionType.WATER_OIL_RATIO
    assert data.gas_fraction_type == VFPGasFractionType.GAS_LIQUID_RATIO
    assert np.allclose(data.flow_rates, [100.0, 500.0])
    assert np.allclose(data.water_fractions, [0.0, 1.0])
    assert np.allclose(data.gas_fractions, [100.0, 500.0])


def test_field_gas_axes_are_rescaled_per_their_definition(tmp_path):
    gas_flow = load_vfp_data(
        producer_table(tmp_path, flo="GAS", wfr="WGR", gfr="OGR"), table_number=3
    )
    assert np.allclose(gas_flow.flow_rates, [100_000.0, 500_000.0])  # Mscf/d -> scf/d
    assert np.allclose(gas_flow.water_fractions, [0.0, 0.001])  # STB/Mscf -> STB/scf
    assert np.allclose(gas_flow.gas_fractions, [0.0001, 0.0005])  # STB/Mscf -> STB/scf
    liquid = load_vfp_data(producer_table(tmp_path, gfr="GLR"), table_number=3)
    assert np.allclose(liquid.gas_fractions, [100.0, 500.0])  # Mscf/STB -> scf/STB


def test_metric_tables_are_not_rescaled(tmp_path):
    deck = producer_table(tmp_path, units="METRIC", unit="METRIC", flo="GAS", wfr="WGR", gfr="OGR")
    data = load_vfp_data(deck, table_number=3)
    assert np.allclose(data.flow_rates, [100.0, 500.0])
    assert np.allclose(data.water_fractions, [0.0, 1.0])
    assert np.allclose(data.gas_fractions, [0.1, 0.5])


@pytest.mark.parametrize(
    "flow, expected",
    [("OIL", VFPFlowType.OIL), ("WAT", VFPFlowType.WATER), ("GAS", VFPFlowType.GAS)],
)
def test_injector_table_flow_type_is_the_injected_phase(tmp_path, flow, expected):
    deck = deck_from(
        tmp_path,
        f"VFPINJ\n 5 2000 {flow} THP FIELD BHP /\n 100 200 /\n 20 40 /\n 1  50 60 /\n 2  80 90 /\n",
    )
    assert load_vfp_data(deck, table_number=5).flow_type == expected


def lookup_table(*, flow_type, water_type, gas_type):
    return VFPTable(
        VFPData(
            table_number=1,
            well_type=WellType.PRODUCER,
            datum_depth=2000.0,
            flow_type=flow_type,
            water_fraction_type=water_type,
            gas_fraction_type=gas_type,
            flow_rates=np.array([1.0, 10.0]),
            thps=np.array([10.0, 20.0]),
            water_fractions=np.array([0.0, 2.0]),
            gas_fractions=np.array([0.0, 7.0]),
            bhps=np.zeros((2, 2, 2, 2, 1)),
        )
    )


@pytest.mark.parametrize(
    "flow_type, water_type, gas_type, expected",
    [
        (
            VFPFlowType.OIL,
            VFPWaterFractionType.WATER_CUT,
            VFPGasFractionType.GAS_OIL_RATIO,
            (100.0, 0.2, 50.0),
        ),
        (
            VFPFlowType.LIQUID,
            VFPWaterFractionType.WATER_OIL_RATIO,
            VFPGasFractionType.GAS_LIQUID_RATIO,
            (125.0, 0.25, 40.0),
        ),
        (
            VFPFlowType.GAS,
            VFPWaterFractionType.WATER_GAS_RATIO,
            VFPGasFractionType.OIL_GAS_RATIO,
            (5000.0, 0.005, 0.02),
        ),
    ],
)
def test_lookup_variables_follow_the_table_definitions(flow_type, water_type, gas_type, expected):
    table = lookup_table(flow_type=flow_type, water_type=water_type, gas_type=gas_type)
    lookup = table.get_lookup_variables(oil_rate=100.0, water_rate=-25.0, gas_rate=5000.0)
    assert tuple(lookup) == pytest.approx(expected)


def test_lookup_variables_with_a_zero_denominator_stay_finite():
    table = lookup_table(
        flow_type=VFPFlowType.OIL,
        water_type=VFPWaterFractionType.WATER_OIL_RATIO,
        gas_type=VFPGasFractionType.GAS_OIL_RATIO,
    )
    no_oil = table.get_lookup_variables(oil_rate=0.0, water_rate=5.0, gas_rate=10.0)
    assert no_oil.water_fraction == pytest.approx(2.0)  # axis maximum
    assert no_oil.gas_fraction == pytest.approx(7.0)
    nothing = table.get_lookup_variables(oil_rate=0.0, water_rate=0.0, gas_rate=0.0)
    assert tuple(nothing) == (0.0, 0.0, 0.0)


def test_injector_lookup_uses_the_injected_phase_only():
    table = VFPTable(
        VFPData(
            table_number=1,
            well_type=WellType.INJECTOR,
            datum_depth=2000.0,
            flow_type=VFPFlowType.WATER,
            flow_rates=np.array([1.0, 10.0]),
            thps=np.array([10.0, 20.0]),
            bhps=np.zeros((2, 2, 1, 1, 1)),
        )
    )
    lookup = table.get_lookup_variables(oil_rate=0.0, water_rate=7.0, gas_rate=900.0)
    assert tuple(lookup) == (7.0, 0.0, 0.0)


def test_producer_table_rejects_an_injector_flow_type():
    with pytest.raises(ValidationError, match="flow type"):
        lookup_table(
            flow_type=VFPFlowType.WATER,
            water_type=VFPWaterFractionType.WATER_CUT,
            gas_type=VFPGasFractionType.GAS_OIL_RATIO,
        )


def test_conversion_follows_what_each_axis_measures():
    data = lookup_table(
        flow_type=VFPFlowType.GAS,
        water_type=VFPWaterFractionType.WATER_GAS_RATIO,
        gas_type=VFPGasFractionType.OIL_GAS_RATIO,
    )._data
    factors = get_conversion_factors(UnitSystem.FIELD, UnitSystem.METRIC)
    converted = data.convert(UnitSystem.METRIC)
    assert np.allclose(converted.flow_rates, data.flow_rates * factors["gas_surface_rate"])
    ratio = factors["liquid_surface_volume"] / factors["gas_surface_volume"]
    assert np.allclose(converted.water_fractions, data.water_fractions * ratio)
    assert np.allclose(converted.gas_fractions, data.gas_fractions * ratio)

    liquid = lookup_table(
        flow_type=VFPFlowType.OIL,
        water_type=VFPWaterFractionType.WATER_OIL_RATIO,
        gas_type=VFPGasFractionType.GAS_LIQUID_RATIO,
    )._data.convert(UnitSystem.METRIC)
    assert np.allclose(liquid.flow_rates, data.flow_rates * factors["liquid_surface_rate"])
    assert np.allclose(liquid.water_fractions, data.water_fractions)
    assert np.allclose(liquid.gas_fractions, data.gas_fractions * factors["gas_oil_ratio"])


def test_injector_table_becomes_a_five_axis_table_and_gas_flow_is_rescaled(tmp_path):
    deck = deck_from(
        tmp_path,
        "VFPINJ\n 5 2000 GAS THP FIELD BHP /\n 100 200 /\n 20 40 /\n 1  50 60 /\n 2  80 90 /\n",
    )
    data = load_vfp_data(deck, table_number=5)
    assert data.bhps.shape == (2, 2, 1, 1, 1)
    assert np.allclose(data.flow_rates, [100_000.0, 200_000.0])  # Mscf/d -> scf/d
    assert data.bhps[1, 0, 0, 0, 0] == pytest.approx(60.0)


def test_missing_table_is_reported(tmp_path):
    with pytest.raises(ValidationError, match="not found"):
        load_vfp_table(producer_table(tmp_path), table_number=9)
    with pytest.raises(ValidationError, match="No `VFPINJ`"):
        load_vfp_tables(producer_table(tmp_path), well_type=WellType.INJECTOR)


def test_producer_and_injector_tables_may_share_a_number(tmp_path):
    both = deck_from(
        tmp_path,
        VFPPROD_BODY.format(wfr="WCT", gfr="GOR", units="FIELD", gor0=0.1, gor1=0.5)
        + "VFPINJ\n 3 2000 WAT THP FIELD BHP /\n 100 200 /\n 20 40 /\n 1  50 60 /\n 2  80 90 /\n",
    )
    tables = VFPTables.from_deck(both)
    assert set(tables.producers) == {3} and set(tables.injectors) == {3}
    assert tables.table(3, well_type=WellType.PRODUCER).well_type == WellType.PRODUCER
    assert tables.table(3, well_type=WellType.INJECTOR).well_type == WellType.INJECTOR
    with pytest.raises(ValidationError, match="pass `well_type`"):
        load_vfp_table(both, table_number=3)
    assert (
        load_vfp_table(both, table_number=3, well_type=WellType.INJECTOR).well_type
        == WellType.INJECTOR
    )
    assert set(VFPTables.from_deck(both, well_type=WellType.PRODUCER).injectors) == set()


def test_later_definition_of_a_table_number_replaces_the_earlier_one(tmp_path):
    first = VFPPROD_BODY.format(wfr="WCT", gfr="GOR", units="FIELD", gor0=0.1, gor1=0.5)
    second = first.replace("2000 LIQ", "2500 LIQ")
    deck = deck_from(tmp_path, first + second)
    assert load_vfp_data(deck, table_number=3).datum_depth == pytest.approx(2500.0)
    assert len(load_vfp_data(deck)) == 1


def test_vfp_classes_build_from_a_parsed_table(tmp_path):
    deck = producer_table(tmp_path)
    assert isinstance(VFPData.from_deck(deck, table_number=3), VFPData)
    assert isinstance(VFPTable.from_deck(deck, table_number=3), VFPTable)
    assert [t.table_number for t in VFPTable.from_deck(deck)] == [3]
    assert 3 in VFPTables.from_deck(deck).producers


def region_model(region_numbers, unit_system=UnitSystem.FIELD):
    pore_volumes = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    regions = (
        None
        if region_numbers is None
        else types.SimpleNamespace(fluid_in_place_region=np.asarray(region_numbers))
    )
    reservoir = types.SimpleNamespace(
        pore_volumes=pore_volumes,
        compute_pore_volumes=lambda pressure: pore_volumes * (1.0 + 1e-5 * (pressure - 3000.0)),
        regions=regions,
    )
    return types.SimpleNamespace(reservoir=reservoir, unit_system=unit_system)


def region_workspace():
    state = types.SimpleNamespace(
        pressure=np.full(4, 3000.0),
        oil_saturation=np.array([0.5, 0.6, 0.4, 0.3]),
        water_saturation=np.array([0.3, 0.3, 0.3, 0.3]),
        gas_saturation=np.array([0.2, 0.1, 0.3, 0.4]),
        solution_gor=np.array([500.0, 400.0, 300.0, 200.0]),
        gas_solubility_in_water=np.array([20.0, 20.0, 10.0, 10.0]),
    )
    pvt = types.SimpleNamespace(
        oil_formation_volume_factor=np.array([1.2, 1.25, 1.3, 1.35]),
        water_formation_volume_factor=np.array([1.0, 1.0, 1.0, 1.0]),
        gas_formation_volume_factor=np.array([0.005, 0.005, 0.006, 0.006]),
        vaporized_oil_gas_ratio=np.array([1e-4, 2e-4, np.nan, 5e-5]),
    )
    return types.SimpleNamespace(reservoir=state, physics=types.SimpleNamespace(pvt=pvt))


@pytest.fixture
def with_workspace(monkeypatch):
    monkeypatch.setattr(well_summary, "get_workspace", lambda *, context: region_workspace())


def test_region_in_place_matches_hand_calculation(with_workspace):
    model = region_model([1, 1, 2, 2])
    to_stb = c.CUBIC_FEET_TO_STB
    oil_ft3 = np.array([0.5 * 1000 / 1.2, 0.6 * 2000 / 1.25, 0.4 * 3000 / 1.3, 0.3 * 4000 / 1.35])
    free_gas = np.array([0.2 * 1000 / 0.005, 0.1 * 2000 / 0.005, 0.3 * 3000 / 0.006])
    water_ft3 = np.array([0.3 * 1000, 0.3 * 2000, 0.3 * 3000, 0.3 * 4000])

    # Oil in the oil phase plus oil vaporized in the free gas, in cubic feet. Rv is in STB per scf.
    vaporized = np.array([1e-4, 2e-4, 0.0]) * free_gas / to_stb
    roip = well_summary.RegionOilInPlace(region=1)(model, None)
    assert roip == pytest.approx(oil_ft3[:2].sum() + vaporized[:2].sum())
    assert well_summary.RegionOilInPlace(region=2)(model, None) == pytest.approx(
        oil_ft3[2:].sum() + 5e-5 * (0.4 * 4000 / 0.006) / to_stb
    )

    assert well_summary.RegionWaterInPlace(region=2)(model, None) == pytest.approx(
        water_ft3[2:].sum()
    )

    # Free gas, gas in the oil (Rs, scf/STB) and gas in the water (Rsw, scf/STB), in scf.
    in_oil = np.array([500.0, 400.0]) * oil_ft3[:2] * to_stb
    in_water = np.array([20.0, 20.0]) * water_ft3[:2] * to_stb
    assert well_summary.RegionGasInPlace(region=1)(model, None) == pytest.approx(
        (free_gas[:2] + in_oil + in_water).sum()
    )


def test_a_model_without_vaporized_oil_or_gas_in_water_reads_those_as_zero(monkeypatch):
    workspace = region_workspace()
    workspace.physics.pvt.vaporized_oil_gas_ratio = np.full(4, np.nan)
    workspace.reservoir.gas_solubility_in_water = np.full(4, np.nan)
    monkeypatch.setattr(well_summary, "get_workspace", lambda *, context: workspace)
    model = region_model(None)
    oil_ft3 = np.array([0.5 * 1000 / 1.2, 0.6 * 2000 / 1.25, 0.4 * 3000 / 1.3, 0.3 * 4000 / 1.35])
    assert well_summary.RegionOilInPlace(region=1)(model, None) == pytest.approx(oil_ft3.sum())
    free = np.array([
        0.2 * 1000 / 0.005,
        0.1 * 2000 / 0.005,
        0.3 * 3000 / 0.006,
        0.4 * 4000 / 0.006,
    ])
    dissolved = np.array([500.0, 400.0, 300.0, 200.0]) * oil_ft3 * c.CUBIC_FEET_TO_STB
    assert well_summary.RegionGasInPlace(region=1)(model, None) == pytest.approx(
        (free + dissolved).sum()
    )


def test_regions_partition_the_field_and_other_unit_systems_need_no_barrel_conversion(
    with_workspace,
):
    model = region_model([1, 2, 1, 2])
    total = sum(well_summary.RegionOilInPlace(region=r)(model, None) for r in (1, 2))
    assert total == pytest.approx(
        well_summary.RegionOilInPlace(region=1)(region_model(None), None)
    )
    metric_model = region_model(None, UnitSystem.METRIC)
    metric_water = well_summary.RegionWaterInPlace(region=1)(metric_model, None)
    field_water = well_summary.RegionWaterInPlace(region=1)(region_model(None), None)
    assert metric_water == pytest.approx(field_water)


def test_region_without_cells_and_keys(with_workspace):
    with pytest.raises(Exception, match="has no cells"):
        well_summary.RegionOilInPlace(region=7)(region_model([1, 1, 2, 2]), None)
    assert well_summary.RegionGasInPlace(region=3).key == "RGIP:3"


def test_deck_loads_region_vectors_for_listed_and_all_regions(tmp_path):
    deck = deck_from(tmp_path, "")
    assert get_fluid_in_place_regions(deck) == [1]
    text = "RUNSPEC\nFIELD\nSUMMARY\nROIP\n 2 /\nRGIP\n/\nTSTEP\n 10 /\n"
    path = tmp_path / "s.DATA"
    path.write_text(text)
    summary_deck = DeckFile.from_path(path) if hasattr(DeckFile, "from_path") else DeckFile(path)
    items = load_summary_items(summary_deck)
    names = {
        type(quantity).__name__ + ":" + str(quantity.region)
        for item in items
        for quantity in getattr(item.action, "quantities", ())
    }
    assert {"RegionOilInPlace:2", "RegionGasInPlace:1"} <= names


def test_load_schedule_can_skip_summary_items(tmp_path):
    from bores.wells.deck import load_schedule

    path = tmp_path / "summary.DATA"
    path.write_text("RUNSPEC\nFIELD\nSUMMARY\nFOPR\nSCHEDULE\nTSTEP\n 10 /\n")
    deck = DeckFile.from_path(path) if hasattr(DeckFile, "from_path") else DeckFile(path)

    assert [item.name for item in load_schedule(deck)] == ["summary@10.0"]
    assert load_schedule(deck, load_summary=False).items == ()


def build_compiled_controls():
    from bores.types import FluidPhase
    from bores.wells.compile import compile_well_controls
    from bores.wells.controls import (
        InjectorControl,
        InjectorControlMode,
        ProducerControl,
        ProducerControlMode,
    )

    wells = {
        name: types.SimpleNamespace(well_type=well_type)
        for name, well_type in (
            ("P1", WellType.PRODUCER),
            ("I1", WellType.INJECTOR),
            ("P2", WellType.PRODUCER),
        )
    }
    controls = {
        "P1": ProducerControl(
            mode=ProducerControlMode.BHP,
            target_bhp=1000.0,
            vfp_table=3,
            artificial_lift_quantity=250.0,
        ),
        "I1": InjectorControl(
            injected_phase=FluidPhase.WATER,
            mode=InjectorControlMode.BHP,
            target_bhp=4000.0,
            vfp_table=4,
        ),
    }
    return compile_well_controls(names=["P1", "I1", "P2"], controls=controls, wells=wells)


def test_compiled_controls_carry_vfp_table_numbers_and_alq():
    from bores.wells.compile import UNSET_INT

    compiled = build_compiled_controls()
    assert compiled.vfp_table_numbers.tolist() == [3, 4, UNSET_INT]
    assert np.allclose(compiled.artificial_lift_quantities, [250.0, 0.0, 0.0])
    assert compiled.get_vfp_table_number(well_row=0) == 3
    assert compiled.get_vfp_table_number(well_row=2) is None
    compiled.set_vfp_table_number(well_row=np.array([0, 2]), value=np.array([7, 8]))
    compiled.set_artificial_lift_quantity(well_row=0, value=400.0)
    assert compiled.get_vfp_table_number(well_row=np.array([0, 2])).tolist() == [7, 8]
    assert compiled.get_artificial_lift_quantity(well_row=0) == pytest.approx(400.0)


def test_decompiled_controls_reflect_a_patched_vfp_table_and_alq():
    from bores.wells.decompile import decompile_well_control

    compiled = build_compiled_controls()
    compiled.set_artificial_lift_quantity(well_row=0, value=900.0)
    compiled.set_vfp_table_number(well_row=1, value=11)
    producer = decompile_well_control(compiled, 0, UnitSystem.FIELD)
    injector = decompile_well_control(compiled, 1, UnitSystem.FIELD)
    assert producer.vfp_table == 3 and producer.artificial_lift_quantity == pytest.approx(900.0)
    assert injector.vfp_table == 11


def lift_model():
    controls = build_compiled_controls()
    rows = {"P1": 0, "I1": 1, "P2": 2}
    return types.SimpleNamespace(
        wells=types.SimpleNamespace(controls=controls, well_row=lambda *, name: rows[name])
    )


def test_set_well_lift_patches_only_what_it_is_given():
    from bores.wells.compile import UNSET_INT
    from bores.wells.schedule import SetWellLift

    model = lift_model()
    SetWellLift(well_name="P1", artificial_lift_quantity=600.0)(model, None)
    assert model.wells.controls.get_vfp_table_number(well_row=0) == 3  # unchanged
    assert model.wells.controls.get_artificial_lift_quantity(well_row=0) == pytest.approx(600.0)
    SetWellLift(well_name="P1", vfp_table=5, efficiency_factor=0.9)(model, None)
    assert model.wells.controls.get_vfp_table_number(well_row=0) == 5
    assert model.wells.controls.get_artificial_lift_quantity(well_row=0) == pytest.approx(600.0)
    assert model.wells.controls.efficiency_factors[0] == pytest.approx(0.9)
    SetWellLift(well_name="P1", vfp_table=0)(model, None)  # 0 = no table
    assert model.wells.controls.vfp_table_numbers[0] == UNSET_INT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"well_name": "I1", "artificial_lift_quantity": 10.0},
        {"well_name": "P1", "artificial_lift_quantity": -1.0},
        {"well_name": "P1", "efficiency_factor": 1.5},
    ],
)
def test_set_well_lift_rejects_invalid_requests(kwargs):
    from bores.wells.schedule import SetWellLift

    with pytest.raises(ValidationError):
        SetWellLift(**kwargs)(lift_model(), None)


def test_schedule_turns_wlift_and_reissued_controls_into_lift_actions(tmp_path):
    from bores.wells.deck import load_schedule
    from bores.wells.schedule import SetWellLift

    deck = deck_from(
        tmp_path,
        "WCONPROD\n 'P1' OPEN LRAT 1* 1* 1* 500 1* 80 1* 3 250 /\n/\nTSTEP\n 30 /\n"
        "WLIFT\n 'P1' 1* 1* 2 400 0.8 /\n/\nTSTEP\n 30 /\n",
    )
    actions = {
        item.name.split("@")[0]: item.action for item in load_schedule(deck, compiled_at=-1.0)
    }
    assert isinstance(actions["wconprod-lift:P1"], SetWellLift)
    assert actions["wconprod-lift:P1"].vfp_table == 3
    assert actions["wconprod-lift:P1"].artificial_lift_quantity == pytest.approx(250.0)
    lift = actions["wlift:P1"]
    assert (lift.vfp_table, lift.artificial_lift_quantity, lift.efficiency_factor) == (
        2,
        400.0,
        0.8,
    )


class StubWorkspace:
    """Stands in for `SimulationWorkspace`, which the rate events type-check their input against."""

    def __init__(self, oil, water, gas):
        self.wells = types.SimpleNamespace(
            surface_oil_rates=np.array(oil, dtype=float),
            surface_water_rates=np.array(water, dtype=float),
            surface_gas_rates=np.array(gas, dtype=float),
            oil_rates=np.array(oil, dtype=float),
            water_rates=np.array(water, dtype=float),
            gas_rates=np.array(gas, dtype=float),
        )


@pytest.fixture
def stub_workspace_type(monkeypatch):
    monkeypatch.setattr("bores.simulation.workspace.SimulationWorkspace", StubWorkspace)


def event_context(time, oil=(0.0, 0.0, 0.0), water=(0.0, 0.0, 0.0), gas=(0.0, 0.0, 0.0)):
    return types.SimpleNamespace(
        time=time, previous_time=None, extra={"workspace": StubWorkspace(oil, water, gas)}
    )


def test_time_window_event_is_live_only_inside_the_window():
    from bores.schedule.events import TimeWindowEvent

    window = TimeWindowEvent(start=30.0, end=90.0)
    assert [
        window(None, types.SimpleNamespace(time=t)) for t in (10.0, 30.0, 60.0, 90.0, 120.0)
    ] == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert TimeWindowEvent(start=30.0)(None, types.SimpleNamespace(time=1e9))


def test_rate_water_cut_and_gas_oil_ratio_events_read_the_workspace(stub_workspace_type):
    from bores.schedule.events import ComparisonOperator
    from bores.wells.controls import RateQuantity
    from bores.wells.schedule import GasOilRatioThreshold, RateThreshold, WaterCutThreshold

    model = lift_model()
    context = event_context(1.0, oil=(80.0, 0, 0), water=(20.0, 0, 0), gas=(40000.0, 0, 0))
    liquid = RateThreshold(
        well_name="P1",
        quantity=RateQuantity.LIQUID,
        surface=True,
        op=ComparisonOperator.LT,
        threshold=150.0,
    )
    oil = RateThreshold(
        well_name="P1",
        quantity=RateQuantity.OIL,
        surface=True,
        op=ComparisonOperator.LT,
        threshold=50.0,
    )
    assert liquid(model, context) and not oil(model, context)
    assert WaterCutThreshold(well_name="P1", op=ComparisonOperator.GT, threshold=0.15)(
        model, context
    )
    assert not WaterCutThreshold(well_name="P1", op=ComparisonOperator.GT, threshold=0.25)(
        model, context
    )
    assert GasOilRatioThreshold(well_name="P1", op=ComparisonOperator.GT, threshold=400.0)(
        model, context
    )
    assert not GasOilRatioThreshold(well_name="P1", op=ComparisonOperator.GT, threshold=600.0)(
        model, context
    )


def test_water_cut_and_gas_oil_ratio_are_safe_for_a_well_with_no_oil_or_liquid(
    stub_workspace_type,
):
    from bores.schedule.events import ComparisonOperator
    from bores.wells.schedule import GasOilRatioThreshold, WaterCutThreshold

    model = lift_model()
    assert not WaterCutThreshold(well_name="P2", op=ComparisonOperator.GT, threshold=0.0)(
        model, event_context(1.0)
    )
    only_gas = event_context(1.0, gas=(0.0, 0.0, 500.0))
    assert GasOilRatioThreshold(well_name="P2", op=ComparisonOperator.GT, threshold=1e9)(
        model, only_gas
    )


WLIFT_SCHEDULE = (
    "WCONPROD\n 'P1' OPEN LRAT 1* 1* 1* 500 1* 80 1* 3 250 /\n/\nTSTEP\n 30 /\n"
    "WLIFT\n 'P1' 100 OIL 2 400 0.8 /\n/\nTSTEP\n 60 /\n"
    "WCONPROD\n 'P1' OPEN LRAT 1* 1* 1* 500 1* 80 1* 6 0 /\n/\nTSTEP\n 30 /\n"
)


def wlift_item(tmp_path, text=WLIFT_SCHEDULE):
    from bores.wells.deck import load_schedule

    deck = deck_from(tmp_path, text)
    return next(
        item for item in load_schedule(deck, compiled_at=-1.0) if item.name.startswith("wlift:")
    )


def test_triggered_wlift_is_live_between_the_record_and_the_next_lift_change(
    tmp_path, stub_workspace_type
):
    item = wlift_item(tmp_path)
    model = lift_model()
    low, high = (50.0, 0, 0), (150.0, 0, 0)
    assert not item.event(model, event_context(20.0, oil=low))  # before the WLIFT record
    assert item.event(model, event_context(40.0, oil=low))  # inside the window, rate below limit
    assert not item.event(model, event_context(40.0, oil=high))  # inside the window, rate fine
    assert not item.event(model, event_context(95.0, oil=low))  # after the next WCONPROD


def test_triggered_wlift_changes_persist_after_the_trigger_clears(tmp_path, stub_workspace_type):
    item = wlift_item(tmp_path)
    model = lift_model()
    assert item.event(model, event_context(40.0, oil=(50.0, 0, 0)))
    item.action(model, event_context(40.0))
    controls = model.wells.controls
    assert controls.get_vfp_table_number(well_row=0) == 2
    assert controls.get_artificial_lift_quantity(well_row=0) == pytest.approx(400.0)
    # The rate recovers, the trigger no longer holds, and the new settings stay.
    assert not item.event(model, event_context(50.0, oil=(500.0, 0, 0)))
    assert controls.get_vfp_table_number(well_row=0) == 2


def test_wlift_combines_its_triggers_and_applies_a_new_thp_limit(tmp_path):
    from bores.schedule.actions import RunSequence
    from bores.schedule.events import AllOf, AnyOf

    text = "WLIFT\n 'P1' 100 OIL 2 400 1* 0.7 5 3000 /\n/\nTSTEP\n 30 /\n"
    item = wlift_item(tmp_path, text)
    assert isinstance(item.event, AllOf) and isinstance(item.event.events[1], AnyOf)
    kinds = {type(event).__name__ for event in item.event.events[1].events}
    assert kinds == {"RateThreshold", "WaterCutThreshold", "GasOilRatioThreshold"}
    assert isinstance(item.action, RunSequence) and len(item.action.actions) == 2
    assert item.event.events[0].end is None  # nothing later changes this well's lift


def test_wlift_with_no_trigger_applies_at_its_own_time_and_shifts_stay_unsupported(tmp_path):
    from bores.errors import NotSupportedError
    from bores.schedule.events import TimeEvent
    from bores.wells.deck import load_schedule

    assert isinstance(
        wlift_item(tmp_path, "WLIFT\n 'P1' 1* 1* 2 400 /\n/\nTSTEP\n 30 /\n").event, TimeEvent
    )
    deck = deck_from(tmp_path, "WLIFT\n 'P1' 100 OIL 2 400 1* 1* 1* 1* 50 /\n/\nTSTEP\n 30 /\n")
    with pytest.raises(NotSupportedError, match="alq_shift"):
        load_schedule(deck, compiled_at=-1.0)


def build_well_system(**kwargs):
    from bores.wells.factories import make_producer, make_well_controls, make_wells
    from bores.wells.model import WellSystem

    built = {
        name: make_producer(
            name, surface_location=(0.0, 0.0), perforation_depths=(0.0, 10.0), target_bhp=1000.0
        )
        for name in ("P1", "P2", "P3")
    }
    return WellSystem(
        wells=make_wells(*(well for well, _ in built.values())),
        well_controls=make_well_controls(**{
            name: control for name, (_, control) in built.items()
        }),
        **kwargs,
    )


def test_tubing_performance_single_item_applies_to_every_well():
    from bores.wells.hydraulics import homogeneous_wellbore

    model = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)
    system = build_well_system(tubing_performance=model)
    assert all(system.get_tubing_performance(name) is model for name in ("P1", "P2", "P3"))
    assert build_well_system().get_tubing_performance("P1") is None


def test_tubing_performance_mapping_uses_the_wildcard_as_the_fallback(tmp_path):
    from bores.wells.hydraulics import homogeneous_wellbore

    model = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)
    tables = VFPTables.from_deck(producer_table(tmp_path))
    system = build_well_system(tubing_performance={"*": model, "P1": tables, "P2": None})
    assert system.get_tubing_performance("P1") is tables
    assert system.get_tubing_performance("P2") is None  # explicitly none, despite "*"
    assert system.get_tubing_performance("P3") is model
    assert build_well_system(tubing_performance={"P1": model}).get_tubing_performance("P3") is None


def test_tubing_performance_rejects_wells_that_do_not_exist():
    from bores.wells.hydraulics import homogeneous_wellbore

    model = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)
    with pytest.raises(ValidationError, match="not in this system"):
        build_well_system(tubing_performance={"NOPE": model})


def test_tubing_performance_converts_with_the_system(tmp_path):
    tables = VFPTables.from_deck(producer_table(tmp_path))
    single = build_well_system(tubing_performance=tables).convert(UnitSystem.METRIC)
    assert single.tubing_performance.unit_system == UnitSystem.METRIC
    mapped = build_well_system(tubing_performance={"P1": tables, "P2": None}).convert(
        UnitSystem.METRIC
    )
    assert mapped.tubing_performance["P1"].unit_system == UnitSystem.METRIC
    assert mapped.tubing_performance["P2"] is None


def test_get_thp_inverts_the_table_and_extends_past_its_ends(tmp_path):
    table = VFPTable.from_deck(producer_table(tmp_path), table_number=3)
    args = {
        "flow_rate": 100.0,
        "water_fraction": 0.0,
        "gas_fraction": 100.0,
        "artificial_lift_quantity": 0.0,
    }
    low = float(table.query(thp=20.0, **args))
    high = float(table.query(thp=40.0, **args))
    assert table.get_thp(bhp=low, **args) == pytest.approx(20.0)
    assert table.get_thp(bhp=high, **args) == pytest.approx(40.0)
    assert table.get_thp(bhp=(low + high) / 2, **args) == pytest.approx(30.0)
    assert table.get_thp(bhp=high + (high - low), **args) == pytest.approx(60.0)  # extended
    assert table.get_thp(bhp=low - (high - low), **args) == pytest.approx(0.0)


def tubing_stand_ins(tmp_path, *, vfp_number):
    from bores.wells.compile import UNSET_INT

    controls = types.SimpleNamespace(
        artificial_lift_quantities=np.array([0.0, 25.0]),
        well_kinds=np.array([0, 0]),
        get_vfp_table_number=lambda *, well_row: (
            None if vfp_number[well_row] == UNSET_INT else int(vfp_number[well_row])
        ),
    )
    return types.SimpleNamespace(controls=controls), VFPTables.from_deck(producer_table(tmp_path))


def test_tubing_is_resolved_from_the_compiled_controls_at_solve_time(tmp_path):
    from bores.wells.compile import UNSET_INT
    from bores.wells.resolution.tubing import resolve_well_tubing

    system, tables = tubing_stand_ins(tmp_path, vfp_number=[3, UNSET_INT])
    with_table = resolve_well_tubing(tubing_performance=tables, compiled_system=system, well_row=0)
    assert with_table.vfp_table is tables.producers[3] and with_table.can_evaluate_thp
    without = resolve_well_tubing(tubing_performance=tables, compiled_system=system, well_row=1)
    assert without.vfp_table is None and not without.can_evaluate_thp
    assert without.artificial_lift_quantity == pytest.approx(25.0)
    none = resolve_well_tubing(tubing_performance=None, compiled_system=system, well_row=0)
    assert not none.can_evaluate_thp


def test_missing_table_number_in_the_collection_is_reported(tmp_path):
    from bores.wells.resolution.tubing import resolve_well_tubing

    system, tables = tubing_stand_ins(tmp_path, vfp_number=[9, 9])
    with pytest.raises(ValidationError, match="No producer VFP table numbered 9"):
        resolve_well_tubing(tubing_performance=tables, compiled_system=system, well_row=0)


def test_vfp_thp_uses_surface_rates_and_refuses_a_datum_mismatch(tmp_path):
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.tubing import WellTubing, compute_tubing_head_pressure

    table = VFPTable.from_deck(producer_table(tmp_path), table_number=3)
    tubing = WellTubing(None, table, 0.0)
    rates = PhaseValues(oil=100.0, water=0.0, gas=10_000.0)  # gas-oil ratio 100
    common = {
        "tubing": tubing,
        "phase_rates": rates,
        "surface_phase_rates": rates,
        "surface_sample": None,
        "is_injector": False,
    }
    expected_bhp = float(
        table.query(
            flow_rate=100.0,
            thp=30.0,
            water_fraction=0.0,
            gas_fraction=100.0,
            artificial_lift_quantity=0.0,
        )
    )
    thp = compute_tubing_head_pressure(
        reference_depth=2000.0, reference_pressure=expected_bhp, **common
    )
    assert thp == pytest.approx(30.0)
    with pytest.raises(ValidationError, match="datum depth"):
        compute_tubing_head_pressure(
            reference_depth=2100.0, reference_pressure=expected_bhp, **common
        )
    with pytest.raises(ValidationError, match="no `WellBoreModel` or VFP table"):
        compute_tubing_head_pressure(
            reference_depth=2000.0,
            reference_pressure=1.0,
            **{**common, "tubing": WellTubing(None, None, 0.0)},
        )


def test_vfp_thp_follows_an_oil_rate_table_with_no_gas(tmp_path):
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.tubing import WellTubing, compute_tubing_head_pressure

    table = VFPTable.from_deck(producer_table(tmp_path, wfr="WOR", flo="OIL"), table_number=3)
    rates = PhaseValues(oil=100.0, water=100.0, gas=0.0)  # liquid 200, water-oil ratio 1
    expected_bhp = float(
        table.query(
            flow_rate=100.0,  # oil rate, not oil plus water
            thp=30.0,
            water_fraction=1.0,
            gas_fraction=0.0,
            artificial_lift_quantity=0.0,
        )
    )
    thp = compute_tubing_head_pressure(
        tubing=WellTubing(None, table, 0.0),
        reference_depth=2000.0,
        reference_pressure=expected_bhp,
        phase_rates=rates,
        surface_phase_rates=rates,
        surface_sample=None,
        is_injector=False,
    )
    assert thp == pytest.approx(30.0)


ALL_TYPE_COMBINATIONS = [
    (flow, water, gas)
    for flow in (VFPFlowType.OIL, VFPFlowType.LIQUID, VFPFlowType.GAS)
    for water in VFPWaterFractionType
    for gas in VFPGasFractionType
]


@pytest.mark.parametrize("flow_type, water_type, gas_type", ALL_TYPE_COMBINATIONS)
def test_sampled_rates_reproduce_the_lookup_values_for_every_definition(
    flow_type, water_type, gas_type
):
    from bores.wells.hydraulics.vfp import compute_surface_phase_rates

    table = lookup_table(flow_type=flow_type, water_type=water_type, gas_type=gas_type)
    rates = compute_surface_phase_rates(
        flow_type=flow_type,
        water_fraction_type=water_type,
        gas_fraction_type=gas_type,
        flow_rate=300.0,
        water_fraction=0.4,
        gas_fraction=0.3,
    )
    lookup = table.get_lookup_variables(
        oil_rate=rates.oil, water_rate=rates.water, gas_rate=rates.gas
    )
    assert tuple(lookup) == pytest.approx((300.0, 0.4, 0.3))


def test_values_that_describe_no_rates_are_rejected():
    from bores.wells.hydraulics.vfp import compute_surface_phase_rates

    with pytest.raises(ValidationError, match="non-negative"):
        compute_surface_phase_rates(
            flow_type=VFPFlowType.OIL,
            water_fraction_type=VFPWaterFractionType.WATER_CUT,
            gas_fraction_type=VFPGasFractionType.GAS_OIL_RATIO,
            flow_rate=100.0,
            water_fraction=1.0,  # all water, so no oil rate to speak of
            gas_fraction=0.0,
        )


def sampled_table(**kwargs):
    from bores.wells.hydraulics.base import SurfaceSample
    from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
    from bores.wells.hydraulics.vfp import as_vfp_table

    return as_vfp_table(
        homogeneous_wellbore(tubing_inner_diameter=0.25),
        table_number=1,
        reference_depth=5000.0,
        thps=np.array([100.0, 300.0]),
        surface_sample=SurfaceSample(density=50.0, viscosity=1.0),
        min_bhp=0.0,
        max_bhp=20000.0,
        **kwargs,
    )


def test_correlation_is_sampled_in_the_declared_terms():
    table = sampled_table(
        well_type=WellType.PRODUCER,
        flow_rates=np.array([100.0, 400.0]),
        flow_type=VFPFlowType.OIL,
        water_fraction_type=VFPWaterFractionType.WATER_OIL_RATIO,
        gas_fraction_type=VFPGasFractionType.GAS_LIQUID_RATIO,
        water_fractions=np.array([0.0, 1.0]),
        gas_fractions=np.array([0.0, 100.0]),
    )
    assert table.flow_type == VFPFlowType.OIL
    assert table.water_fraction_type == VFPWaterFractionType.WATER_OIL_RATIO
    assert table.gas_fraction_type == VFPGasFractionType.GAS_LIQUID_RATIO
    bhps = table._data.bhps
    assert bhps.shape == (2, 2, 2, 2, 1)
    assert (bhps < 20000.0).all()
    assert (np.diff(bhps, axis=1) > 150.0).all()  # THP rises 200, so BHP does


def test_injector_table_is_sampled_in_the_injected_phase():
    from bores.types import FluidPhase

    table = sampled_table(
        well_type=WellType.INJECTOR,
        flow_rates=np.array([100.0, 1000.0]),
        injected_phase=FluidPhase.WATER,
    )
    assert table.flow_type == VFPFlowType.WATER
    with pytest.raises(ValidationError, match="water or gas"):
        sampled_table(
            well_type=WellType.INJECTOR,
            flow_rates=np.array([100.0, 1000.0]),
            flow_type=VFPFlowType.OIL,
        )


def flowing_fluid(*, zero_flow_density=50.0):
    from bores.wells.resolution.tubing import FlowingFluid

    return FlowingFluid(
        oil_densities=np.array([50.0, 50.0]),
        water_densities=np.array([62.4, 62.4]),
        gas_densities=np.array([5.0, 5.0]),
        zero_flow_density=zero_flow_density,
    )


def connection_rates(*, oil, water=(0.0, 0.0), gas=(0.0, 0.0)):
    from bores.wells.workspace import ConnectionRates

    return ConnectionRates(oil=np.array(oil), water=np.array(water), gas=np.array(gas))


def test_datum_shift_is_the_weight_of_the_flowing_column(tmp_path):
    from bores.wells.resolution.tubing import compute_datum_pressure_shift

    table = VFPTable.from_deck(producer_table(tmp_path), table_number=3)  # datum 2000 ft
    rates = connection_rates(oil=(60.0, 40.0))
    shift = compute_datum_pressure_shift(
        table=table,
        reference_depth=1900.0,
        flowing_fluid=flowing_fluid(),
        connection_rates=rates,
    )
    assert shift == pytest.approx(50.0 * 100.0 / 144.0)  # psi: density 50 lb/ft3 over 100 ft
    shallower = compute_datum_pressure_shift(
        table=table,
        reference_depth=2100.0,
        flowing_fluid=flowing_fluid(),
        connection_rates=rates,
    )
    assert shallower == pytest.approx(-50.0 * 100.0 / 144.0)
    same = compute_datum_pressure_shift(
        table=table, reference_depth=2000.0, flowing_fluid=None, connection_rates=None
    )
    assert same == pytest.approx(0.0)


def test_flowing_density_is_mass_rate_over_volume_rate_across_connections():
    from bores.wells.resolution.tubing import FlowingFluid, compute_flowing_density

    fluid = FlowingFluid(
        oil_densities=np.array([40.0, 60.0]),
        water_densities=np.array([62.4, 62.4]),
        gas_densities=np.array([4.0, 6.0]),
        zero_flow_density=55.0,
    )
    rates = connection_rates(oil=(10.0, 30.0), water=(0.0, -20.0), gas=(20.0, 20.0))
    expected = (10 * 40.0 + 30 * 60.0 + 20 * 62.4 + 20 * 4.0 + 20 * 6.0) / 100.0
    assert compute_flowing_density(flowing_fluid=fluid, connection_rates=rates) == pytest.approx(
        expected
    )
    idle = connection_rates(oil=(0.0, 0.0))
    assert compute_flowing_density(flowing_fluid=fluid, connection_rates=idle) == pytest.approx(
        55.0
    )


def test_thp_is_found_through_a_datum_shift(tmp_path):
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.tubing import (
        WellTubing,
        compute_datum_pressure_shift,
        compute_tubing_head_pressure,
    )

    table = VFPTable.from_deck(producer_table(tmp_path), table_number=3)
    rates = PhaseValues(oil=100.0, water=0.0, gas=10_000.0)
    rates_by_connection = connection_rates(oil=(60.0, 40.0), gas=(6000.0, 4000.0))
    tubing = WellTubing(None, table, 0.0, flowing_fluid())
    datum_bhp = float(
        table.query(
            flow_rate=100.0,
            thp=30.0,
            water_fraction=0.0,
            gas_fraction=100.0,
            artificial_lift_quantity=0.0,
        )
    )
    shift = compute_datum_pressure_shift(
        table=table,
        reference_depth=1900.0,
        flowing_fluid=flowing_fluid(),
        connection_rates=rates_by_connection,
    )
    kwargs = {
        "tubing": tubing,
        "reference_depth": 1900.0,
        "reference_pressure": datum_bhp - shift,
        "phase_rates": rates,
        "surface_phase_rates": rates,
        "surface_sample": None,
        "is_injector": False,
    }
    thp = compute_tubing_head_pressure(**kwargs, connection_rates=rates_by_connection)
    assert thp == pytest.approx(30.0)
    with pytest.raises(ValidationError, match="datum depth"):
        compute_tubing_head_pressure(**kwargs)


def test_attached_flowing_fluid_records_densities_and_a_zero_flow_density():
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.tubing import WellTubing, attach_flowing_fluid

    samples = [
        sample._replace(phase_mobilities=PhaseValues(1.0, 0.0, 2.0))
        for sample in hydrostatic_samples(oil_densities=(40.0, 60.0))
    ]
    workspace = types.SimpleNamespace(
        well_indices=np.array([1.0, 3.0]),
        connection_fluids=compile_connection_fluids(connection_samples=samples, dtype=np.float64),
    )
    table = lookup_table(
        flow_type=VFPFlowType.LIQUID,
        water_type=VFPWaterFractionType.WATER_CUT,
        gas_type=VFPGasFractionType.GAS_OIL_RATIO,
    )
    tubing = attach_flowing_fluid(
        tubing=WellTubing(None, table, 0.0),
        workspace=workspace,
        relevant_phases=PhaseValues(oil=1.0, water=1.0, gas=1.0),
    )
    fluid = tubing.flowing_fluid
    assert fluid.oil_densities.tolist() == [40.0, 60.0]
    # Deliverability is oil 4 : gas 8, so the idle column is 1/3 oil (mean 55) and 2/3 gas (10).
    assert fluid.zero_flow_density == pytest.approx(55.0 / 3 + 2 * 10.0 / 3)
    untouched = attach_flowing_fluid(
        tubing=WellTubing(None, None, 0.0),
        workspace=workspace,
        relevant_phases=PhaseValues(oil=1.0, water=1.0, gas=1.0),
    )
    assert untouched.flowing_fluid is None


def resolve_producer(
    *, limits=None, vfp_tables=None, vfp_number=None, mode=None, target_rate=None
):
    """Resolves a three-connection producer on a real compiled system, and returns the workspace."""
    import attrs

    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.types import Orientation
    from bores.wells.compile import compile_well_system
    from bores.wells.factories import make_producer, make_well_controls, make_wells
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.engine import resolve_well_control
    from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
    from bores.wells.state import ConnectionSample
    from bores.wells.workspace import build_wells_workspace

    grid = make_cartesian_grid(nx=1, ny=1, nz=3, dx=50.0, dy=50.0, dz=50.0)
    well, control = make_producer(
        "P1",
        surface_location=(25.0, 25.0),
        perforation_depths=(0.0, 150.0),
        target_bhp=None if target_rate else 1000.0,
        target_rate=target_rate,
        reference_depth=75.0,
        limits=limits,
    )
    if vfp_number is not None:
        control = attrs.evolve(control, vfp_table=vfp_number)
    compiled = compile_well_system(
        make_wells(well),
        make_well_controls(P1=control),
        grid,
        permeabilities={orientation: np.full(3, 100.0) for orientation in Orientation},
    )
    samples = [
        ConnectionSample(
            cell_index=i,
            pressure=3000.0 + 10.0 * i,
            temperature=200.0,
            phase_saturations=PhaseValues(0.6, 0.3, 0.1),
            phase_mobilities=PhaseValues(2.0, 1.0, 5.0),
            phase_densities=PhaseValues(50.0 + i, 62.4, 5.0),
            phase_viscosities=PhaseValues(1.0, 0.5, 0.02),
            gas_liquid_surface_tension=20.0,
            phase_fvfs=PhaseValues(1.2, 1.0, 0.01),
        )
        for i in range(3)
    ]
    workspace = build_wells_workspace(n_wells=1, n_connections=3)
    resolve_well_control(
        compiled_system=compiled,
        well_row=0,
        tubing_performance=vfp_tables,
        connection_fluids=compile_connection_fluids(connection_samples=samples),
        workspace=workspace,
        control_spec=WellControlSpec(
            connection_pressure_mode=mode or ConnectionPressureMode.UNIFORM_BHP
        ),
    )
    return workspace, samples


def test_a_limit_that_is_not_violated_leaves_the_connection_rates_unchanged():
    from bores.wells.controls import RateLimit, RateQuantity

    plain, _ = resolve_producer()
    limited, _ = resolve_producer(limits=[RateLimit(quantity=RateQuantity.OIL, max_value=1e12)])
    for name in ("connection_oil_rates", "connection_water_rates", "connection_gas_rates"):
        assert np.all(getattr(plain, name) > 0.0)
        assert np.allclose(getattr(limited, name), getattr(plain, name))
    assert np.allclose(limited.connection_pressures, plain.connection_pressures)


def test_resolved_thp_with_a_vfp_table_accounts_for_the_datum_depth(tmp_path):
    from bores.wells.workspace import ConnectionRates

    tables = VFPTables.from_deck(producer_table(tmp_path))
    from bores.wells.controls import RateLimit, RateQuantity

    # The rate limit is never violated; it is there to prove the THP uses the nominal rates.
    workspace, samples = resolve_producer(
        vfp_tables=tables,
        vfp_number=3,
        limits=[RateLimit(quantity=RateQuantity.OIL, max_value=1e12)],
    )
    table = tables.table(3, well_type=WellType.PRODUCER)
    rates = ConnectionRates(
        oil=workspace.connection_oil_rates,
        water=workspace.connection_water_rates,
        gas=workspace.connection_gas_rates,
    )
    densities = {
        "oil": np.array([s.phase_densities.oil for s in samples]),
        "water": np.array([s.phase_densities.water for s in samples]),
        "gas": np.array([s.phase_densities.gas for s in samples]),
    }
    mass = sum(float(np.dot(getattr(rates, p), densities[p])) for p in densities)
    volume = sum(float(getattr(rates, p).sum()) for p in densities)
    datum_bhp = workspace.bhps[0] + (mass / volume) * (table.datum_depth - 75.0) / 144.0
    lookup = table.get_lookup_variables(
        oil_rate=workspace.surface_oil_rates[0],
        water_rate=workspace.surface_water_rates[0],
        gas_rate=workspace.surface_gas_rates[0],
    )
    expected = table.get_thp(
        flow_rate=lookup.flow_rate,
        bhp=datum_bhp,
        water_fraction=lookup.water_fraction,
        gas_fraction=lookup.gas_fraction,
        artificial_lift_quantity=0.0,
    )
    assert workspace.thps[0] == pytest.approx(expected)


def hydrostatic_samples(oil_densities=(40.0, 50.0, 60.0), mobility=1.0):
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.state import ConnectionSample

    return [
        ConnectionSample(
            cell_index=i,
            pressure=3000.0,
            temperature=200.0,
            phase_saturations=PhaseValues(0.5, 0.3, 0.2),
            phase_mobilities=PhaseValues(mobility, mobility, mobility),
            phase_densities=PhaseValues(oil_density, 60.0, 10.0),
            phase_viscosities=PhaseValues(1.0, 1.0, 1.0),
            gas_liquid_surface_tension=20.0,
            phase_fvfs=PhaseValues(1.0, 1.0, 1.0),
        )
        for i, oil_density in enumerate(oil_densities)
    ]


def hydrostatic_pressures(*, reference_depth, oil, depths=(100.0, 200.0, 400.0), **kwargs):
    from bores.wells.hydraulics.hydrostatic import compute_perforation_pressures

    count = len(depths)
    rates = connection_rates(
        oil=oil,
        water=kwargs.get("water", (0.0,) * count),
        gas=kwargs.get("gas", (0.0,) * count),
    )
    return compute_perforation_pressures(
        reference_depth=reference_depth,
        reference_pressure=1000.0,
        representative_depths=np.array(depths),
        connection_oil_rates=rates.oil,
        connection_water_rates=rates.water,
        connection_gas_rates=rates.gas,
        connection_fluids=compile_connection_fluids(
            connection_samples=hydrostatic_samples()[:count], dtype=np.float64
        ),
        unit_system=UnitSystem.FIELD,
    )


PSI_PER_LB_FT2 = 1.0 / 144.0


def test_hydrostatic_segments_carry_the_mixture_of_every_connection_beneath():
    # Reference at the top connection (100 ft). A segment carries only the connections beneath
    # it: 100-200 ft the middle and bottom ones (oil densities 50, 60 at rates 20, 30), and
    # 200-400 ft only the bottom one.
    pressures = hydrostatic_pressures(reference_depth=100.0, oil=(10.0, 20.0, 30.0))
    top_to_middle = (20 * 50.0 + 30 * 60.0) / 50.0
    middle_to_bottom = 60.0
    assert pressures[0] == pytest.approx(1000.0)
    assert pressures[1] == pytest.approx(1000.0 + top_to_middle * 100.0 * PSI_PER_LB_FT2)
    assert pressures[2] == pytest.approx(pressures[1] + middle_to_bottom * 200.0 * PSI_PER_LB_FT2)


def test_hydrostatic_pressure_falls_above_the_reference_depth():
    # Reference between the first and second connection: the first is above it.
    pressures = hydrostatic_pressures(reference_depth=150.0, oil=(10.0, 20.0, 30.0))
    above = (20 * 50.0 + 30 * 60.0) / 50.0  # everything at or below the reference depth
    assert pressures[0] == pytest.approx(1000.0 - above * 50.0 * PSI_PER_LB_FT2)
    below = above  # segment 150-200 carries connections 2 and 3
    assert pressures[1] == pytest.approx(1000.0 + below * 50.0 * PSI_PER_LB_FT2)
    assert pressures[2] > pressures[1]


def test_hydrostatic_pressure_with_no_flow_uses_the_static_reservoir_fluid():
    pressures = hydrostatic_pressures(reference_depth=100.0, oil=(0.0, 0.0, 0.0))
    # Saturation-weighted densities: 0.5*d + 0.3*60 + 0.2*10 for oil densities 40, 50, 60.
    # The 100-200 ft segment has the middle and bottom connections beneath it.
    static = [0.5 * oil + 0.3 * 60.0 + 0.2 * 10.0 for oil in (50.0, 60.0)]
    top_to_middle = sum(static) / 2
    assert pressures[1] == pytest.approx(1000.0 + top_to_middle * 100.0 * PSI_PER_LB_FT2)


def test_hydrostatic_pressure_is_independent_of_connection_order_and_rate_sign():
    forward = hydrostatic_pressures(reference_depth=100.0, oil=(10.0, 20.0, 30.0))
    reverse = hydrostatic_pressures(
        reference_depth=100.0, oil=(-30.0, -20.0, -10.0), depths=(400.0, 200.0, 100.0)
    )
    # Reversed list reverses which sample belongs to which depth, so only the depth order
    # and the rate sign are being exercised here: pressure still rises with depth.
    assert np.all(np.diff(forward) > 0.0)
    assert reverse[2] == pytest.approx(1000.0) and reverse[0] > reverse[1] > reverse[2]


def test_hydrostatic_mode_is_refused_with_mismatched_lengths():
    from bores.wells.hydraulics.hydrostatic import compute_perforation_pressures

    with pytest.raises(ValidationError, match="same length"):
        compute_perforation_pressures(
            reference_depth=0.0,
            reference_pressure=1000.0,
            representative_depths=np.array([1.0, 2.0]),
            connection_oil_rates=np.ones(3),
            connection_water_rates=np.zeros(3),
            connection_gas_rates=np.zeros(3),
            connection_fluids=compile_connection_fluids(
                connection_samples=hydrostatic_samples()[:3], dtype=np.float64
            ),
            unit_system=UnitSystem.FIELD,
        )


def test_a_vfp_well_needs_a_connection_pressure_mode_that_does_not_need_hydraulics(tmp_path):
    from bores.wells.resolution.spec import ConnectionPressureMode

    tables = VFPTables.from_deck(producer_table(tmp_path))
    with pytest.raises(ValidationError, match="HYDROSTATIC"):
        resolve_producer(vfp_tables=tables, vfp_number=3, mode=ConnectionPressureMode.HYDRAULIC)


def test_a_vfp_well_resolves_with_hydrostatic_connection_pressures(tmp_path):
    from bores.wells.hydraulics.hydrostatic import compute_perforation_pressures
    from bores.wells.resolution.spec import ConnectionPressureMode

    tables = VFPTables.from_deck(producer_table(tmp_path))
    workspace, samples = resolve_producer(
        vfp_tables=tables, vfp_number=3, mode=ConnectionPressureMode.HYDROSTATIC
    )
    uniform, _ = resolve_producer(
        vfp_tables=tables, vfp_number=3, mode=ConnectionPressureMode.UNIFORM_BHP
    )
    assert workspace.bhps[0] == pytest.approx(1000.0)  # BHP control, held at the reference depth
    assert np.allclose(uniform.connection_pressures, 1000.0)

    # Connections are at 25, 75 and 125 ft, with the reference depth at 75 ft.
    expected = compute_perforation_pressures(
        reference_depth=75.0,
        reference_pressure=1000.0,
        representative_depths=np.array([25.0, 75.0, 125.0]),
        connection_oil_rates=workspace.connection_oil_rates,
        connection_water_rates=workspace.connection_water_rates,
        connection_gas_rates=workspace.connection_gas_rates,
        connection_fluids=compile_connection_fluids(connection_samples=samples),
        unit_system=UnitSystem.FIELD,
    )
    assert workspace.connection_pressures == pytest.approx(expected, rel=1e-4)
    assert workspace.connection_pressures[0] < 1000.0 < workspace.connection_pressures[2]
    # The profile feeds back into the rates: a flat profile at the same BHP gives different ones.
    assert not np.allclose(workspace.connection_gas_rates, uniform.connection_gas_rates)


def walk(
    *,
    reference_depth,
    depths,
    oil,
    measured_depths=None,
    reference_measured_depth=None,
    dtype=np.float64,
):
    from bores.wells.hydraulics.base import PhaseValues, plan_connection_walk

    depth_array = np.array(depths, dtype=dtype)
    oil_array = np.array(oil, dtype=dtype)
    zeros = np.zeros_like(oil_array)
    planned = plan_connection_walk(
        reference_depth,
        reference_depth if reference_measured_depth is None else reference_measured_depth,
        depth_array,
        depth_array if measured_depths is None else np.array(measured_depths, dtype=dtype),
        oil_array,
        zeros,
        zeros,
    )
    return [
        types.SimpleNamespace(
            connection_index=int(planned.connection_indices[k]),
            previous_index=None
            if planned.previous_indices[k] < 0
            else int(planned.previous_indices[k]),
            vertical_distance=planned.vertical_distances[k],
            segment_length=planned.segment_lengths[k],
            geometric_sign=planned.geometric_signs[k],
            carried_rates=PhaseValues(
                oil=planned.carried_oil_rates[k],
                water=planned.carried_water_rates[k],
                gas=planned.carried_gas_rates[k],
            ),
        )
        for k in range(len(planned.connection_indices))
    ]


def test_walk_segments_carry_every_connection_beneath_them():
    # Connections at 50, 100, 200 and 400 ft with rates 1, 2, 4, 8 and the reference at 150 ft.
    steps = {
        step.connection_index: step
        for step in walk(
            reference_depth=150.0, depths=[50.0, 100.0, 200.0, 400.0], oil=[1, 2, 4, 8]
        )
    }
    # Downward: the 200 ft connection's segment carries it and the 400 ft one; then only 400 ft.
    assert steps[2].carried_rates.oil == pytest.approx(4 + 8)
    assert steps[3].carried_rates.oil == pytest.approx(8)
    assert steps[2].previous_index is None and steps[3].previous_index == 2
    assert steps[2].geometric_sign == pytest.approx(1.0)
    # Upward: from the reference to 100 ft the flow is everything at or below the reference;
    # past the 100 ft connection it has gained that connection's rate.
    assert steps[1].carried_rates.oil == pytest.approx(4 + 8)
    assert steps[0].carried_rates.oil == pytest.approx(2 + 4 + 8)
    assert steps[1].previous_index is None and steps[0].previous_index == 1
    assert steps[0].geometric_sign == pytest.approx(-1.0)
    assert steps[0].vertical_distance == pytest.approx(50.0)


def test_walk_with_every_connection_above_the_reference_carries_nothing_into_the_first_segment():
    steps = walk(reference_depth=500.0, depths=[100.0, 200.0], oil=[3, 5])
    first, second = steps  # nearest the reference first: 200 ft, then 100 ft
    assert first.connection_index == 1 and first.carried_rates.oil == pytest.approx(0.0)
    assert second.carried_rates.oil == pytest.approx(5.0)


def test_segment_length_is_the_distance_along_the_wellbore():
    # A 60 degree section: 100 ft of vertical distance is 200 ft along the hole.
    steps = walk(
        reference_depth=100.0,
        depths=[100.0, 200.0],
        measured_depths=[1000.0, 1200.0],
        oil=[1, 1],
    )
    assert steps[1].vertical_distance == pytest.approx(100.0)
    assert steps[1].segment_length == pytest.approx(200.0)


def test_a_horizontal_segment_has_length_but_no_vertical_extent():
    steps = walk(
        reference_depth=5000.0,
        depths=[5000.0, 5000.0, 5000.0],
        measured_depths=[8000.0, 8400.0, 8900.0],
        reference_measured_depth=7800.0,
        oil=[1, 1, 1],
    )
    by_connection = {step.connection_index: step for step in steps}
    assert all(step.vertical_distance == pytest.approx(0.0) for step in steps)
    assert by_connection[0].segment_length == pytest.approx(200.0)  # reference to the heel
    assert by_connection[1].segment_length == pytest.approx(400.0)
    assert by_connection[2].segment_length == pytest.approx(500.0)


def test_a_segment_is_never_shorter_than_its_vertical_extent():
    steps = walk(
        reference_depth=100.0, depths=[100.0, 200.0], measured_depths=[1000.0, 1050.0], oil=[1, 1]
    )
    assert steps[1].segment_length == pytest.approx(100.0)


def test_hydrostatic_head_of_an_inclined_segment_is_its_vertical_extent():
    from bores.wells.hydraulics.homogeneous import compute_segment_drop, homogeneous_wellbore

    model = homogeneous_wellbore(tubing_inner_diameter=0.25).options

    def hydrostatic(*, length, inclination):
        return compute_segment_drop(
            model=model,
            length=length,
            inclination_from_vertical=inclination,
            mixture_density=50.0,
            mixture_viscosity=1.0,
            mixture_velocity=1.0,
        ).hydrostatic

    import math

    vertical = hydrostatic(length=100.0, inclination=0.0)
    inclined = hydrostatic(length=200.0, inclination=math.radians(60.0))  # 100 ft vertical
    assert inclined == pytest.approx(vertical)


def correlation_pressures(
    *,
    is_injector,
    depths=(100.0, 200.0),
    reference_depth=150.0,
    measured_depths=None,
    reference_measured_depth=None,
    inclinations=None,
    **kwargs,
):
    from bores.wells.hydraulics.homogeneous import (
        compute_perforation_pressures,
        homogeneous_wellbore,
    )

    model = homogeneous_wellbore(tubing_inner_diameter=0.25).options
    count = len(depths)
    return compute_perforation_pressures(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=(
            reference_depth if reference_measured_depth is None else reference_measured_depth
        ),
        reference_pressure=1000.0,
        connection_oil_rates=np.full(count, 100.0),
        connection_water_rates=np.zeros(count),
        connection_gas_rates=np.zeros(count),
        representative_depths=np.array(depths),
        representative_measured_depths=np.array(
            depths if measured_depths is None else measured_depths
        ),
        inclinations_from_vertical=np.zeros(count)
        if inclinations is None
        else np.array(inclinations),
        connection_fluids=compile_connection_fluids(
            connection_samples=hydrostatic_samples()[:count], dtype=np.float64
        ),
        is_injector=is_injector,
        **kwargs,
    )


def test_friction_opposes_the_flow_on_both_sides_of_the_reference_depth():
    producer = correlation_pressures(is_injector=False)
    injector = correlation_pressures(is_injector=True)
    # Same carried rates and densities, so both have the same head; friction is what differs,
    # and it adds against the flow and subtracts along it. Connection 0 is above the reference
    # depth and connection 1 below it.
    friction_below = (producer[1] - injector[1]) / 2.0
    friction_above = (injector[0] - producer[0]) / 2.0
    assert friction_below > 0.0
    assert friction_above > 0.0


def test_correlation_pressure_rises_with_depth_when_the_reference_is_the_shallowest_depth():
    # There is no upward branch, so the downward walk is the whole well.
    for is_injector in (False, True):
        pressures = correlation_pressures(
            is_injector=is_injector, depths=(100.0, 200.0, 300.0), reference_depth=100.0
        )
        assert pressures[0] == pytest.approx(1000.0)
        assert np.all(np.diff(pressures) > 0.0)


def test_hydrostatic_pressures_keep_the_dtype_they_are_given():
    from bores.wells.hydraulics.hydrostatic import compute_perforation_pressures

    kwargs = dict(reference_depth=100.0, reference_pressure=1000.0, unit_system=UnitSystem.FIELD)
    depths = np.array([100.0, 200.0, 400.0], dtype=np.float32)
    rates = connection_rates(oil=(10.0, 20.0, 30.0), water=(0.0,) * 3, gas=(0.0,) * 3)
    single = compute_perforation_pressures(
        representative_depths=depths,
        connection_oil_rates=rates.oil.astype(np.float32),
        connection_water_rates=rates.water.astype(np.float32),
        connection_gas_rates=rates.gas.astype(np.float32),
        connection_fluids=compile_connection_fluids(
            connection_samples=hydrostatic_samples(), dtype=np.float32
        ),
        dtype=np.float32,
        **kwargs,
    )
    double = compute_perforation_pressures(
        representative_depths=depths.astype(np.float64),
        connection_oil_rates=rates.oil,
        connection_water_rates=rates.water,
        connection_gas_rates=rates.gas,
        connection_fluids=compile_connection_fluids(
            connection_samples=hydrostatic_samples(), dtype=np.float64
        ),
        dtype=np.float64,
        **kwargs,
    )
    assert single.dtype == np.float32 and double.dtype == np.float64
    assert single == pytest.approx(double, rel=1e-5)


WELLBORE_FACTORIES = [
    ("homogeneous", "homogeneous_wellbore"),
    ("beggs_and_brill", "beggs_and_brill_wellbore"),
    ("hagedorn_brown", "hagedorn_brown_wellbore"),
    ("gray", "gray_wellbore"),
    ("woldesemayat_ghajar", "woldesemayat_ghajar_wellbore"),
]


@pytest.mark.parametrize("module, factory", WELLBORE_FACTORIES)
def test_wellbore_options_hold_only_numbers_so_jitted_calls_stay_cheap(module, factory):
    import importlib

    make = getattr(importlib.import_module(f"bores.wells.hydraulics.{module}"), factory)
    options = make(tubing_inner_diameter=0.25).options
    assert all(isinstance(value, int | float) for value in options)


@pytest.mark.parametrize("module, factory", WELLBORE_FACTORIES)
@pytest.mark.parametrize("unit_system", list(UnitSystem))
def test_wellbore_unit_system_survives_the_numeric_tag_and_conversion(
    module, factory, unit_system
):
    import importlib

    make = getattr(importlib.import_module(f"bores.wells.hydraulics.{module}"), factory)
    model = make(tubing_inner_diameter=0.25, unit_system=unit_system)
    assert model.options.unit_system == unit_system
    converted = model.convert(UnitSystem.METRIC)
    assert converted.options.unit_system == UnitSystem.METRIC


def test_a_rate_target_is_met_when_the_pressure_profile_depends_on_every_phase():
    from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
    from bores.wells.resolution.spec import ConnectionPressureMode

    workspace, _ = resolve_producer(
        vfp_tables=homogeneous_wellbore(tubing_inner_diameter=0.25),
        target_rate=1000.0,
        mode=ConnectionPressureMode.HYDRAULIC,
    )
    assert workspace.surface_oil_rates[0] == pytest.approx(1000.0, rel=1e-3)
    # Water and gas flow too, so their rates are recorded at every connection.
    assert np.all(workspace.connection_water_rates > 0.0)
    assert np.all(workspace.connection_gas_rates > 0.0)


def test_a_rate_target_is_met_with_hydrostatic_connection_pressures(tmp_path):
    from bores.wells.resolution.spec import ConnectionPressureMode

    tables = VFPTables.from_deck(producer_table(tmp_path))
    workspace, _ = resolve_producer(
        vfp_tables=tables,
        vfp_number=3,
        target_rate=1000.0,
        mode=ConnectionPressureMode.HYDROSTATIC,
    )
    assert workspace.surface_oil_rates[0] == pytest.approx(1000.0, rel=1e-3)


def test_a_horizontal_section_loses_pressure_to_friction_along_the_wellbore():
    import math

    # Three connections at one true vertical depth, spread along 1000 ft of horizontal hole, with
    # the reference at the heel: no head differences, only friction.
    kwargs = dict(
        depths=(5000.0, 5000.0, 5000.0),
        measured_depths=(8200.0, 8700.0, 9200.0),
        reference_depth=5000.0,
        reference_measured_depth=8000.0,
        inclinations=[math.pi / 2] * 3,
    )
    producer = correlation_pressures(is_injector=False, **kwargs)
    injector = correlation_pressures(is_injector=True, **kwargs)
    # Against the flow of a producer, pressure rises moving away from the heel; along the flow
    # of an injector it falls.
    assert np.all(np.diff(producer) > 0.0) and producer[0] > 1000.0
    assert np.all(np.diff(injector) < 0.0) and injector[0] < 1000.0
    # Friction scales with the length it acts over: doubling every distance along the hole
    # doubles it.
    doubled = dict(kwargs)
    doubled["measured_depths"] = (8400.0, 9400.0, 10400.0)
    doubled["reference_measured_depth"] = 8000.0
    longer = correlation_pressures(is_injector=False, **doubled)
    assert (longer[2] - 1000.0) == pytest.approx(2.0 * (producer[2] - 1000.0), rel=1e-3)


def trajectory(*points):
    from bores.wells.trajectory import TrajectoryStation, WellTrajectory

    return WellTrajectory(
        stations=[TrajectoryStation(measured_depth=md, x=0.0, y=0.0, z=z) for md, z in points]
    )


def test_a_reference_depth_maps_to_the_shallowest_matching_measured_depth():
    # Straight down to 5000 ft, then 1000 ft horizontally.
    well_path = trajectory((0.0, 0.0), (5000.0, 5000.0), (6000.0, 5000.0))
    assert well_path.get_measured_depth_at(2500.0) == pytest.approx(2500.0)
    assert well_path.get_measured_depth_at(5000.0) == pytest.approx(5000.0)  # the heel


def test_a_reference_depth_follows_an_inclined_section():
    well_path = trajectory((0.0, 0.0), (1000.0, 1000.0), (2000.0, 1500.0))
    assert well_path.get_measured_depth_at(1250.0) == pytest.approx(1500.0)


def test_a_reference_depth_the_path_never_reaches_is_rejected():
    with pytest.raises(ValidationError, match="never reaches"):
        trajectory((0.0, 0.0), (1000.0, 1000.0)).get_measured_depth_at(2000.0)


def resolve_horizontal_producer(*, bhp, mobility=1e-4, max_fixed_point_iterations=None):
    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.types import Orientation
    from bores.wells.base import Perforation
    from bores.wells.compile import compile_well_system
    from bores.wells.factories import make_producer, make_well_controls, make_wells
    from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
    from bores.wells.resolution.engine import resolve_well_control
    from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
    from bores.wells.workspace import build_wells_workspace

    grid = make_cartesian_grid(nx=5, ny=1, nz=1, dx=100.0, dy=100.0, dz=20.0)
    well_path = trajectory((0.0, 0.0), (10.0, 10.0), (460.0, 10.0))
    well_path = type(well_path)(
        stations=[
            type(station)(
                measured_depth=station.measured_depth,
                x=50.0 + max(station.measured_depth - 10.0, 0.0),
                y=50.0,
                z=station.z,
            )
            for station in well_path.stations
        ]
    )
    well, control = make_producer(
        "H1",
        surface_location=(50.0, 50.0),
        trajectory=well_path,
        perforations=[Perforation(top_md=60.0, bottom_md=450.0)],
        target_bhp=bhp,
        reference_depth=10.0,
    )
    compiled = compile_well_system(
        make_wells(well),
        make_well_controls(H1=control),
        grid,
        permeabilities={orientation: np.full(5, 100.0) for orientation in Orientation},
    )
    count = len(compiled.perforations.well_indices)
    workspace = build_wells_workspace(n_wells=1, n_connections=count)
    resolve_well_control(
        compiled_system=compiled,
        well_row=0,
        tubing_performance=homogeneous_wellbore(tubing_inner_diameter=0.25),
        connection_fluids=compile_connection_fluids(
            connection_samples=hydrostatic_samples(
                oil_densities=(50.0,) * count, mobility=mobility
            )
        ),
        workspace=workspace,
        control_spec=WellControlSpec(
            connection_pressure_mode=ConnectionPressureMode.HYDRAULIC,
            **(
                {}
                if max_fixed_point_iterations is None
                else {"max_fixed_point_iterations": max_fixed_point_iterations}
            ),
        ),
    )
    return compiled, workspace


def test_a_horizontal_well_records_measured_depths_and_a_reference_at_its_heel():
    compiled, _ = resolve_horizontal_producer(bhp=1000.0)
    perforations = compiled.perforations
    assert np.all(np.diff(perforations.representative_measured_depths) > 0.0)
    assert np.allclose(perforations.representative_depths, 10.0)
    assert compiled.reference_measured_depths[0] == pytest.approx(10.0)


def test_a_horizontal_well_has_friction_pressure_loss_along_its_length():
    _, workspace = resolve_horizontal_producer(bhp=1000.0)
    pressures = workspace.connection_pressures
    # One true vertical depth, so there is no head: the whole difference is friction, and a
    # producer's pressure rises moving away from the heel against its flow.
    assert pressures[0] > 1000.0
    assert np.all(np.diff(pressures) > 0.0)


def test_a_well_whose_rates_never_settle_raises_instead_of_returning_a_state():
    from bores.errors import SolverError, WellConvergenceError

    # A mobility of 1 gives millions of rb/d; friction at that rate takes the connection
    # pressures past reservoir pressure, so the drawdown and the rate flip between two states.
    with pytest.raises(WellConvergenceError, match="Well 'H1'") as raised:
        resolve_horizontal_producer(bhp=1000.0, mobility=1.0)
    assert "did not converge in 20 iterations" in str(raised.value)
    assert isinstance(raised.value, SolverError)


def test_the_iteration_limit_is_enforced():
    from bores.errors import WellConvergenceError

    with pytest.raises(WellConvergenceError, match="did not converge in 1 iterations"):
        resolve_horizontal_producer(bhp=1000.0, max_fixed_point_iterations=1)
    resolve_horizontal_producer(bhp=1000.0, max_fixed_point_iterations=20)


def test_a_connection_with_nothing_beneath_it_carries_exactly_zero():
    # 0.1 + 0.2 + 0.3 - 0.1 - 0.2 - 0.3 is not exactly zero in floating point, so a total that is
    # subtracted from as the walk goes would leave a spurious rate on the deepest idle connection.
    steps = walk(reference_depth=0.0, depths=[10.0, 20.0, 30.0, 40.0], oil=[0.1, 0.2, 0.3, 0.0])
    deepest = next(step for step in steps if step.connection_index == 3)
    assert deepest.carried_rates.oil == pytest.approx(0.0, abs=0.0)
    assert tuple(deepest.carried_rates) == (0.0, 0.0, 0.0)


def test_a_reference_depth_in_another_precision_still_matches_a_connection_at_that_depth():
    # float32(152.4) is a little less than the float64 152.4. The reference is taken in the depth
    # arrays' data type, so the connection is at the reference depth, not above it.
    steps = walk(
        reference_depth=152.4,
        depths=[152.4, 152.4, 152.4],
        oil=[1, 1, 1],
        measured_depths=[274.32, 335.28, 396.24],
        reference_measured_depth=274.32,
        dtype=np.float32,
    )
    assert all(step.geometric_sign == pytest.approx(1.0) for step in steps)
    first = next(step for step in steps if step.connection_index == 0)
    assert first.vertical_distance == pytest.approx(0.0)
    assert first.segment_length == pytest.approx(0.0)


def scenario(*, unit_system, geometry, reference_kind, rate_kind, dtype):
    """Per-connection inputs for one well in `unit_system`, consistent in its units."""
    import math

    from bores.units import get_conversion_factors
    from bores.wells.hydraulics.base import compile_connection_fluids
    from bores.wells.state import ConnectionSample, PhaseValues

    count = 6
    rng = np.random.RandomState(7)
    if geometry == "vertical":
        depth = np.linspace(100.0, 600.0, count)
        measured, inclination = depth.copy(), np.zeros(count)
    elif geometry == "horizontal":
        depth = np.full(count, 500.0)
        measured, inclination = np.linspace(900.0, 1900.0, count), np.full(count, math.pi / 2)
    else:
        depth = np.array([300.0, 250.0, 320.0, 310.0, 280.0, 340.0])
        measured = np.array([1000.0, 1100.0, 1230.0, 1330.0, 1450.0, 1580.0])
        inclination = np.full(count, 1.3)
    reference_depth = {
        "top": depth.min(),
        "middle": np.median(depth) + 3.0,
        "below": depth.max() + 60.0,
    }[reference_kind]
    reference_measured = {
        "top": measured.min(),
        "middle": np.median(measured) + 3.0,
        "below": measured.max() + 60.0,
    }[reference_kind]
    oil, water, gas = (
        np.zeros(count) if rate_kind == "zero" else rng.uniform(50.0, 900.0, count)
        for _ in range(3)
    )
    raw = [
        ConnectionSample(
            cell_index=i,
            pressure=float(rng.uniform(1500, 3500)),
            temperature=200.0,
            phase_saturations=PhaseValues(0.5, 0.3, 0.2),
            phase_mobilities=PhaseValues(1.0, 1.0, 1.0),
            phase_densities=PhaseValues(
                float(rng.uniform(40, 55)), 62.0, float(rng.uniform(2, 12))
            ),
            phase_viscosities=PhaseValues(float(rng.uniform(0.3, 3)), 0.5, 0.02),
            gas_liquid_surface_tension=20.0,
            phase_fvfs=PhaseValues(1.2, 1.0, 0.01),
        )
        for i in range(count)
    ]
    factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
    tension = 1e-3 if unit_system is UnitSystem.SI else 1.0
    samples = [
        ConnectionSample(
            cell_index=s.cell_index,
            pressure=s.pressure * factors["pressure"],
            temperature=s.temperature,
            phase_saturations=s.phase_saturations,
            phase_mobilities=s.phase_mobilities,
            phase_densities=PhaseValues(*(v * factors["density"] for v in s.phase_densities)),
            phase_viscosities=PhaseValues(
                *(v * factors["viscosity"] for v in s.phase_viscosities)
            ),
            gas_liquid_surface_tension=s.gas_liquid_surface_tension * tension,
            phase_fvfs=s.phase_fvfs,
        )
        for s in raw
    ]
    rate_factor = factors["reservoir_rate"]
    length = factors["length"]
    return types.SimpleNamespace(
        reference_depth=float(reference_depth * length),
        reference_measured_depth=float(reference_measured * length),
        reference_pressure=1000.0 * factors["pressure"],
        depths=(depth * length).astype(dtype),
        measured_depths=(measured * length).astype(dtype),
        inclinations=inclination.astype(dtype),
        oil=(oil * rate_factor).astype(dtype),
        water=(water * rate_factor).astype(dtype),
        gas=(gas * rate_factor).astype(dtype),
        fluids=compile_connection_fluids(connection_samples=samples, dtype=dtype),
        samples=samples,
        diameter=0.25 * length,
    )


KERNELS = [
    ("homogeneous", "homogeneous_wellbore"),
    ("beggs_and_brill", "beggs_and_brill_wellbore"),
    ("hagedorn_brown", "hagedorn_brown_wellbore"),
    ("gray", "gray_wellbore"),
    ("woldesemayat_ghajar", "woldesemayat_ghajar_wellbore"),
]


@pytest.mark.parametrize("module, factory", KERNELS)
@pytest.mark.parametrize("unit_system", list(UnitSystem))
@pytest.mark.parametrize("geometry", ["vertical", "horizontal", "undulating"])
@pytest.mark.parametrize("rate_kind", ["zero", "flowing"])
@pytest.mark.parametrize("is_injector", [False, True])
def test_the_compiled_walk_matches_the_same_code_run_as_python(
    module, factory, unit_system, geometry, rate_kind, is_injector
):
    import importlib

    from bores.wells.hydraulics.base import get_hydrostatic_divisor

    case = scenario(
        unit_system=unit_system,
        geometry=geometry,
        reference_kind="middle",
        rate_kind=rate_kind,
        dtype=np.float64,
    )
    target = importlib.import_module(f"bores.wells.hydraulics.{module}")
    model = getattr(target, factory)(
        tubing_inner_diameter=case.diameter, unit_system=unit_system
    ).options
    arguments = (
        model,
        case.reference_depth,
        case.reference_measured_depth,
        case.reference_pressure,
        case.oil,
        case.water,
        case.gas,
        case.depths,
        case.measured_depths,
        case.inclinations,
        case.fluids,
        is_injector,
        get_hydrostatic_divisor(unit_system),
    )
    compiled, interpreted = np.zeros(6), np.zeros(6)
    target.walk_perforation_pressures(*arguments, compiled)
    target.walk_perforation_pressures.py_func(*arguments, interpreted)
    assert compiled == pytest.approx(interpreted, rel=1e-10)


@pytest.mark.parametrize("unit_system", list(UnitSystem))
def test_a_static_column_follows_the_unit_systems_own_constants(unit_system):
    from bores.wells.hydraulics.base import (
        compute_hydrostatic_pressure,
        compute_static_mixture_density,
        get_gravitational_acceleration,
    )
    from bores.wells.hydraulics.homogeneous import (
        compute_perforation_pressures,
        homogeneous_wellbore,
    )

    case = scenario(
        unit_system=unit_system,
        geometry="vertical",
        reference_kind="top",
        rate_kind="zero",
        dtype=np.float64,
    )
    model = homogeneous_wellbore(tubing_inner_diameter=case.diameter, unit_system=unit_system)
    pressures = compute_perforation_pressures(
        model.options,
        case.reference_depth,
        case.reference_measured_depth,
        case.reference_pressure,
        case.oil,
        case.water,
        case.gas,
        case.depths,
        case.measured_depths,
        case.inclinations,
        case.fluids,
        False,
    )
    expected = case.reference_pressure
    previous_depth = case.reference_depth
    for i in range(1, 6):
        sample = case.samples[i]
        expected += compute_hydrostatic_pressure(
            density=compute_static_mixture_density(
                phase_saturations=sample.phase_saturations, phase_densities=sample.phase_densities
            ),
            gravitational_acceleration=get_gravitational_acceleration(unit_system),
            length=case.depths[i] - case.depths[i - 1]
            if i > 1
            else case.depths[i] - previous_depth,
            unit_system=unit_system,
        )
        assert pressures[i] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("module, factory", KERNELS)
def test_the_walk_keeps_the_precision_it_is_given(module, factory):
    import importlib

    target = importlib.import_module(f"bores.wells.hydraulics.{module}")
    results = {}
    for dtype in (np.float32, np.float64):
        case = scenario(
            unit_system=UnitSystem.FIELD,
            geometry="vertical",
            reference_kind="middle",
            rate_kind="flowing",
            dtype=dtype,
        )
        model = getattr(target, factory)(tubing_inner_diameter=case.diameter).options
        results[dtype] = target.compute_perforation_pressures(
            model,
            dtype(case.reference_depth),
            dtype(case.reference_measured_depth),
            dtype(case.reference_pressure),
            case.oil,
            case.water,
            case.gas,
            case.depths,
            case.measured_depths,
            case.inclinations,
            case.fluids,
            False,
            dtype=dtype,
        )
    assert results[np.float32].dtype == np.float32
    assert results[np.float64].dtype == np.float64
    assert results[np.float32] == pytest.approx(results[np.float64], rel=1e-4)


def test_per_connection_inputs_of_different_lengths_are_rejected():
    from bores.wells.hydraulics.homogeneous import (
        compute_perforation_pressures,
        homogeneous_wellbore,
    )

    case = scenario(
        unit_system=UnitSystem.FIELD,
        geometry="vertical",
        reference_kind="top",
        rate_kind="flowing",
        dtype=np.float64,
    )
    model = homogeneous_wellbore(tubing_inner_diameter=case.diameter).options
    with pytest.raises(ValueError, match="same length"):
        compute_perforation_pressures(
            model,
            case.reference_depth,
            case.reference_measured_depth,
            case.reference_pressure,
            case.oil[:-1],
            case.water,
            case.gas,
            case.depths,
            case.measured_depths,
            case.inclinations,
            case.fluids,
            False,
        )


def test_region_in_place_uses_the_pore_volume_at_the_current_pressure(monkeypatch):
    workspace = region_workspace()
    workspace.reservoir.pressure = np.full(4, 4000.0)
    monkeypatch.setattr(well_summary, "get_workspace", lambda *, context: workspace)
    reference = well_summary.RegionWaterInPlace(region=1)(region_model(None), None)
    compressed = 0.3 * 1000 + 0.3 * 2000 + 0.3 * 3000 + 0.3 * 4000
    assert reference == pytest.approx(compressed * 1.01)


def well_vector_context():
    from bores.schedule.base import ScheduleContext
    from bores.simulation.workspace import SimulationWorkspace
    from bores.wells.compile import WellKind
    from bores.wells.workspace import accumulate_well_volumes, build_wells_workspace

    rows = {"P1": 0, "I1": 1}
    wells = types.SimpleNamespace(
        names=("P1", "I1"),
        well_kinds=np.array([WellKind.PRODUCER, WellKind.INJECTOR], dtype=np.int32),
        well_row=lambda *, name: rows[name],
    )
    workspace = build_wells_workspace(n_wells=2, n_connections=2)
    workspace.surface_oil_rates[:] = [1000.0, 0.0]
    workspace.surface_water_rates[:] = [400.0, 1200.0]
    workspace.surface_gas_rates[:] = [5e5, 0.0]
    accumulate_well_volumes(workspace=workspace, step_size=10.0)
    context = ScheduleContext(
        time=10.0,
        extra={"workspace": SimulationWorkspace(None, None, None, workspace, None, None, None)},
    )
    return types.SimpleNamespace(wells=wells, unit_system=UnitSystem.FIELD), context


@pytest.mark.parametrize(
    ("vector", "well_name", "expected"),
    [
        (well_summary.WellWaterProductionRate, "P1", 400.0),
        (well_summary.WellWaterProductionRate, "I1", 0.0),
        (well_summary.WellWaterInjectionRate, "I1", 1200.0),
        (well_summary.WellWaterInjectionRate, "P1", 0.0),
        (well_summary.WellWaterProductionTotal, "I1", 0.0),
        (well_summary.WellWaterInjectionTotal, "P1", 0.0),
        (well_summary.WellWaterInjectionTotal, "I1", 12000.0),
        (well_summary.WellWaterCut, "I1", 0.0),
    ],
)
def test_per_well_vectors_only_report_wells_of_their_own_kind(vector, well_name, expected):
    model, context = well_vector_context()
    assert vector(well_name=well_name)(model, context) == pytest.approx(expected)


def test_cumulative_well_volumes_are_double_precision():
    from bores.wells.workspace import build_wells_workspace

    workspace = build_wells_workspace(n_wells=1, n_connections=1)
    assert workspace.cumulative_gas_volumes.dtype == np.float64


def test_interval_event_does_not_miss_a_boundary_to_floating_point_error():
    from bores.schedule.base import ScheduleContext
    from bores.schedule.events import IntervalEvent

    event = IntervalEvent(every=0.1)
    time, fired = 0.0, []
    for _ in range(10):
        previous, time = time, time + 0.1
        fired.append(bool(event(None, ScheduleContext(time=time, previous_time=previous))))
    assert all(fired)
    with pytest.raises(ValueError):
        IntervalEvent(every=0.0)


def test_default_record_item_names_tell_different_quantities_apart():
    from bores.schedule import record_every

    first = types.SimpleNamespace(key="FOPR")
    second = types.SimpleNamespace(key="FWPR")
    assert (
        record_every(every=30, quantities=[first]).name
        != record_every(every=30, quantities=[second]).name
    )


def control_model():
    rows = {"P1": 0, "I1": 1, "P2": 2}

    def well_row(*, name):
        return [rows[one] for one in name] if isinstance(name, (list, tuple)) else rows[name]

    return types.SimpleNamespace(
        wells=types.SimpleNamespace(controls=build_compiled_controls(), well_row=well_row)
    )


def test_control_mode_of_the_wrong_kind_is_rejected_before_anything_changes():
    from bores.wells.controls import ProducerControlMode
    from bores.wells.schedule import SetWellControls

    model = control_model()
    before = model.wells.controls.control_modes.copy()
    with pytest.raises(ValidationError):
        SetWellControls(well_names=("I1",), mode=ProducerControlMode.OIL_RATE)(model, None)
    with pytest.raises(ValidationError):
        SetWellControls(well_names=("P1", "I1"), mode=ProducerControlMode.OIL_RATE)(model, None)
    assert model.wells.controls.control_modes.tolist() == before.tolist()


def test_set_well_target_without_a_needed_value_changes_nothing():
    from bores.wells.controls import WellTargetMode
    from bores.wells.schedule import SetWellTargets

    model = control_model()
    before = model.wells.controls.control_modes.copy()
    with pytest.raises(ValidationError):
        SetWellTargets(well_names=("P1",), control_mode=WellTargetMode.OIL_RATE)(model, None)
    assert model.wells.controls.control_modes.tolist() == before.tolist()


def test_negative_vfp_table_number_is_rejected():
    from bores.wells.schedule import SetWellLift

    with pytest.raises(ValidationError):
        SetWellLift(well_name="P1", vfp_table=-5)(lift_model(), None)


def connection_target(**target):
    from bores.datastructures import GridDimensions
    from bores.wells.schedule import get_matching_connection_rows

    dims = GridDimensions(nx=5, ny=5, nz=3)
    cells = np.array([dims.flat_index(2, 3, layer) for layer in range(3)])
    perforations = types.SimpleNamespace(
        connection_rows=lambda *, well_row: range(0, 3), get_cell_index=lambda *, row: cells[row]
    )
    grid = types.SimpleNamespace(dimensions=dims)
    return get_matching_connection_rows(
        perforations=perforations, well_row=0, grid=grid, well_name="P1", **target
    ).tolist()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ({"i": 3, "j": 4, "k1": 1, "k2": 3}, [0, 1, 2]),
        ({"i": 3, "j": 4, "k1": 2, "k2": 2}, [1]),
        ({"i": 3, "j": 4, "k1": 0, "k2": 0}, [0, 1, 2]),
        ({"i": 0, "j": 4, "k1": 1, "k2": 2}, [0, 1]),
        ({"i": 0, "j": 0, "k1": 3, "k2": 3}, [2]),
    ],
)
def test_connection_targets_treat_zero_as_any(target, expected):
    assert connection_target(**target) == expected


@pytest.mark.parametrize(
    "target",
    [
        {"i": 9, "j": 9, "k1": 1, "k2": 1},
        {"i": 3, "j": 4, "k1": 3, "k2": 1},
        {"i": 3, "j": 4, "k1": 0, "k2": 2},
    ],
)
def test_connection_target_that_selects_nothing_or_is_malformed_is_an_error(target):
    with pytest.raises(ValidationError):
        connection_target(**target)


def test_welopen_with_defaulted_layers_or_no_match_in_the_deck_loader():
    from bores.datastructures import GridDimensions
    from bores.wells.schedule import match_connection_target

    # specs span layers, the same matcher the runtime actions use
    spec = {"connection_i": 3, "connection_j": 4, "connection_k1": 1, "connection_k2": 3}
    assert match_connection_target(i=3, j=4, k1=0, k2=0, **spec)
    assert match_connection_target(i=0, j=0, k1=2, k2=2, **spec)
    assert not match_connection_target(i=3, j=5, k1=0, k2=0, **spec)
    assert not match_connection_target(i=3, j=4, k1=4, k2=5, **spec)
    with pytest.raises(ValidationError):
        match_connection_target(i=3, j=4, k1=2, k2=0, **spec)
    assert GridDimensions(nx=1, ny=1, nz=1)


def welopen_well(welopen_records):
    from bores.datastructures import GridDimensions
    from bores.wells.deck import load_well_from_records

    dims = GridDimensions(nx=3, ny=3, nz=3)
    cell_count = dims.nx * dims.ny * dims.nz
    depths = np.zeros((cell_count, 3))
    depths[:, 2] = np.arange(cell_count, dtype=float)
    grid = types.SimpleNamespace(
        dimensions=dims,
        cell_min_xyz=depths,
        cell_max_xyz=depths + 1.0,
        get_cell_center_at=lambda i, j, k: np.array([0.0, 0.0, 0.0]),
    )
    welspecs = {"well": "P1", "inflow_equation": "STD", "i": 2, "j": 2}
    compdat = [
        {"well": "P1", "i": 2, "j": 2, "k1": layer, "k2": layer, "status": "OPEN", "diameter": 0.5}
        for layer in (1, 2, 3)
    ]
    return load_well_from_records(grid, welspecs, compdat, welopen_records=welopen_records)


def test_deck_welopen_shuts_only_the_connections_it_targets():
    from bores.wells.base import CompletionStatus

    well = welopen_well([{"status": "SHUT", "i": 2, "j": 2, "k1": 2, "k2": 3}])
    assert [perforation.status for perforation in well.perforations] == [
        CompletionStatus.OPEN,
        CompletionStatus.SHUT,
        CompletionStatus.SHUT,
    ]
    every_layer = welopen_well([{"status": "SHUT", "i": 2, "j": 2, "k1": 0, "k2": 0}])
    assert all(
        perforation.status == CompletionStatus.SHUT for perforation in every_layer.perforations
    )
    # Naming no connection sets the well's own flag, and leaves its connections as they were
    whole_well = welopen_well([{"status": "SHUT"}])
    assert whole_well.open_status == CompletionStatus.SHUT
    assert all(
        perforation.status == CompletionStatus.OPEN for perforation in whole_well.perforations
    )


def test_deck_welopen_that_matches_no_connection_is_an_error():
    with pytest.raises(ValidationError, match="matches no connection"):
        welopen_well([{"status": "SHUT", "i": 1, "j": 1, "k1": 1, "k2": 1}])


def test_deck_welopen_no_match_error_names_the_layers_of_the_welopen_record():
    # The layers named are the `WELOPEN`'s own, not those of the last `COMPDAT` of the well
    with pytest.raises(ValidationError, match=r"i=1, j=1, k1=1, k2=1 "):
        welopen_well([{"status": "SHUT", "i": 1, "j": 1, "k1": 1, "k2": 1}])


class FiringProbe:
    """Counts how many times it has run."""

    def __init__(self):
        self.runs = 0

    def __call__(self, model, context):
        self.runs += 1
        return model


def always_high_threshold():
    from bores.schedule import ComparisonOperator, ThresholdEvent

    class Always(ThresholdEvent):
        def get_value(self, *, model, context):
            return 10.0

    return Always(op=ComparisonOperator.GT, threshold=1.0)


def run_checks(schedule, *, times, firings):
    for time in times:
        schedule.advance(None, time=time, firings=firings)


def test_a_threshold_item_runs_its_action_once_by_default():
    from bores.schedule import Firings, Schedule, ScheduleItem

    probe, firings = FiringProbe(), Firings()
    item = ScheduleItem(event=always_high_threshold(), action=probe, name="cut")
    run_checks(Schedule(items=(item,)), times=[1.0, 2.0, 3.0, 4.0], firings=firings)
    assert probe.runs == 1
    assert firings.count("cut") == 1
    assert firings.last_time("cut") == pytest.approx(1.0)


def test_an_item_can_be_told_to_repeat_or_to_wait_between_runs():
    import math

    from bores.schedule import Firings, Schedule, ScheduleItem

    probe, firings = FiringProbe(), Firings()
    repeat = ScheduleItem(
        event=always_high_threshold(), action=probe, name="repeat", max_firings=math.inf
    )
    run_checks(Schedule(items=(repeat,)), times=[1.0, 2.0, 3.0], firings=firings)
    assert probe.runs == 3

    spaced_probe, spaced_firings = FiringProbe(), Firings()
    spaced = ScheduleItem(
        event=always_high_threshold(),
        action=spaced_probe,
        name="spaced",
        max_firings=math.inf,
        min_interval=5.0,
    )
    run_checks(
        Schedule(items=(spaced,)), times=[1.0, 2.0, 5.0, 6.0, 7.0, 11.0], firings=spaced_firings
    )
    assert spaced_probe.runs == 3  # at 1.0, 6.0 and 11.0


def test_events_that_fire_once_by_nature_are_not_limited():
    from bores.schedule import AllOf, ScheduleItem, TimeEvent

    plain = ScheduleItem(event=TimeEvent(at=5.0), action=FiringProbe())
    assert not plain.is_limited
    wrapped = ScheduleItem(
        event=AllOf(events=(TimeEvent(at=5.0), always_high_threshold())), action=FiringProbe()
    )
    assert wrapped.max_firings == 1


def test_a_limited_item_needs_firings_and_a_distinct_name():
    from bores.errors import ValidationError
    from bores.schedule import Schedule, ScheduleItem

    item = ScheduleItem(event=always_high_threshold(), action=FiringProbe(), name="cut")
    with pytest.raises(ValidationError, match="firings"):
        Schedule(items=(item,)).advance(None, time=1.0)
    with pytest.raises(ValidationError, match="distinct names"):
        Schedule(items=(item, item))


def test_firings_survive_a_save_and_load():
    from bores.schedule import Firings

    firings = Firings()
    firings.record(name="cut", time=3.0)
    firings.record(name="cut", time=8.0)
    restored = Firings.load(firings.dump())
    assert restored.count("cut") == 2
    assert restored.last_time("cut") == pytest.approx(8.0)


def changed_compiled_system():
    """A compiled system whose controls, limits, connections and group limits have all been changed."""
    import attrs

    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.schedule.base import ScheduleContext
    from bores.types import FluidPhase, Orientation
    from bores.wells.base import CompletionStatus
    from bores.wells.compile import LimitKind, compile_well_system
    from bores.wells.controls import (
        BHPLimit,
        EconomicLimit,
        EconomicQuantity,
        ProducerControlMode,
    )
    from bores.wells.factories import (
        make_group_control,
        make_injector,
        make_producer,
        make_well_controls,
        make_wells,
    )
    from bores.wells.groups import GroupControls, WellGroup, WellGroups
    from bores.wells.schedule import (
        MultiplyConnectionFactors,
        SetGroupLimit,
        SetLimits,
        SetWellControls,
        SetWellLift,
        UpdateWellStatuses,
    )

    grid = make_cartesian_grid(nx=2, ny=1, nz=3, dx=50.0, dy=50.0, dz=50.0)
    producer, producer_control = make_producer(
        "P1",
        surface_location=(25.0, 25.0),
        perforation_depths=(0.0, 150.0),
        target_rate=500.0,
        limits=[
            BHPLimit(min_value=500.0),
            EconomicLimit(quantity=EconomicQuantity.WATER_CUT, max_value=0.9),
        ],
        group="G1",
    )
    injector, injector_control = make_injector(
        "I1",
        injected_phase=FluidPhase.WATER,
        surface_location=(75.0, 25.0),
        perforation_depths=(0.0, 150.0),
        target_rate=800.0,
        group="G1",
    )
    wells = make_wells(producer, injector)
    group_controls = GroupControls(
        controls={
            "G1": attrs.evolve(
                make_group_control(target_rate=1000.0),
                limits=(EconomicLimit(quantity=EconomicQuantity.OIL_RATE, min_value=50.0),),
            )
        },
        unit_system=wells.unit_system,
    )
    compiled = compile_well_system(
        wells,
        make_well_controls(P1=producer_control, I1=injector_control),
        grid,
        permeabilities={orientation: np.full(6, 100.0) for orientation in Orientation},
        group_controls=group_controls,
        groups=WellGroups({"G1": WellGroup(name="G1", parent="FIELD")}),
    )
    model = types.SimpleNamespace(wells=compiled, reservoir=types.SimpleNamespace(grid=grid))
    context = ScheduleContext(time=1.0)
    for action in (
        MultiplyConnectionFactors(well_names=("P1",), multiplier=0.5),
        MultiplyConnectionFactors(well_names=("I1",), multiplier=2.0, i=2, j=1, k1=2, k2=2),
        UpdateWellStatuses(well_names=("P1",), status=CompletionStatus.SHUT, i=1, j=1, k1=2, k2=2),
        SetWellControls(
            well_names=("P1",),
            mode=ProducerControlMode.BHP,
            target_bhp=900.0,
            efficiency_factor=0.8,
            guide_rate=3.0,
        ),
        SetLimits(
            well_names=("P1", "P1"),
            kinds=(LimitKind.BHP, LimitKind.ECONOMIC),
            quantities=(None, EconomicQuantity.WATER_CUT),
            min_values=(600.0, None),
            max_values=(None, 0.7),
        ),
        SetGroupLimit(group_name="G1", quantity=EconomicQuantity.OIL_RATE, min_value=75.0),
        SetWellLift(well_name="P1", artificial_lift_quantity=12.0),
    ):
        action(model, context)
    return wells, compiled


def test_decompile_shows_every_change_made_to_the_controls_limits_and_groups():
    from bores.wells.controls import EconomicQuantity, ProducerControlMode
    from bores.wells.decompile import decompile_group_controls, decompile_well_controls

    _, compiled = changed_compiled_system()
    control = decompile_well_controls(compiled)["P1"]
    assert control.mode == ProducerControlMode.BHP
    assert control.target_bhp == pytest.approx(900.0)
    assert control.efficiency_factor == pytest.approx(0.8)
    assert control.guide_rate == pytest.approx(3.0)
    assert control.artificial_lift_quantity == pytest.approx(12.0)
    bhp = next(limit for limit in control.limits if type(limit).__name__ == "BHPLimit")
    economic = next(limit for limit in control.limits if type(limit).__name__ == "EconomicLimit")
    assert bhp.min_value == pytest.approx(600.0)
    assert economic.max_value == pytest.approx(0.7)

    group = decompile_group_controls(compiled.group_controls, compiled.unit_system)["G1"]
    (group_limit,) = group.limits
    assert group_limit.quantity == EconomicQuantity.OIL_RATE
    assert group_limit.min_value == pytest.approx(75.0)


def test_decompiled_connections_show_their_own_status_and_current_well_index():
    from bores.wells.base import CompletionStatus
    from bores.wells.decompile import decompile_perforations

    wells, compiled = changed_compiled_system()
    producer_rows = decompile_perforations(wells, "P1", compiled.perforations, 0)
    assert [row.status for row in producer_rows] == [
        CompletionStatus.OPEN,
        CompletionStatus.SHUT,
        CompletionStatus.OPEN,
    ]
    assert [row.connection_factor_override for row in producer_rows] == pytest.approx(
        compiled.perforations.well_indices[:3].tolist()
    )
    injector_rows = decompile_perforations(wells, "I1", compiled.perforations, 1)
    middle, outer = injector_rows[1], injector_rows[0]
    assert middle.connection_factor_override == pytest.approx(
        2.0 * outer.connection_factor_override
    )
    assert all(row.connection_factor_multiplier is None for row in injector_rows)


def test_a_connection_factor_multiplier_also_scales_a_connection_factor_override():
    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.types import Orientation
    from bores.wells.compile import compile_well_system
    from bores.wells.factories import make_producer, make_well_controls, make_wells

    grid = make_cartesian_grid(nx=1, ny=1, nz=1, dx=50.0, dy=50.0, dz=50.0)
    well, control = make_producer(
        "P1",
        surface_location=(25.0, 25.0),
        perforation_depths=(0.0, 50.0),
        target_bhp=1000.0,
        connection_factor_override=10.0,
        connection_factor_multiplier=0.5,
    )
    compiled = compile_well_system(
        make_wells(well),
        make_well_controls(P1=control),
        grid,
        permeabilities={orientation: np.full(1, 100.0) for orientation in Orientation},
    )
    assert compiled.perforations.well_indices[0] == pytest.approx(5.0)


def wpimult_perforations(wpimult_records, *, current_time):
    from bores.datastructures import GridDimensions
    from bores.wells.deck import load_well_from_records

    dims = GridDimensions(nx=3, ny=3, nz=3)
    cell_count = dims.nx * dims.ny * dims.nz
    depths = np.zeros((cell_count, 3))
    depths[:, 2] = np.arange(cell_count, dtype=float)
    grid = types.SimpleNamespace(
        dimensions=dims,
        cell_min_xyz=depths,
        cell_max_xyz=depths + 1.0,
        get_cell_center_at=lambda i, j, k: np.array([0.0, 0.0, 0.0]),
    )
    welspecs = {"well": "P1", "inflow_equation": "STD", "i": 2, "j": 2}
    compdat = [
        {"well": "P1", "i": 2, "j": 2, "k1": layer, "k2": layer, "status": "OPEN", "diameter": 0.5}
        for layer in (1, 2, 3)
    ]
    well = load_well_from_records(
        grid, welspecs, compdat, wpimult_records=wpimult_records, current_time=current_time
    )
    return [perforation.connection_factor_multiplier for perforation in well.perforations]


def test_deck_wpimult_accumulates_and_only_applies_up_to_the_load_time():
    records = [
        {"well": "P1", "multiplier": 0.5, "schedule_time": 0.0},
        {"well": "P1", "multiplier": 0.5, "i": 2, "j": 2, "k1": 2, "k2": 2, "schedule_time": 5.0},
        {"well": "P1", "multiplier": 3.0, "schedule_time": 10.0},
    ]
    assert wpimult_perforations(records, current_time=7.0) == pytest.approx([0.5, 0.25, 0.5])
    assert wpimult_perforations(records, current_time=10.0) == pytest.approx([1.5, 0.75, 1.5])
    assert wpimult_perforations(records, current_time=-1.0) == [None, None, None]


def test_deck_wpimult_that_matches_no_connection_is_an_error():
    records = [
        {"well": "P1", "multiplier": 2.0, "i": 1, "j": 1, "k1": 1, "k2": 1, "schedule_time": 0.0}
    ]
    with pytest.raises(ValidationError, match="matches no connection"):
        wpimult_perforations(records, current_time=1.0)


def test_deck_wpimult_no_match_error_names_the_layers_of_the_wpimult_record():
    records = [
        {"well": "P1", "multiplier": 2.0, "i": 1, "j": 1, "k1": 1, "k2": 1, "schedule_time": 0.0}
    ]
    with pytest.raises(ValidationError, match=r"i=1, j=1, k1=1, k2=1 "):
        wpimult_perforations(records, current_time=1.0)


def build_case_workspace(
    *, unit_system, geometry="vertical", mobility=1e-4, d_factor=float("nan")
):
    """A real perforation workspace for six connections, with rates of an ordinary size."""
    from bores.units import get_conversion_factors
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.state import ConnectionSample
    from bores.wells.workspace import build_perforation_workspace

    case = scenario(
        unit_system=unit_system,
        geometry=geometry,
        reference_kind="middle",
        rate_kind="zero",
        dtype=np.float64,
    )
    from bores.wells.resolution.spec import WellControlSpec

    factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
    # The rate is well index * mobility * drawdown * the radial flow constant, so scaling the well
    # index by the other factors keeps the rate the same size, in the unit system's own units, as
    # it is in field units.
    well_index_scale = (
        factors["reservoir_rate"]
        * factors["viscosity"]
        * WellControlSpec(unit_system=UnitSystem.FIELD).radial_flow_constant
        / (WellControlSpec(unit_system=unit_system).radial_flow_constant * factors["pressure"])
    )
    samples = [
        ConnectionSample(
            cell_index=s.cell_index,
            pressure=s.pressure,
            temperature=s.temperature,
            phase_saturations=s.phase_saturations,
            phase_mobilities=PhaseValues(*([mobility / factors["viscosity"]] * 3)),
            phase_densities=s.phase_densities,
            phase_viscosities=s.phase_viscosities,
            gas_liquid_surface_tension=s.gas_liquid_surface_tension,
            phase_fvfs=s.phase_fvfs,
        )
        for s in case.samples
    ]
    workspace = build_perforation_workspace(
        well_indices=np.full(6, 400.0 * well_index_scale),
        representative_depths=case.depths,
        representative_measured_depths=case.measured_depths,
        inclinations_from_vertical=case.inclinations,
        connection_fluids=compile_connection_fluids(connection_samples=samples, dtype=np.float64),
        reference_measured_depth=case.reference_measured_depth,
        connection_conductivities=np.full(6, 4000.0 * well_index_scale),
        d_factor=d_factor / factors["reservoir_rate"],
        dtype=np.float64,
    )
    return case, workspace


def copy_workspace(workspace):
    """A deep copy of a perforation workspace, since solving updates its arrays in place."""
    import copy

    return copy.deepcopy(workspace)


def solver_arguments(solver_name, case, workspace, unit_system):
    from bores.units import get_conversion_factors
    from bores.wells.hydraulics.base import get_hydrostatic_divisor
    from bores.wells.hydraulics.hydrostatic import get_pressure_per_density_and_depth
    from bores.wells.resolution.spec import WellControlSpec

    spec = WellControlSpec(unit_system=unit_system)
    pressure = 2000.0 * get_conversion_factors(UnitSystem.FIELD, unit_system)["pressure"]
    common = (True, True, True, False)
    tail = (spec.radial_flow_constant, 20, 1e-6)
    if solver_name == "uniform":
        return (pressure, workspace, *common, *tail)
    if solver_name == "hydrostatic":
        return (
            case.reference_depth,
            pressure,
            workspace,
            *common,
            get_pressure_per_density_and_depth(unit_system),
            *tail,
        )
    import importlib

    factory = dict(KERNELS)[solver_name]
    options = getattr(importlib.import_module(f"bores.wells.hydraulics.{solver_name}"), factory)(
        tubing_inner_diameter=case.diameter, unit_system=unit_system
    ).options
    return (
        options,
        case.reference_depth,
        pressure,
        workspace,
        *common,
        get_hydrostatic_divisor(unit_system),
        *tail,
    )


FIXED_POINT_DRIVERS = ["uniform", "hydrostatic", *[name for name, _ in KERNELS]]


@pytest.mark.parametrize("solver_name", FIXED_POINT_DRIVERS)
@pytest.mark.parametrize("unit_system", list(UnitSystem))
@pytest.mark.parametrize("d_factor", [float("nan"), 0.002])
def test_each_compiled_solver_matches_the_same_code_run_as_python(
    solver_name, unit_system, d_factor
):
    from bores.wells.resolution import fixed_point

    driver = {
        "uniform": fixed_point.solve_uniform,
        "hydrostatic": fixed_point.solve_hydrostatic,
        "homogeneous": fixed_point.solve_homogeneous,
        "beggs_and_brill": fixed_point.solve_beggs_and_brill,
        "hagedorn_brown": fixed_point.solve_hagedorn_brown,
        "gray": fixed_point.solve_gray,
        "woldesemayat_ghajar": fixed_point.solve_woldesemayat_ghajar,
    }[solver_name]
    case, workspace = build_case_workspace(unit_system=unit_system, d_factor=d_factor)
    compiled_workspace, python_workspace = copy_workspace(workspace), copy_workspace(workspace)
    compiled = driver(*solver_arguments(solver_name, case, compiled_workspace, unit_system))
    interpreted = driver.py_func(
        *solver_arguments(solver_name, case, python_workspace, unit_system)
    )
    assert compiled.converged and interpreted.converged
    assert tuple(compiled) == pytest.approx(tuple(interpreted), rel=1e-9)
    for name in (
        "connection_pressures",
        "connection_oil_rates",
        "connection_water_rates",
        "connection_gas_rates",
        "well_indices",
    ):
        assert getattr(compiled_workspace, name) == pytest.approx(
            getattr(python_workspace, name), rel=1e-9
        )
    assert compiled.oil_rate > 0.0


def test_a_total_rate_only_settles_when_it_is_finite():
    from bores.wells.resolution.fixed_point import has_settled

    assert has_settled(100.0, 100.0, 1e-6)
    assert has_settled(100.0, 100.0000001, 1e-6)
    assert not has_settled(100.0, 120.0, 1e-6)
    assert not has_settled(float("inf"), float("inf"), 1e-6)
    assert not has_settled(float("inf"), 0.0, 1e-6)
    assert not has_settled(float("nan"), 0.0, 1e-6)


def test_a_non_finite_rate_raises_instead_of_being_reported_as_converged():
    from bores.errors import WellConvergenceError
    from bores.wells.resolution.solvers import solve_connection_pressures_and_rates
    from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
    from bores.wells.resolution.tubing import WellTubing

    case, workspace = build_case_workspace(unit_system=UnitSystem.FIELD, mobility=float("inf"))
    with pytest.raises(WellConvergenceError, match="did not converge"):
        solve_connection_pressures_and_rates(
            tubing=WellTubing(None, None, 0.0),
            reference_depth=case.reference_depth,
            workspace=workspace,
            reference_pressure=2000.0,
            relevant_phases=types.SimpleNamespace(oil=1.0, water=1.0, gas=1.0),
            is_injector=False,
            control_spec=WellControlSpec(
                connection_pressure_mode=ConnectionPressureMode.UNIFORM_BHP
            ),
        )


def test_an_unknown_wellbore_model_is_rejected_by_name():
    from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
    from bores.wells.resolution.solvers import solve_connection_pressures_and_rates
    from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
    from bores.wells.resolution.tubing import WellTubing

    case, workspace = build_case_workspace(unit_system=UnitSystem.FIELD)
    unknown = homogeneous_wellbore(tubing_inner_diameter=case.diameter)._replace(name="invented")
    with pytest.raises(ValidationError, match="Unknown wellbore model 'invented'"):
        solve_connection_pressures_and_rates(
            tubing=WellTubing(unknown, None, 0.0),
            reference_depth=case.reference_depth,
            workspace=workspace,
            reference_pressure=2000.0,
            relevant_phases=types.SimpleNamespace(oil=1.0, water=1.0, gas=1.0),
            is_injector=False,
            control_spec=WellControlSpec(
                connection_pressure_mode=ConnectionPressureMode.HYDRAULIC
            ),
        )


def reentry_records():
    base = {"well": "P1", "i": 2, "j": 2, "k1": 1, "k2": 1, "status": "OPEN", "diameter": 0.5}
    return [
        {**base, "skin": 0.0, "schedule_time": 0.0},
        {**base, "skin": 5.0, "schedule_time": 10.0},
    ]


def load_reentered_well(*, current_time, compdat=None, wpimult=None):
    from bores.datastructures import GridDimensions
    from bores.wells.deck import load_well_from_records

    dims = GridDimensions(nx=3, ny=3, nz=3)
    depths = np.zeros((27, 3))
    depths[:, 2] = np.arange(27, dtype=float)
    grid = types.SimpleNamespace(
        dimensions=dims,
        cell_min_xyz=depths,
        cell_max_xyz=depths + 1.0,
        get_cell_center_at=lambda i, j, k: np.zeros(3),
    )
    return load_well_from_records(
        grid,
        {"well": "P1", "inflow_equation": "STD", "i": 2, "j": 2},
        compdat or reentry_records(),
        wpimult_records=wpimult,
        current_time=current_time,
    )


def test_a_reentered_connection_is_one_connection_at_a_time():
    from bores.wells.base import WellStatus

    early = load_reentered_well(current_time=0.0)
    assert [(p.schedule_status, p.skin) for p in early.perforations] == [
        (WellStatus.ACTIVE, 0.0),
        (WellStatus.PENDING, 5.0),
    ]
    for time in (10.0, 15.0):
        later = load_reentered_well(current_time=time)
        assert [(p.schedule_status, p.skin) for p in later.perforations] == [
            (WellStatus.ACTIVE, 5.0)
        ]


def test_wpimult_applies_to_the_connection_version_in_effect_and_resets_on_reentry():
    wpimult = [
        {"well": "P1", "multiplier": 0.5, "schedule_time": 5.0},
        {"well": "P1", "multiplier": 0.8, "schedule_time": 12.0},
    ]
    before = load_reentered_well(current_time=7.0, wpimult=wpimult)
    assert [p.connection_factor_multiplier for p in before.perforations] == [0.5, None]
    during = load_reentered_well(current_time=11.0, wpimult=wpimult)
    assert [p.connection_factor_multiplier for p in during.perforations] == [None]
    after = load_reentered_well(current_time=20.0, wpimult=wpimult)
    assert [p.connection_factor_multiplier for p in after.perforations] == [pytest.approx(0.8)]


def test_wpimult_before_a_connection_starts_is_an_error():
    from bores.errors import ValidationError

    compdat = [{**reentry_records()[0], "schedule_time": 20.0}]
    wpimult = [{"well": "P1", "multiplier": 0.5, "schedule_time": 5.0}]
    with pytest.raises(ValidationError, match="in effect"):
        load_reentered_well(current_time=6.0, compdat=compdat, wpimult=wpimult)


def test_overlapping_but_different_layer_ranges_cannot_be_reentered():
    from bores.errors import NotSupportedError

    base = reentry_records()[0]
    compdat = [
        {**base, "k1": 1, "k2": 3, "schedule_time": 0.0},
        {**base, "k1": 2, "k2": 2, "schedule_time": 10.0},
    ]
    with pytest.raises(NotSupportedError, match="overlap"):
        load_reentered_well(current_time=0.0, compdat=compdat)


def reentered_compiled_system():
    import attrs

    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.types import Orientation
    from bores.wells.base import WellStatus
    from bores.wells.compile import compile_well_system
    from bores.wells.factories import make_producer, make_well_controls, make_wells

    grid = make_cartesian_grid(nx=1, ny=1, nz=3, dx=50.0, dy=50.0, dz=50.0)
    well, control = make_producer(
        "P1", surface_location=(25.0, 25.0), perforation_depths=(0.0, 150.0), target_bhp=1000.0
    )
    first = well.perforations[0]
    versions = (
        first,
        attrs.evolve(first, skin=5.0, schedule_status=WellStatus.PENDING),
        attrs.evolve(first, skin=-2.0, schedule_status=WellStatus.PENDING),
    )
    compiled = compile_well_system(
        make_wells(attrs.evolve(well, perforations=versions)),
        make_well_controls(P1=control),
        grid,
        permeabilities={orientation: np.full(3, 100.0) for orientation in Orientation},
    )
    return types.SimpleNamespace(wells=compiled, reservoir=types.SimpleNamespace(grid=grid))


def test_activating_a_reentered_connection_swaps_versions_in_order():
    from bores.schedule.base import ScheduleContext
    from bores.wells.schedule import ActivateCompletions

    model = reentered_compiled_system()
    statuses = model.wells.perforations.schedule_statuses
    assert statuses.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    swap = ActivateCompletions(well_names=("P1",), i=1, j=1, k1=1, k2=3)
    swap(model, ScheduleContext(time=10.0))
    assert statuses.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    swap(model, ScheduleContext(time=20.0))
    assert statuses.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1]
    swap(model, ScheduleContext(time=30.0))
    assert statuses.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_connection_multipliers_and_targeted_status_changes_skip_connections_not_in_effect():
    from bores.schedule.base import ScheduleContext
    from bores.wells.schedule import (
        ActivateCompletions,
        MultiplyConnectionFactors,
        ShutInWells,
    )

    model = reentered_compiled_system()
    perforations = model.wells.perforations
    base_index = perforations.well_indices.copy()
    MultiplyConnectionFactors(well_names=("P1",), multiplier=0.5)(model, ScheduleContext(time=5.0))
    assert perforations.well_indices[:3].tolist() == pytest.approx((base_index[:3] * 0.5).tolist())
    assert perforations.well_indices[3:].tolist() == pytest.approx(base_index[3:].tolist())

    ShutInWells(well_names=("P1",), i=1, j=1, k1=1, k2=3)(model, ScheduleContext(time=6.0))
    assert perforations.completion_statuses.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]

    ShutInWells(well_names=("P1",))(model, ScheduleContext(time=7.0))
    assert model.wells.open_statuses.tolist() == [0]
    assert perforations.completion_statuses.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]

    ActivateCompletions(well_names=("P1",), i=1, j=1, k1=1, k2=3)(
        model, ScheduleContext(time=10.0)
    )
    assert perforations.well_indices[3:6].tolist() == pytest.approx(base_index[3:6].tolist())


def vfp_system(tables, *, producer_table_number=3, injector_table_number=None):
    controls = {
        "P1": types.SimpleNamespace(vfp_table=producer_table_number),
        "I1": types.SimpleNamespace(vfp_table=injector_table_number),
    }
    return types.SimpleNamespace(
        wells={
            "P1": types.SimpleNamespace(is_injector=False),
            "I1": types.SimpleNamespace(is_injector=True),
        },
        well_controls=types.SimpleNamespace(get=controls.get),
        get_tubing_performance=lambda name: tables,
    )


def test_vfp_preflight_accepts_tables_that_exist(tmp_path):
    from bores.schedule import RunSequence, at, schedule
    from bores.wells.schedule import SetWellLift, check_vfp_references

    tables = VFPTables.from_deck(producer_table(tmp_path))
    plan = schedule(
        at(time=10.0, action=RunSequence(actions=(SetWellLift(well_name="P1", vfp_table=3),)))
    )
    check_vfp_references(well_system=vfp_system(tables), schedule=plan)
    check_vfp_references(well_system=vfp_system(tables))


def test_vfp_preflight_reports_every_missing_table_and_where_it_came_from(tmp_path):
    from bores.errors import ValidationError
    from bores.schedule import RunSequence, at, schedule
    from bores.wells.schedule import SetWellLift, check_vfp_references

    tables = VFPTables.from_deck(producer_table(tmp_path))
    plan = schedule(
        at(
            time=10.0,
            action=RunSequence(actions=(SetWellLift(well_name="P1", vfp_table=9),)),
            name="swap",
        )
    )
    system = vfp_system(tables, injector_table_number=3)
    with pytest.raises(ValidationError) as caught:
        check_vfp_references(well_system=system, schedule=plan)
    message = str(caught.value)
    assert "well 'P1' (producer) uses VFP table 9 from schedule item 'swap'" in message
    assert "well 'I1' (injector) uses VFP table 3 from its initial control" in message


def test_vfp_preflight_skips_wells_that_do_not_use_vfp_tables():
    from bores.wells.schedule import check_vfp_references

    check_vfp_references(well_system=vfp_system(None, producer_table_number=7))


def test_schedule_names_may_repeat_but_lookup_by_name_must_be_unambiguous():
    from bores.errors import ValidationError
    from bores.schedule import Schedule, ScheduleItem, TimeEvent

    first = ScheduleItem(event=TimeEvent(at=5.0), action=FiringProbe(), name="twice")
    second = ScheduleItem(event=TimeEvent(at=5.0), action=FiringProbe(), name="twice")
    other = ScheduleItem(event=TimeEvent(at=6.0), action=FiringProbe(), name="once")
    plan = Schedule(items=(first, second, other))

    assert len(plan) == 3
    assert plan.get_all("twice") == (first, second)
    assert plan.get("once") is other
    assert plan.get("missing") is None
    with pytest.raises(ValidationError, match="get_all"):
        plan.get("twice")
    with pytest.raises(ValidationError):
        plan["twice"]
    with pytest.raises(KeyError):
        plan["missing"]
    assert "twice" in plan

    # `+` keeps both, `-` removes every item with the name, `|` replaces every item with it.
    assert len(plan + first) == 4
    assert (plan - second).items == (other,)
    replacement = ScheduleItem(event=TimeEvent(at=9.0), action=FiringProbe(), name="twice")
    assert (plan | replacement).items == (other, replacement)


def mode_setup(solver_name, case, unit_system):
    """The tubing and control spec that select each of the seven fixed-point solvers."""
    import importlib

    from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
    from bores.wells.resolution.tubing import WellTubing

    if solver_name == "uniform":
        mode, wellbore = ConnectionPressureMode.UNIFORM_BHP, None
    elif solver_name == "hydrostatic":
        mode, wellbore = ConnectionPressureMode.HYDROSTATIC, None
    else:
        mode = ConnectionPressureMode.HYDRAULIC
        wellbore = getattr(
            importlib.import_module(f"bores.wells.hydraulics.{solver_name}"),
            dict(KERNELS)[solver_name],
        )(tubing_inner_diameter=case.diameter, unit_system=unit_system)
    return (
        WellTubing(wellbore, None, 0.0),
        WellControlSpec(connection_pressure_mode=mode, unit_system=unit_system),
    )


def reference_rate_bisection(
    *, tubing, case, workspace, mask, is_injector, spec, target, low, high, condition
):
    """The rate bisection written as a plain Python loop over the compiled solve."""
    from bores.wells.resolution.solvers import solve_connection_pressures_and_rates

    average = 0.5 * (low + high)
    for _ in range(spec.max_bisection_iterations):
        average = 0.5 * (low + high)
        pressures, reservoir, surface = solve_connection_pressures_and_rates(
            tubing=tubing,
            reference_depth=case.reference_depth,
            workspace=workspace,
            reference_pressure=average,
            relevant_phases=mask,
            is_injector=is_injector,
            control_spec=spec,
        )
        rates = reservoir if condition == "reservoir" else surface
        value = rates.oil + rates.water + rates.gas
        if abs(value - target) <= spec.rate_convergence_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return average, pressures, reservoir


@pytest.mark.parametrize("solver_name", FIXED_POINT_DRIVERS)
@pytest.mark.parametrize("is_injector", [False, True])
@pytest.mark.parametrize("condition", ["surface", "reservoir"])
@pytest.mark.parametrize("fraction", [0.4, 50.0])
def test_the_rate_bisection_matches_a_python_loop_over_the_solve(
    solver_name, is_injector, condition, fraction
):
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.solvers import (
        get_pressure_bracket,
        solve_bhp,
        solve_connection_pressures_and_rates,
    )

    unit_system = UnitSystem.FIELD
    case, workspace = build_case_workspace(unit_system=unit_system)
    tubing, spec = mode_setup(solver_name, case, unit_system)
    mask = PhaseValues(0.0, 1.0, 0.0) if is_injector else PhaseValues(1.0, 1.0, 1.0)
    low, high = get_pressure_bracket(workspace, is_injector=is_injector, control_spec=spec)
    _, _, probe = solve_connection_pressures_and_rates(
        tubing=tubing,
        reference_depth=case.reference_depth,
        workspace=copy_workspace(workspace),
        reference_pressure=0.5 * (low + high),
        relevant_phases=mask,
        is_injector=is_injector,
        control_spec=spec,
    )
    # A fraction of the rate at the middle of the bracket is reachable; fifty times it is not, so
    # the bisection runs out of steps at one end of the bracket.
    target = fraction * (probe.oil + probe.water + probe.gas)

    expected_bhp, expected_pressures, expected_rates = reference_rate_bisection(
        tubing=tubing,
        case=case,
        workspace=copy_workspace(workspace),
        mask=mask,
        is_injector=is_injector,
        spec=spec,
        target=target,
        low=low,
        high=high,
        condition=condition,
    )
    bhp, pressures, rates = solve_bhp(
        tubing=tubing,
        reference_depth=case.reference_depth,
        workspace=copy_workspace(workspace),
        relevant_phases=mask,
        is_injector=is_injector,
        target=target,
        min_pressure=low,
        max_pressure=high,
        control_spec=spec,
        target_rate_condition=condition,
    )
    assert bhp == pytest.approx(expected_bhp, rel=1e-12)
    assert pressures == pytest.approx(expected_pressures, rel=1e-12)
    assert tuple(rates) == pytest.approx(tuple(expected_rates), rel=1e-12)


@pytest.mark.parametrize("solver_name", FIXED_POINT_DRIVERS)
def test_each_compiled_bisection_matches_the_same_code_run_as_python(solver_name):
    from bores.wells.resolution import fixed_point

    bisector = {
        "uniform": fixed_point.bisect_uniform,
        "hydrostatic": fixed_point.bisect_hydrostatic,
        "homogeneous": fixed_point.bisect_homogeneous,
        "beggs_and_brill": fixed_point.bisect_beggs_and_brill,
        "hagedorn_brown": fixed_point.bisect_hagedorn_brown,
        "gray": fixed_point.bisect_gray,
        "woldesemayat_ghajar": fixed_point.bisect_woldesemayat_ghajar,
    }[solver_name]
    unit_system = UnitSystem.FIELD
    case, workspace = build_case_workspace(unit_system=unit_system)
    # A solver takes its reference pressure after the model and depth. A bisection chooses its own.
    reference_index = {"uniform": 0, "hydrostatic": 1}.get(solver_name, 2)
    bisection = (1500.0, 800.0, 3100.0, False, 60, 1e-9)

    def arguments(copy):
        solver = list(solver_arguments(solver_name, case, copy, unit_system))
        solver.pop(reference_index)
        return (*solver, *bisection)

    compiled = bisector(*arguments(copy_workspace(workspace)))
    interpreted = bisector.py_func(*arguments(copy_workspace(workspace)))
    assert compiled.result.converged and interpreted.result.converged
    assert compiled.bhp == pytest.approx(interpreted.bhp, rel=1e-12)
    assert tuple(compiled.result) == pytest.approx(tuple(interpreted.result), rel=1e-9)


def build_group_system(*, wells_per_group=4, groups=2, connections=4, weak_well=None):
    """
    Resolves producers in group control, in several groups, each with an oil rate target, on a real
    compiled system. A `weak_well` is given too little productivity to reach its share.
    """
    import attrs

    from bores.grids.factories.cartesian import make_cartesian_grid
    from bores.types import Orientation
    from bores.wells.compile import compile_well_system
    from bores.wells.controls import ProducerControlMode
    from bores.wells.factories import (
        make_group_control,
        make_producer,
        make_well_controls,
        make_wells,
    )
    from bores.wells.groups import GroupControls, WellGroup, WellGroups
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
    from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
    from bores.wells.resolution.system import resolve_wells
    from bores.wells.state import ConnectionSample
    from bores.wells.workspace import build_wells_workspace

    count = wells_per_group * groups
    grid = make_cartesian_grid(nx=count, ny=1, nz=connections, dx=50.0, dy=50.0, dz=10.0)
    wells, controls = [], {}
    for i in range(count):
        well, control = make_producer(
            f"W{i}",
            surface_location=(25.0 + 50.0 * i, 25.0),
            perforation_depths=(0.0, 10.0 * connections),
            target_bhp=1000.0,
            reference_depth=5.0,
            group=f"G{i % groups}",
        )
        wells.append(well)
        controls[f"W{i}"] = attrs.evolve(
            control, mode=ProducerControlMode.GROUP, guide_rate=1.0 + (i % 3) * 0.5
        )
    group_target = wells_per_group * 400.0
    compiled = compile_well_system(
        make_wells(*wells),
        make_well_controls(**controls),
        grid,
        permeabilities={
            orientation: np.full(count * connections, 100.0) for orientation in Orientation
        },
        group_controls=GroupControls({
            f"G{g}": make_group_control(target_rate=group_target) for g in range(groups)
        }),
        groups=WellGroups({
            f"G{g}": WellGroup(name=f"G{g}", parent="FIELD") for g in range(groups)
        }),
    )
    if weak_well is not None:
        offsets = compiled.perforations.well_offsets
        compiled.perforations.well_indices[offsets[weak_well] : offsets[weak_well + 1]] *= 1e-4

    samples = [
        ConnectionSample(
            cell_index=k,
            pressure=3000.0 + 2.0 * k,
            temperature=200.0,
            phase_saturations=PhaseValues(0.6, 0.3, 0.1),
            phase_mobilities=PhaseValues(0.2, 0.1, 0.5),
            phase_densities=PhaseValues(50.0, 62.4, 5.0),
            phase_viscosities=PhaseValues(1.0, 0.5, 0.02),
            gas_liquid_surface_tension=20.0,
            phase_fvfs=PhaseValues(1.2, 1.0, 0.01),
        )
        for k in range(connections)
    ]
    fluids = compile_connection_fluids(connection_samples=samples)
    tubing = homogeneous_wellbore(tubing_inner_diameter=0.25)
    workspace = build_wells_workspace(
        n_wells=count, n_connections=int(compiled.perforations.well_indices.shape[0])
    )
    resolve_wells(
        compiled_system=compiled,
        workspace=workspace,
        control_spec=WellControlSpec(connection_pressure_mode=ConnectionPressureMode.HYDRAULIC),
        get_tubing_performance=lambda row: tubing,
        get_connection_fluids=lambda row: fluids,
    )
    return workspace, group_target, groups, count


def test_group_oil_rate_targets_are_met_across_the_wells_of_each_group():
    workspace, group_target, groups, count = build_group_system()
    for group in range(groups):
        total = float(workspace.surface_oil_rates[group:count:groups].sum())
        assert total == pytest.approx(group_target, rel=1e-3)
    assert np.all(workspace.surface_oil_rates > 0.0)


def test_a_well_that_cannot_reach_its_share_has_the_shortfall_taken_up_by_its_group():
    even, group_target, groups, count = build_group_system()
    workspace, _, _, _ = build_group_system(weak_well=0)
    # The weak well falls short of what it produced with the others, and its group still makes
    # its target, so the other wells in the group are producing more.
    assert workspace.surface_oil_rates[0] < 0.5 * even.surface_oil_rates[0]
    total = float(workspace.surface_oil_rates[0:count:groups].sum())
    assert total == pytest.approx(group_target, rel=1e-3)
    others = workspace.surface_oil_rates[groups:count:groups]
    assert np.all(others > even.surface_oil_rates[groups:count:groups])
