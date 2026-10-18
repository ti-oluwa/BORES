import types

import numpy as np
import pytest

from bores.constants import c, get_conversion_factors
from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.types import UnitSystem
from bores.wells import summary as well_summary
from bores.wells.base import WellType
from bores.wells.deck import (
    get_fluid_in_place_regions,
    load_producer_control_from_record,
    load_summary_items,
)
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
    reservoir = types.SimpleNamespace(pore_volumes=pore_volumes, regions=regions)
    return types.SimpleNamespace(reservoir=reservoir, unit_system=unit_system)


def region_workspace():
    state = types.SimpleNamespace(
        oil_saturation=np.array([0.5, 0.6, 0.4, 0.3]),
        water_saturation=np.array([0.3, 0.3, 0.3, 0.3]),
        gas_saturation=np.array([0.2, 0.1, 0.3, 0.4]),
        solution_gor=np.array([500.0, 400.0, 300.0, 200.0]),
    )
    pvt = types.SimpleNamespace(
        oil_formation_volume_factor=np.array([1.2, 1.25, 1.3, 1.35]),
        water_formation_volume_factor=np.array([1.0, 1.0, 1.0, 1.0]),
        gas_formation_volume_factor=np.array([0.005, 0.005, 0.006, 0.006]),
    )
    return types.SimpleNamespace(reservoir=state, physics=types.SimpleNamespace(pvt=pvt))


@pytest.fixture
def with_workspace(monkeypatch):
    monkeypatch.setattr(well_summary, "get_workspace", lambda *, context: region_workspace())


def test_region_in_place_matches_hand_calculation(with_workspace):
    model = region_model([1, 1, 2, 2])
    to_stb = c.CUBIC_FEET_TO_STB
    oil_ft3 = np.array([0.5 * 1000 / 1.2, 0.6 * 2000 / 1.25, 0.4 * 3000 / 1.3, 0.3 * 4000 / 1.35])
    roip = well_summary.RegionOilInPlace(region=1)(model, None)
    assert roip == pytest.approx(oil_ft3[:2].sum() * to_stb)
    rwip = well_summary.RegionWaterInPlace(region=2)(model, None)
    assert rwip == pytest.approx((0.3 * 3000 + 0.3 * 4000) * to_stb)
    free = np.array([0.2 * 1000 / 0.005, 0.1 * 2000 / 0.005])
    dissolved = np.array([500.0, 400.0]) * oil_ft3[:2] * to_stb
    assert well_summary.RegionGasInPlace(region=1)(model, None) == pytest.approx(
        (free + dissolved).sum()
    )


def test_regions_partition_the_field_and_metric_needs_no_barrel_conversion(with_workspace):
    model = region_model([1, 2, 1, 2])
    total = sum(well_summary.RegionOilInPlace(region=r)(model, None) for r in (1, 2))
    assert total == pytest.approx(
        well_summary.RegionOilInPlace(region=1)(region_model(None), None)
    )
    metric = well_summary.RegionOilInPlace(region=1)(region_model(None, UnitSystem.METRIC), None)
    assert metric == pytest.approx(total / c.CUBIC_FEET_TO_STB)


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
        "surface_fluid_properties": None,
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
        surface_fluid_properties=None,
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
    from bores.wells.hydraulics.base import SurfaceFluidProperties
    from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
    from bores.wells.hydraulics.vfp import as_vfp_table

    return as_vfp_table(
        homogeneous_wellbore(tubing_inner_diameter=0.25),
        table_number=1,
        reference_depth=5000.0,
        thps=np.array([100.0, 300.0]),
        surface_fluid_properties=SurfaceFluidProperties(density=50.0, viscosity=1.0),
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
        "surface_fluid_properties": None,
        "is_injector": False,
    }
    thp = compute_tubing_head_pressure(**kwargs, connection_rates=rates_by_connection)
    assert thp == pytest.approx(30.0)
    with pytest.raises(ValidationError, match="datum depth"):
        compute_tubing_head_pressure(**kwargs)


def test_attached_flowing_fluid_records_densities_and_a_zero_flow_density():
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.resolution.tubing import WellTubing, attach_flowing_fluid

    def sample(oil_density):
        return types.SimpleNamespace(
            phase_densities=PhaseValues(oil=oil_density, water=62.4, gas=5.0)
        )

    workspace = types.SimpleNamespace(
        well_indices=np.array([1.0, 3.0]),
        oil_mobilities=np.array([1.0, 1.0]),
        water_mobilities=np.array([0.0, 0.0]),
        gas_mobilities=np.array([2.0, 2.0]),
    )
    table = lookup_table(
        flow_type=VFPFlowType.LIQUID,
        water_type=VFPWaterFractionType.WATER_CUT,
        gas_type=VFPGasFractionType.GAS_OIL_RATIO,
    )
    tubing = attach_flowing_fluid(
        tubing=WellTubing(None, table, 0.0),
        workspace=workspace,
        connection_samples=[sample(40.0), sample(60.0)],
        relevant_phases=PhaseValues(oil=1.0, water=1.0, gas=1.0),
    )
    fluid = tubing.flowing_fluid
    assert fluid.oil_densities.tolist() == [40.0, 60.0]
    # Deliverability is oil 4 : gas 8, so the idle column is 1/3 oil (mean 55) and 2/3 gas.
    assert fluid.zero_flow_density == pytest.approx(55.0 / 3 + 2 * 5.0 / 3)
    untouched = attach_flowing_fluid(
        tubing=WellTubing(None, None, 0.0),
        workspace=workspace,
        connection_samples=[],
        relevant_phases=PhaseValues(oil=1.0, water=1.0, gas=1.0),
    )
    assert untouched.flowing_fluid is None


def resolve_producer(*, limits=None, vfp_tables=None, vfp_number=None, mode=None):
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
        target_bhp=1000.0,
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
        connection_samples=samples,
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


def hydrostatic_samples(oil_densities=(40.0, 50.0, 60.0)):
    from bores.wells.hydraulics.base import PhaseValues
    from bores.wells.state import ConnectionSample

    return [
        ConnectionSample(
            cell_index=i,
            pressure=3000.0,
            temperature=200.0,
            phase_saturations=PhaseValues(0.5, 0.3, 0.2),
            phase_mobilities=PhaseValues(1.0, 1.0, 1.0),
            phase_densities=PhaseValues(oil_density, 60.0, 10.0),
            phase_viscosities=PhaseValues(1.0, 1.0, 1.0),
            gas_liquid_surface_tension=20.0,
            phase_fvfs=PhaseValues(1.0, 1.0, 1.0),
        )
        for i, oil_density in enumerate(oil_densities)
    ]


def hydrostatic_pressures(*, reference_depth, oil, depths=(100.0, 200.0, 400.0), **kwargs):
    from bores.wells.resolution.hydrostatic import compute_hydrostatic_connection_pressures

    count = len(depths)
    return compute_hydrostatic_connection_pressures(
        reference_depth=reference_depth,
        reference_pressure=1000.0,
        representative_depths=np.array(depths),
        connection_samples=hydrostatic_samples()[:count],
        connection_rates=connection_rates(
            oil=oil,
            water=kwargs.get("water", (0.0,) * count),
            gas=kwargs.get("gas", (0.0,) * count),
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
    from bores.wells.resolution.hydrostatic import compute_hydrostatic_connection_pressures

    with pytest.raises(ValidationError, match="same length"):
        compute_hydrostatic_connection_pressures(
            reference_depth=0.0,
            reference_pressure=1000.0,
            representative_depths=np.array([1.0, 2.0]),
            connection_samples=hydrostatic_samples()[:3],
            connection_rates=connection_rates(oil=(1.0, 1.0, 1.0), water=(0, 0, 0), gas=(0, 0, 0)),
            unit_system=UnitSystem.FIELD,
        )


def test_a_vfp_well_needs_a_connection_pressure_mode_that_does_not_need_hydraulics(tmp_path):
    from bores.wells.resolution.spec import ConnectionPressureMode

    tables = VFPTables.from_deck(producer_table(tmp_path))
    with pytest.raises(ValidationError, match="HYDROSTATIC"):
        resolve_producer(vfp_tables=tables, vfp_number=3, mode=ConnectionPressureMode.HYDRAULIC)


def test_a_vfp_well_resolves_with_hydrostatic_connection_pressures(tmp_path):
    from bores.wells.resolution.hydrostatic import compute_hydrostatic_connection_pressures
    from bores.wells.resolution.spec import ConnectionPressureMode
    from bores.wells.workspace import ConnectionRates

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
    expected = compute_hydrostatic_connection_pressures(
        reference_depth=75.0,
        reference_pressure=1000.0,
        representative_depths=np.array([25.0, 75.0, 125.0]),
        connection_samples=samples,
        connection_rates=ConnectionRates(
            oil=workspace.connection_oil_rates,
            water=workspace.connection_water_rates,
            gas=workspace.connection_gas_rates,
        ),
        unit_system=UnitSystem.FIELD,
    )
    assert workspace.connection_pressures == pytest.approx(expected, rel=1e-4)
    assert workspace.connection_pressures[0] < 1000.0 < workspace.connection_pressures[2]
    # The profile feeds back into the rates: a flat profile at the same BHP gives different ones.
    assert not np.allclose(workspace.connection_gas_rates, uniform.connection_gas_rates)
