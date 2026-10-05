import types

import numpy as np
import pytest

from bores.constants import c
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
    VFPTable,
    VFPTables,
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
    assert np.allclose(data.gas_oil_ratios, [100.0, 500.0])  # Mscf/STB -> scf/STB
    table = VFPTable.from_deck(deck, table_number=3)
    bhp = table.query(
        flow_rate=500.0,
        thp=40.0,
        water_cut=1.0,
        gas_oil_ratio=500.0,
        artificial_lift_quantity=1000.0,
    )
    assert float(bhp) == pytest.approx(100 + 10 * 2 + 2 + 2 + 2)


def test_water_oil_ratio_axis_is_converted_to_water_cut(tmp_path):
    data = load_vfp_data(producer_table(tmp_path, wfr="WOR"), table_number=3)
    assert np.allclose(data.water_cuts, [0.0, 0.5])  # WOR 0 and 1


def test_metric_tables_keep_their_gas_oil_ratio(tmp_path):
    deck = producer_table(tmp_path, units="METRIC", unit="METRIC")
    assert np.allclose(load_vfp_data(deck, table_number=3).gas_oil_ratios, [0.1, 0.5])


@pytest.mark.parametrize(
    "kwargs, message",
    [({"flo": "GAS"}, "liquid rate"), ({"gfr": "GLR"}, "GOR"), ({"wfr": "WGR"}, "WCT")],
)
def test_vfp_tables_that_cannot_map_exactly_are_rejected_clearly(tmp_path, kwargs, message):
    with pytest.raises(ValidationError, match=message):
        load_vfp_data(producer_table(tmp_path, **kwargs), table_number=3)


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


def test_oil_rate_table_is_re_expressed_against_liquid_rate(tmp_path):
    deck = producer_table(tmp_path, wfr="WOR", flo="OIL")
    with pytest.warns(UserWarning, match="re-expressed"):
        data = VFPData.from_deck(deck, table_number=3)
    table = VFPTable(data)
    # Oil rates 100 and 500 at WOR 0 and 1 (water cut 0 and 0.5) are liquid rates
    # 100/500 and 200/1000, so the union of those is the new flow axis and each water cut's
    # original points are reproduced exactly.
    assert np.allclose(data.flow_rates, [100.0, 200.0, 500.0, 1000.0])
    args = {"thp": 20.0, "gas_oil_ratio": 100.0, "artificial_lift_quantity": 0.0}
    assert float(table.query(flow_rate=100.0, water_cut=0.0, **args)) == pytest.approx(13.0)
    assert float(table.query(flow_rate=500.0, water_cut=0.0, **args)) == pytest.approx(113.0)
    assert float(table.query(flow_rate=200.0, water_cut=0.5, **args)) == pytest.approx(14.0)
    assert float(table.query(flow_rate=1000.0, water_cut=0.5, **args)) == pytest.approx(114.0)


def test_oil_rate_table_with_a_water_cut_of_one_is_rejected(tmp_path):
    deck = producer_table(tmp_path, wfr="WCT", flo="OIL")  # the template's water axis is 0 and 1
    with pytest.raises(ValidationError, match="water cut of 1"):
        load_vfp_data(deck, table_number=3)


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


def test_wlift_triggers_are_reported_as_unsupported_not_ignored(tmp_path):
    from bores.errors import NotSupportedError
    from bores.wells.deck import load_schedule

    deck = deck_from(tmp_path, "WLIFT\n 'P1' 100 OIL 2 400 /\n/\nTSTEP\n 30 /\n")
    with pytest.raises(NotSupportedError, match="trigger_limit"):
        load_schedule(deck, compiled_at=-1.0)
