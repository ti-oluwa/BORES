import types

import numpy as np
import pytest

from bores.constants import c
from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.types import UnitSystem
from bores.wells import summary as well_summary
from bores.wells.deck import (
    get_fluid_in_place_regions,
    load_producer_control_from_record,
    load_summary_items,
    load_vfp_data,
    load_vfp_table,
    load_vfp_tables,
)
from bores.wells.hydraulics.vfp import VFPData, VFPTable

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
    assert producer.bhps.shape == (2, 2, 2, 2, 2)
    assert producer.bhps[0, 1, 0, 1, 1] == 10 * 2 + 1 + 2 + 2  # row `2 1 2 2` -> first flow value
    injector = deck.get("VFPINJ")[0]
    assert injector.bhps.shape == (3, 2) and injector.bhps[2, 1] == pytest.approx(100.0)


def test_vfp_table_with_missing_rows_is_rejected(tmp_path):
    text = VFPPROD_BODY.format(wfr="WCT", gfr="GOR", units="FIELD", gor0=0.1, gor1=0.5)
    truncated = "\n".join(text.splitlines()[:-1]) + "\n"
    with pytest.raises(Exception, match="cover every combination"):
        deck_from(tmp_path, truncated).get("VFPPROD")


def test_field_vfp_table_rescales_gas_oil_ratio_and_queries_exactly(tmp_path):
    deck = producer_table(tmp_path)
    data = load_vfp_data(deck)[0]
    assert data.unit_system == UnitSystem.FIELD
    assert np.allclose(data.gas_oil_ratios, [100.0, 500.0])  # Mscf/STB -> scf/STB
    table = load_vfp_table(deck, 3)
    bhp = table.query(
        flow_rate=500.0,
        thp=40.0,
        water_cut=1.0,
        gas_oil_ratio=500.0,
        artificial_lift_quantity=1000.0,
    )
    assert float(bhp) == pytest.approx(100 + 10 * 2 + 2 + 2 + 2)


def test_water_oil_ratio_axis_is_converted_to_water_cut(tmp_path):
    data = load_vfp_data(producer_table(tmp_path, wfr="WOR"))[0]
    assert np.allclose(data.water_cuts, [0.0, 0.5])  # WOR 0 and 1


def test_metric_tables_keep_their_gas_oil_ratio(tmp_path):
    deck = producer_table(tmp_path, units="METRIC", unit="METRIC")
    assert np.allclose(load_vfp_data(deck)[0].gas_oil_ratios, [0.1, 0.5])


@pytest.mark.parametrize(
    "kwargs, message",
    [({"flo": "OIL"}, "liquid rate"), ({"gfr": "GLR"}, "GOR"), ({"wfr": "WGR"}, "WCT")],
)
def test_vfp_tables_that_cannot_map_exactly_are_rejected_clearly(tmp_path, kwargs, message):
    with pytest.raises(ValidationError, match=message):
        load_vfp_data(producer_table(tmp_path, **kwargs))


def test_injector_table_becomes_a_five_axis_table_and_gas_flow_is_rescaled(tmp_path):
    deck = deck_from(
        tmp_path,
        "VFPINJ\n 5 2000 GAS THP FIELD BHP /\n 100 200 /\n 20 40 /\n 1  50 60 /\n 2  80 90 /\n",
    )
    data = load_vfp_data(deck)[0]
    assert data.bhps.shape == (2, 2, 1, 1, 1)
    assert np.allclose(data.flow_rates, [100_000.0, 200_000.0])  # Mscf/d -> scf/d
    assert data.bhps[1, 0, 0, 0, 0] == pytest.approx(60.0)


def test_missing_table_and_shared_numbers_are_reported(tmp_path):
    deck = producer_table(tmp_path)
    with pytest.raises(ValidationError, match="no VFP table number 9"):
        load_vfp_table(deck, 9)
    both = deck_from(
        tmp_path,
        VFPPROD_BODY.format(wfr="WCT", gfr="GOR", units="FIELD", gor0=0.1, gor1=0.5)
        + "VFPINJ\n 3 2000 WAT THP FIELD BHP /\n 100 200 /\n 20 40 /\n 1  50 60 /\n 2  80 90 /\n",
    )
    with pytest.raises(ValidationError, match="both"):
        load_vfp_tables(both)
    assert set(load_vfp_tables(producer_table(tmp_path)).tables) == {3}


def test_vfp_classes_build_from_a_parsed_table(tmp_path):
    parsed = producer_table(tmp_path).get("VFPPROD")[0]
    assert isinstance(VFPData.from_deck(parsed), VFPData)
    assert isinstance(VFPTable.from_deck(parsed), VFPTable)


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
