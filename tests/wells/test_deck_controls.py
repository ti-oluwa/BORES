import pytest

from bores.errors import ValidationError
from bores.types import UnitSystem
from bores.wells.deck import (
    is_status_only_producer_record,
    load_controls_from_records,
    load_group_controls_from_records,
    load_producer_control_from_record,
)


@pytest.mark.parametrize("unit_system", [UnitSystem.METRIC, UnitSystem.FIELD])
def test_empty_well_controls_take_the_deck_unit_system(unit_system):
    controls = load_controls_from_records([], [], unit_system=unit_system)
    assert controls.unit_system is unit_system


@pytest.mark.parametrize("unit_system", [UnitSystem.METRIC, UnitSystem.FIELD])
def test_empty_group_controls_take_the_deck_unit_system(unit_system):
    controls = load_group_controls_from_records([], [], unit_system=unit_system)
    assert controls.unit_system is unit_system


def shut_record(status: str, control_mode=None):
    return {"well": "W1", "status": status, "control_mode": control_mode, "schedule_time": 0.0}


@pytest.mark.parametrize("status", ["SHUT", "STOP", "shut"])
def test_shut_or_stopped_record_without_a_control_mode_only_changes_status(status):
    assert is_status_only_producer_record(shut_record(status))


@pytest.mark.parametrize(
    "record",
    [shut_record("OPEN"), shut_record("SHUT", control_mode="ORAT"), {"well": "W1"}],
)
def test_other_records_are_not_status_only(record):
    assert not is_status_only_producer_record(record)


def test_status_only_record_leaves_no_control_in_the_replayed_controls():
    controls = load_controls_from_records([shut_record("SHUT")], [], unit_system=UnitSystem.METRIC)
    assert "W1" not in controls
    assert controls.unit_system is UnitSystem.METRIC


def test_open_producer_without_a_control_mode_raises_a_clear_error():
    with pytest.raises(ValidationError, match="no control mode"):
        load_producer_control_from_record(shut_record("OPEN"), unit_system=UnitSystem.METRIC)
