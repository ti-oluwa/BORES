from bores.types import Orientation
from bores.wells.base import CompletionStatus
from bores.wells.controls import EconomicQuantity, InjectorControlMode, ProducerControlMode
from bores.wells.groups import GroupInjectorControlMode, GroupProducerControlMode

DIRECTION_MAP = {"X": Orientation.X, "Y": Orientation.Y, "Z": Orientation.Z}
PRODUCER_CONTROL_MODE_MAP = {
    "ORAT": ProducerControlMode.OIL_RATE,
    "WRAT": ProducerControlMode.WATER_RATE,
    "GRAT": ProducerControlMode.GAS_RATE,
    "LRAT": ProducerControlMode.LIQUID_RATE,
    "RESV": ProducerControlMode.RESERVOIR_VOLUME_RATE,
    "BHP": ProducerControlMode.BHP,
    "THP": ProducerControlMode.THP,
    "GRUP": ProducerControlMode.GROUP,
}
"""
Maps `WCONPROD`/`WELTARG` item 2's deck-literal control mode string to the
internal `ProducerControlMode`. The deck keeps Eclipse's own abbreviated
vocabulary (`ORAT`, `WRAT`, and so on); the internal enum spells things
out. This is the one place that translation happens.
"""
INJECTOR_CONTROL_MODE_MAP = {
    "RATE": InjectorControlMode.RATE,
    "RESV": InjectorControlMode.RESERVOIR_VOLUME_RATE,
    "BHP": InjectorControlMode.BHP,
    "THP": InjectorControlMode.THP,
    "GRUP": InjectorControlMode.GROUP,
}
"""Injector analogue of `PRODUCER_CONTROL_MODE_MAP`, for `WCONINJE`/`WELTARG`."""
GROUP_PRODUCER_CONTROL_MODE_MAP = {
    "ORAT": GroupProducerControlMode.OIL_RATE,
    "WRAT": GroupProducerControlMode.WATER_RATE,
    "GRAT": GroupProducerControlMode.GAS_RATE,
    "LRAT": GroupProducerControlMode.LIQUID_RATE,
    "RESV": GroupProducerControlMode.RESERVOIR_VOLUME_RATE,
    "FLD": GroupProducerControlMode.FIELD,
    "NONE": GroupProducerControlMode.NONE,
}
"""Group-control analogue of `PRODUCER_CONTROL_MODE_MAP`, for `GCONPROD`."""
GROUP_INJECTOR_CONTROL_MODE_MAP = {
    "RATE": GroupInjectorControlMode.RATE,
    "RESV": GroupInjectorControlMode.RESERVOIR_VOLUME_RATE,
    "VREP": GroupInjectorControlMode.VOIDAGE_REPLACEMENT,
    "REIN": GroupInjectorControlMode.REINJECTION,
    "FLD": GroupInjectorControlMode.FIELD,
}
"""Group-control analogue of `INJECTOR_CONTROL_MODE_MAP`, for `GCONINJE`."""
WELOPEN_STATUS_MAP = {
    "OPEN": CompletionStatus.OPEN,
    "AUTO": CompletionStatus.OPEN,
    "SHUT": CompletionStatus.SHUT,
    "STOP": CompletionStatus.SHUT,
}
"""
`CompletionStatus` only distinguishes open from shut, so `WELOPEN`'s
`STOP` (stop flow, but not the same as a deck author marking a completion
as never meant to flow) is treated the same as `SHUT` here, and `AUTO`
(resume normal operation) the same as `OPEN`. A real distinction between
"shut" and "temporarily stopped" would need `CompletionStatus` itself
extended, not something patched in at the deck-loading layer.
"""

WELTARG_TARGET_FIELD = {
    "ORAT": "target_rate",
    "WRAT": "target_rate",
    "GRAT": "target_rate",
    "LRAT": "target_rate",
    "RESV": "target_rate",
    "RATE": "target_rate",
    "BHP": "target_bhp",
    "THP": "target_thp",
    "GRUP": None,
}
"""
Which `ProducerControl`/`InjectorControl` field a `WELTARG` record's
`control_mode` writes to. `GRUP` takes no value, just switches the mode.
"""


ECONOMIC_QUANTITY_FIELDS = {
    EconomicQuantity.WATER_CUT: "max_water_cut",
    EconomicQuantity.GOR: "max_gor",
    EconomicQuantity.WATER_GAS_RATIO: "max_wgr",
}
ECONOMIC_MIN_RATE_QUANTITY_FIELDS = {EconomicQuantity.OIL_RATE: "min_oil_rate"}
