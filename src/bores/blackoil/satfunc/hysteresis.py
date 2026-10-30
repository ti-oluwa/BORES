import typing

import attrs
from typing_extensions import Self

from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.types import FluidPhase, RelPermHysteresisModel

__all__ = [
    "HYSTERESIS_WETTING_PHASE_MAP",
    "LIMITING_CURVES_MAP",
    "RELPERM_HYSTERESIS_MODEL_MAP",
    "HysteresisOptions",
]

RELPERM_HYSTERESIS_MODEL_MAP: typing.Mapping[int, tuple[RelPermHysteresisModel, bool]] = {
    -1: (RelPermHysteresisModel.NONE, False),
    0: (RelPermHysteresisModel.CARLSON, False),
    1: (RelPermHysteresisModel.CARLSON, True),
    2: (RelPermHysteresisModel.KILLOUGH, False),
    3: (RelPermHysteresisModel.KILLOUGH, True),
}
"""
Maps `EHYSTR` item 2 to the relative permeability hysteresis model and to whether the wetting
phase takes its imbibition curve rather than its drainage curve. This is the one place the
numeric codes are translated.
"""

LIMITING_CURVES_MAP: typing.Mapping[str, tuple[bool, bool]] = {
    "BOTH": (True, True),
    "KR": (True, False),
    "PC": (False, True),
}
"""
Maps `EHYSTR` item 5 to `(apply_to_relperm, apply_to_capillary_pressure)`. This is the one place
the deck's abbreviations are translated.
"""

HYSTERESIS_WETTING_PHASE_MAP: typing.Mapping[str, FluidPhase] = {
    "OIL": FluidPhase.OIL,
    "GAS": FluidPhase.GAS,
    "WATER": FluidPhase.WATER,
}
"""Maps `EHYSTR` item 8 to the wetting phase."""


@attrs.frozen(kw_only=True, slots=True)
class HysteresisOptions:
    """
    How saturation function hysteresis is modelled, as set by `SATOPTS` and `EHYSTR`.

    Every field has the Eclipse default, so `HysteresisOptions()` is what a deck gets from
    `SATOPTS HYSTER` with no `EHYSTR`. The deck's abbreviations are translated once, when the
    options are loaded, so nothing here holds an Eclipse mnemonic.
    """

    relperm_model: RelPermHysteresisModel = RelPermHysteresisModel.CARLSON
    """
    Construction used for the relative permeability scanning curves. `BLENDED_SCANNING` keeps
    BORES's own weighted blend available next to Eclipse's two models.
    """

    wetting_phase_uses_imbibition: bool = False
    """Whether the wetting phase takes its imbibition curve instead of its drainage curve."""

    capillary_pressure_curvature: float = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_CAPILLARY_PRESSURE_CURVATURE)
    )
    """Curvature parameter of the capillary pressure scanning curves (`EHYSTR` item 1)."""

    killough_wetting_curvature: float = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_KILLOUGH_WETTING_CURVATURE)
    )
    """Curvature parameter of the Killough wetting phase scanning curves (`EHYSTR` item 3)."""

    killough_trapped_modifier: float = attrs.field(
        factory=lambda: typing.cast(float, c.HYSTERESIS_KILLOUGH_TRAPPED_MODIFIER)
    )
    """Modification parameter of the Killough trapped non-wetting saturation (`EHYSTR` item 4)."""

    apply_to_relperm: bool = True
    """Whether relative permeability follows hysteresis. Set from `EHYSTR` item 5."""

    apply_to_capillary_pressure: bool = True
    """Whether capillary pressure follows hysteresis. Set from `EHYSTR` item 5."""

    retrace_capillary_pressure: bool = True
    """Whether capillary pressure scanning curves are retraced rather than new (`EHYSTR` item 6)."""

    starts_on_imbibition_curve: bool = False
    """Whether the mobility starts on the imbibition curve (`EHYSTR` item 7)."""

    wetting_phase: FluidPhase = FluidPhase.OIL
    """Wetting phase (`EHYSTR` item 8). Water and gas are the non-wetting phases when it is oil."""

    baker_interpolation_oil: bool = False
    """Whether Baker's interpolation is used for oil relative permeability (`EHYSTR` item 9)."""

    baker_interpolation_gas: bool = False
    """Whether Baker's interpolation is used for gas relative permeability (`EHYSTR` item 10)."""

    baker_interpolation_water: bool = False
    """Whether Baker's interpolation is used for water relative permeability (`EHYSTR` item 11)."""

    threshold_saturation: float = 0.0
    """Saturation change below which hysteresis is not triggered (`EHYSTR` item 12)."""

    fix_killough_wetting_phase: bool = False
    """Whether the Killough wetting phase correction is applied (`EHYSTR` item 13)."""

    scale_capillary_pressure: bool = False
    """Whether capillary pressure is scaled (`EHYSTR` item 14)."""

    def __attrs_post_init__(self) -> None:
        if self.wetting_phase not in HYSTERESIS_WETTING_PHASE_MAP.values():
            raise ValidationError(
                f"`wetting_phase` must be oil, gas or water; got {self.wetting_phase}."
            )
        if self.capillary_pressure_curvature < 0.0:
            raise ValidationError("`capillary_pressure_curvature` must not be negative.")
        if self.killough_wetting_curvature <= 0.0:
            raise ValidationError("`killough_wetting_curvature` must be positive.")
        if self.killough_trapped_modifier < 0.0:
            raise ValidationError("`killough_trapped_modifier` must not be negative.")
        if not 0.0 <= self.threshold_saturation < 1.0:
            raise ValidationError("`threshold_saturation` must lie in [0, 1).")

    @property
    def relperm_hysteresis_active(self) -> bool:
        """Whether relative permeability follows hysteresis at all."""
        return self.apply_to_relperm and self.relperm_model != RelPermHysteresisModel.NONE

    @property
    def water_imbibition_when_increasing(self) -> bool:
        """
        Whether water imbibes as its saturation rises, which it does when water is the wetting
        phase. This is the argument the saturation history tracker takes for water.
        """
        return self.wetting_phase == FluidPhase.WATER

    @property
    def gas_imbibition_when_increasing(self) -> bool:
        """
        Whether gas imbibes as its saturation rises, which it does only when gas is the wetting
        phase. This is the argument the saturation history tracker takes for gas.
        """
        return self.wetting_phase == FluidPhase.GAS

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        relperm_model: RelPermHysteresisModel | None = None,
    ) -> Self | None:
        """
        Read the hysteresis options from a deck.

        Hysteresis is on when `SATOPTS` lists `HYSTER`. `EHYSTR` then sets the options, and
        any item it leaves out, or all of them if it is absent, keeps its Eclipse default.

        :param deck_file: The deck to read.
        :param relperm_model: Use this relative permeability model instead of the one `EHYSTR`
            item 2 selects. This is how `BLENDED_SCANNING`, which no `EHYSTR` code selects, is
            chosen for a deck. The wetting phase still follows item 2.
        :returns: The options, or `None` when the deck does not turn hysteresis on.
        """
        if "HYSTER" not in (deck_file.get("SATOPTS") or ()):
            return None
        record = deck_file.get("EHYSTR")
        if record is None:
            return cls() if relperm_model is None else cls(relperm_model=relperm_model)

        code = int(record["relative_permeability_model"])
        if code not in RELPERM_HYSTERESIS_MODEL_MAP:
            raise ValidationError(
                f"`EHYSTR` item 2 must be one of {sorted(RELPERM_HYSTERESIS_MODEL_MAP)}; got {code}."
            )
        deck_model, wetting_phase_uses_imbibition = RELPERM_HYSTERESIS_MODEL_MAP[code]
        apply_to_relperm, apply_to_capillary_pressure = LIMITING_CURVES_MAP[
            record["limiting_curves"]
        ]
        return cls(
            relperm_model=deck_model if relperm_model is None else relperm_model,
            wetting_phase_uses_imbibition=wetting_phase_uses_imbibition,
            capillary_pressure_curvature=float(record["capillary_pressure_curvature"]),
            killough_wetting_curvature=float(record["killough_wetting_curvature"]),
            killough_trapped_modifier=float(record["killough_trapped_modifier"]),
            apply_to_relperm=apply_to_relperm,
            apply_to_capillary_pressure=apply_to_capillary_pressure,
            retrace_capillary_pressure=record["capillary_pressure_shape"] == "RETR",
            starts_on_imbibition_curve=record["initial_mobility_curve"] == "IMBIB",
            wetting_phase=HYSTERESIS_WETTING_PHASE_MAP[record["wetting_phase"]],
            baker_interpolation_oil=record["baker_oil"] == "YES",
            baker_interpolation_gas=record["baker_gas"] == "YES",
            baker_interpolation_water=record["baker_water"] == "YES",
            threshold_saturation=float(record["threshold_saturation"]),
            fix_killough_wetting_phase=bool(record["fix_killough_wetting_phase"]),
            scale_capillary_pressure=bool(record["scale_capillary_pressure"]),
        )
