"""Dispatch wellbore pressure calculations to the configured correlation."""

import typing

from bores.errors import ValidationError
from bores.types import Number, NumberArray, OneDimension
from bores.wells.hydraulics.base import WellBoreModel
from bores.wells.hydraulics.beggs_and_brill import (
    compute_perforation_pressures as compute_beggs_brill_perforation_pressures,
)
from bores.wells.hydraulics.beggs_and_brill import (
    compute_tubing_head_pressure as compute_beggs_brill_tubing_head_pressure,
)
from bores.wells.hydraulics.gray import (
    compute_perforation_pressures as compute_gray_perforation_pressures,
)
from bores.wells.hydraulics.gray import (
    compute_tubing_head_pressure as compute_gray_tubing_head_pressure,
)
from bores.wells.hydraulics.hagedorn_brown import (
    compute_perforation_pressures as compute_hagedorn_brown_perforation_pressures,
)
from bores.wells.hydraulics.hagedorn_brown import (
    compute_tubing_head_pressure as compute_hagedorn_brown_tubing_head_pressure,
)
from bores.wells.hydraulics.homogeneous import (
    compute_perforation_pressures as compute_homogeneous_perforation_pressures,
)
from bores.wells.hydraulics.homogeneous import (
    compute_tubing_head_pressure as compute_homogeneous_tubing_head_pressure,
)
from bores.wells.hydraulics.woldesemayat_ghajar import (
    compute_perforation_pressures as compute_woldesemayat_ghajar_perforation_pressures,
)
from bores.wells.hydraulics.woldesemayat_ghajar import (
    compute_tubing_head_pressure as compute_woldesemayat_ghajar_tubing_head_pressure,
)

__all__ = ["compute_perforation_pressures", "compute_tubing_head_pressure"]


class PerforationPressuresFunction(typing.Protocol):
    """A correlation function returning one pressure per perforation."""

    def __call__(
        self,
        model: typing.Any,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> NumberArray[OneDimension]: ...


class TubingHeadPressureFunction(typing.Protocol):
    """A correlation function returning a single tubing head pressure."""

    def __call__(
        self,
        model: typing.Any,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> Number: ...


PERFORATION_PRESSURE_FUNCTIONS: dict[str, PerforationPressuresFunction] = {
    "homogeneous": compute_homogeneous_perforation_pressures,
    "beggs_brill": compute_beggs_brill_perforation_pressures,
    "beggs_and_brill": compute_beggs_brill_perforation_pressures,
    "hagedorn_brown": compute_hagedorn_brown_perforation_pressures,
    "gray": compute_gray_perforation_pressures,
    "woldesemayat_ghajar": compute_woldesemayat_ghajar_perforation_pressures,
}

TUBING_HEAD_PRESSURE_FUNCTIONS: dict[str, TubingHeadPressureFunction] = {
    "homogeneous": compute_homogeneous_tubing_head_pressure,
    "beggs_brill": compute_beggs_brill_tubing_head_pressure,
    "beggs_and_brill": compute_beggs_brill_tubing_head_pressure,
    "hagedorn_brown": compute_hagedorn_brown_tubing_head_pressure,
    "gray": compute_gray_tubing_head_pressure,
    "woldesemayat_ghajar": compute_woldesemayat_ghajar_tubing_head_pressure,
}


def compute_perforation_pressures(
    *, wellbore: WellBoreModel | None, **kwargs: typing.Any
) -> NumberArray[OneDimension]:
    """
    Compute perforation pressures using the correlation named by `wellbore.name`.

    :param wellbore: `WellBoreModel` naming a hydraulics correlation. Required; `None` raises.
    :param kwargs: Forwarded to that correlation's `compute_perforation_pressures`.
    :returns: Pressure at each connection.
    :raises ValidationError: If `wellbore` is `None`, or its name isn't recognized.
    """
    if wellbore is None:
        raise ValidationError(
            "Perforation pressures require a `WellBoreModel`; without one, "
            "choose a connection pressure mode in the well-control solver."
        )
    compute = PERFORATION_PRESSURE_FUNCTIONS.get(wellbore.name)
    if compute is None:
        raise ValidationError(f"Unknown `WellBoreModel` name: {wellbore.name!r}")
    return typing.cast(NumberArray[OneDimension], compute(model=wellbore.options, **kwargs))


def compute_tubing_head_pressure(
    *, wellbore: WellBoreModel | None, **kwargs: typing.Any
) -> Number:
    """
    Compute tubing head pressure using the correlation named by `wellbore.name`.

    :param wellbore: `WellBoreModel` naming a hydraulics correlation. Required; `None` raises.
    :param kwargs: Forwarded to that correlation's `compute_tubing_head_pressure`.
    :returns: Tubing head pressure.
    :raises ValidationError: If `wellbore` is `None`, or its name isn't recognized.
    """
    if wellbore is None:
        raise ValidationError(
            "Tubing head pressure requires a `WellBoreModel` or VFP table "
            "assigned to this well; THP control, a `THPLimit`, and THP "
            "reporting all need one. `ConnectionPressureMode` only covers "
            "connection-to-connection pressure distribution, "
            "not the datum-to-surface relationship."
        )
    compute = TUBING_HEAD_PRESSURE_FUNCTIONS.get(wellbore.name)
    if compute is None:
        raise ValidationError(f"Unknown `WellBoreModel` name: {wellbore.name!r}")
    return typing.cast(Number, compute(model=wellbore.options, **kwargs))
