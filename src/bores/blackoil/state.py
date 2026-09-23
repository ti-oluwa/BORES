"""Dynamic simulation state for one point in time: reservoir state plus optional well states."""

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.reservoir.state.base import ReservoirState
from bores.serde.base import Serializable
from bores.types import Number, UnitConversionTable, UnitSystem
from bores.utils import scale
from bores.wells.state import WellStates

__all__ = ["BlackOilModelState"]


@attrs.frozen(kw_only=True, slots=True)
class BlackOilModelState(Serializable):
    """Reservoir state plus well states at one simulation time."""

    reservoir: ReservoirState
    """Dynamic reservoir state, including pressure and phase saturations."""

    wells: WellStates | None = None
    """Optional dynamic state for the wells in the model."""

    time: Number = 0.0
    """Simulation time this state corresponds to (in `unit_system`)."""

    def __attrs_post_init__(self) -> None:
        if self.wells is not None and self.wells.unit_system != self.reservoir.unit_system:
            raise ValidationError(
                f"`wells.unit_system` ({self.wells.unit_system.value}) != "
                f"`reservoir.unit_system` ({self.reservoir.unit_system.value})."
            )

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system shared by reservoir and wells."""
        return self.reservoir.unit_system

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the reservoir, well state, and simulation time to *target*.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `BlackOilModelState` with reservoir and wells (if set)
            converted to target.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self,
            reservoir=self.reservoir.convert(target, table=table),
            wells=self.wells.convert(target, table=table) if self.wells is not None else None,
            time=scale(self.time, factors["time"]),
        )
