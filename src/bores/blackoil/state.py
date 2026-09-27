"""Dynamic simulation state for one point in time: reservoir state plus optional well states."""

import typing

import attrs
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.compile import CompiledBlackOilModel
from bores.blackoil.model import BlackOilModel
from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.reservoir.workspace import HysteresisWorkspace, load_reservoir_state
from bores.serde.base import Serializable
from bores.types import CellArray, Number, UnitConversionTable, UnitSystem
from bores.utils import scale
from bores.wells.state import WellStates
from bores.wells.workspace import load_wells_states

if typing.TYPE_CHECKING:
    from bores.simulation.workspace import SimulationWorkspace

__all__ = ["BlackOilModelState", "load_model_state"]

_UNSET: typing.Any = object()


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


def load_model_state(
    model: BlackOilModel,
    compiled_model: CompiledBlackOilModel,
    workspace: "SimulationWorkspace",
    *,
    temperature: CellArray,
    time: Number = 0.0,
    hysteresis: Hysteresis | HysteresisWorkspace | None = _UNSET,
    dtype: npt.DTypeLike = None,
) -> BlackOilModelState:
    """
    Load a `BlackOilModelState` snapshot from a running simulation's workspace.

    :param model: The original rich model `compiled_model` was compiled
        from. Supplies `wells.wells` (each well's own `Well`/`Perforation`
        objects), which the compiled layer doesn't retain a reference to.
    :param compiled_model: The compiled model `workspace` was built for.
        Supplies `reservoir.pore_volumes`, `reservoir.regions.pvt_region`,
        `fluid.pvt`, and `wells` (for `load_wells_states`).
    :param workspace: The running simulation's workspace, read for
        `workspace.reservoir` and `workspace.wells`.
    :param temperature: Reservoir temperature per cell. See `load_reservoir_state`.
    :param time: Simulation time this state corresponds to, in `compiled_model.unit_system`.
    :param hysteresis: Hysteresis history to include in the reservoir
        state. Defaults to `workspace.hysteresis`, the run's own tracked
        history, if any. Pass `None` explicitly to omit it even if the
        workspace has one.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `BlackOilModelState` combining the current reservoir state
        and, if the model has wells, the current well states. `wells` is
        `None` if the model has no wells.
    :raises ValidationError: If `model.wells` or `compiled_model.wells` is
        set but the other isn't. Both or neither must have wells.
    """
    if (model.wells is None) != (compiled_model.wells is None):
        raise ValidationError(
            "`model.wells` and `compiled_model.wells` must either both be "
            "set or both be `None`. One is set and the other isn't."
        )
    if hysteresis is _UNSET:
        hysteresis = workspace.hysteresis

    reservoir_regions = compiled_model.reservoir.regions
    reservoir_state = load_reservoir_state(
        workspace.reservoir,
        temperature=temperature,
        pore_volumes=compiled_model.reservoir.pore_volumes,
        pvt=compiled_model.fluid.pvt,
        pvt_region_index=reservoir_regions.pvt_region if reservoir_regions is not None else None,
        unit_system=compiled_model.unit_system,
        hysteresis=hysteresis,
        dtype=dtype,
    )

    wells_state: WellStates | None = None
    if model.wells is not None and compiled_model.wells is not None:
        wells_state = load_wells_states(
            wells=model.wells.wells,
            compiled_system=compiled_model.wells,
            workspace=workspace.wells,
        )

    return BlackOilModelState(reservoir=reservoir_state, wells=wells_state, time=time)
