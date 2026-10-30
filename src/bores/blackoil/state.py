"""Dynamic simulation state for one point in time: reservoir state plus optional well states."""

import typing

import attrs
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.compile import CompiledBlackOilModel
from bores.blackoil.model import BlackOilModel
from bores.dataframes import AnyDataFrame, Backend, resolve_backend
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.reservoir.state import as_dataframe as reservoir_as_dataframe
from bores.reservoir.workspace import HysteresisWorkspace, load_reservoir_state
from bores.serde.base import Serializable
from bores.types import Boolean, CellArray, Integer, Number, UnitConversionTable, UnitSystem
from bores.units import get_conversion_factors, resolve_unit_conversion_table
from bores.utils import scale
from bores.wells.state import WellStates
from bores.wells.state import as_dataframe as wells_as_dataframe
from bores.wells.workspace import load_wells_states

if typing.TYPE_CHECKING:
    from bores.simulation.workspace import SimulationWorkspace

__all__ = ["BlackOilModelState", "as_dataframe", "load_model_state"]

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
        table = resolve_unit_conversion_table(table)
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
        Supplies `reservoir.compute_pore_volumes`, `reservoir.regions.pvt_region`,
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
        pore_volumes=compiled_model.reservoir.compute_pore_volumes(workspace.reservoir.pressure),
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


ModelStatePart = typing.Literal["reservoir", "wells", "perforations"]
"""Which part of a `BlackOilModelState` `as_dataframe` reports."""


def as_dataframe(
    states: BlackOilModelState | typing.Iterable[BlackOilModelState],
    *,
    backend: Backend = "pandas",
    part: ModelStatePart = "reservoir",
    fields: typing.Sequence[str] | typing.Literal["all"] | None = None,
    cells: typing.Sequence[Integer] | npt.NDArray | None = None,
    active_only: Boolean = False,
    wells: str | typing.Sequence[str] | None = None,
    open_only: Boolean = False,
    grid: Grid | None = None,
    centroids: Boolean = False,
    unit_system: UnitSystem | None = None,
    dtype: npt.DTypeLike = None,
    categorical: Boolean = False,
) -> AnyDataFrame:
    """
    Turns one model state, or a sequence of them such as the states a run yields, into a pandas
    or polars data frame of one part of the model.

    A reservoir and its wells are different shapes of data, one row for each cell and one row
    for each well or perforation, so they are not forced into one table. Choose the part with
    `part`. Every row carries the `time` of the state it came from as its first column.

    - `"reservoir"`: one row for each cell of each state, as `bores.reservoir.state.as_dataframe`.
    - `"wells"`: one row for each well of each state, as `bores.wells.state.as_dataframe` at the
      well level.
    - `"perforations"`: one row for each open perforation of each state, as
      `bores.wells.state.as_dataframe` at the perforation level.

    States that carry no well states add no rows to the `"wells"` and `"perforations"` parts.

    :param states: A model state, or states in time order.
    :param backend: `"pandas"` or `"polars"`. The library must be installed.
    :param part: `"reservoir"`, `"wells"` or `"perforations"`.
    :param fields: Reservoir part only. Fields to include, or `"all"`. Defaults to pressure,
        temperature, the saturations, solution gas-oil ratio and bubble point pressure.
    :param cells: Reservoir part only. Cells to include, as flat indices or a boolean mask.
    :param active_only: Reservoir part only. Leave out the cells the grid marks inactive.
    :param wells: Wells parts only. Well names or patterns such as `"PROD*"` to include.
    :param open_only: `"wells"` part only. Leave out shut wells.
    :param grid: The grid the model uses. Adds `i`, `j` and `k` to the reservoir and
        perforation parts, and is needed by `active_only` and `centroids`.
    :param centroids: Add cell centroid coordinates, for the reservoir and perforation parts.
    :param unit_system: Unit system to report in. Each state, and so its `time`, is converted to
        it. If not given, the states must all share one unit system, which is used as it is.
    :param dtype: Reservoir part only. Floating point dtype of the field columns.
    :param categorical: Wells parts only. Store text columns as categoricals.
    :returns: A `pandas.DataFrame` or `polars.DataFrame`.
    :raises ValidationError: If an option does not apply to `part`, the states differ in unit
        system, or none of the states carries well states for a wells part.
    :raises ImportError: If the library behind `backend` is not installed.
    """
    resolve_backend(backend)
    if part not in ("reservoir", "wells", "perforations"):
        raise ValidationError(
            f"`part` must be 'reservoir', 'wells' or 'perforations'; got {part!r}."
        )

    inapplicable: dict[str, bool]
    if part == "reservoir":
        inapplicable = {
            "wells": wells is not None,
            "open_only": open_only,
            "categorical": categorical,
        }
    else:
        inapplicable = {
            "fields": fields is not None,
            "cells": cells is not None,
            "active_only": active_only,
            "dtype": dtype is not None,
        }
        if part == "wells":
            inapplicable |= {"grid": grid is not None, "centroids": centroids}
        else:
            inapplicable["open_only"] = open_only
    misplaced = [name for name, given in inapplicable.items() if given]
    if misplaced:
        raise ValidationError(f"{misplaced} do not apply to `part={part!r}`.")

    snapshots = [states] if isinstance(states, BlackOilModelState) else list(states)
    for position, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, BlackOilModelState):
            raise ValidationError(
                f"`states` must hold `BlackOilModelState`s; item {position} is a "
                f"{type(snapshot).__name__}."
            )
    if unit_system is None:
        systems = {snapshot.unit_system for snapshot in snapshots}
        if len(systems) > 1:
            raise ValidationError(
                "The states use different unit systems, "
                f"{sorted(system.value for system in systems)}. Pass `unit_system` to report "
                "them all in one."
            )
    else:
        snapshots = [snapshot.convert(unit_system) for snapshot in snapshots]

    if part == "reservoir":
        return reservoir_as_dataframe(
            [snapshot.reservoir for snapshot in snapshots],
            backend=backend,
            fields=fields,
            times=[snapshot.time for snapshot in snapshots],
            cells=cells,
            grid=grid,
            active_only=active_only,
            centroids=centroids,
            dtype=dtype,
        )

    with_wells = [snapshot for snapshot in snapshots if snapshot.wells is not None]
    if snapshots and not with_wells:
        raise ValidationError(
            f"None of the states carries well states, so `part={part!r}` is empty."
        )
    return wells_as_dataframe(
        [typing.cast("WellStates", snapshot.wells) for snapshot in with_wells],
        backend=backend,
        level="well" if part == "wells" else "perforation",
        times=[snapshot.time for snapshot in with_wells],
        wells=wells,
        open_only=open_only,
        grid=grid,
        centroids=centroids,
        categorical=categorical,
    )
