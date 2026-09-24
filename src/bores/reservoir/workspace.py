"""Reusable workspace for one simulation run."""

import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.precision import get_dtype
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.types import BooleanCellArray, CellArray, NumberArray, OneDimension

__all__ = [
    "HysteresisWorkspace",
    "ReservoirWorkspace",
    "build_reservoir_workspace",
    "load_reservoir_state",
]


class HysteresisWorkspace(typing.NamedTuple):
    """
    Drainage/imbibition hysteresis tracking for Killough scanning curves.

    Maintains historical saturation extrema and displacement-regime flags
    required to compute effective residual saturations on the scanning curves.

    All arrays are dimensionless (saturations, flags) and therefore require
    no unit conversion.
    """

    max_water_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum water saturation reached in each
    cell (fraction).

    Updated whenever the current water saturation exceeds the stored maximum. 
    Determines the imbibition end-point on the scanning curve when drainage reverses.
    """

    max_gas_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum gas saturation reached in each
    cell (fraction).

    Analogous to `max_water_saturation` for the gas phase.
    """

    water_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - `True` if the current water-phase displacement is
    imbibition (water saturation increasing toward `max_water_saturation`).

    `False` indicates drainage (water saturation decreasing).
    """

    gas_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - `True` if gas saturation is currently increasing
    toward `max_gas_saturation`, `False` if decreasing (drainage).

    Same convention as `water_imbibition_flag`: the flag tracks that
    phase's own saturation, not a wetting/non-wetting distinction.
    """

    water_reversal_saturation: CellArray
    """
    Shape (n_cells,) - water saturation at the most recent
    drainage-to-imbibition (or reverse) reversal point (fraction).

    Starting saturation of the Killough scanning curve when the displacement
    regime changes.
    """

    gas_reversal_saturation: CellArray
    """
    Shape (n_cells,) - gas saturation at the most recent reversal point
    (fraction).

    Analogous to `water_reversal_saturation` for the gas phase.
    """

    @classmethod
    def from_state(cls, state: Hysteresis, *, dtype: npt.DTypeLike = None) -> Self:
        """
        Create a workspace seeded from an immutable hysteresis state.

        :param state: The hysteresis state to seed the workspace from.
        :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
        :returns: `HysteresisWorkspace` seeded from `state`.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        return cls(
            max_water_saturation=typing.cast(
                CellArray, np.array(state.max_water_saturation, dtype=dtype, copy=True)
            ),
            max_gas_saturation=typing.cast(
                CellArray, np.array(state.max_gas_saturation, dtype=dtype, copy=True)
            ),
            water_imbibition_flag=typing.cast(
                BooleanCellArray,
                np.array(state.water_imbibition_flag, dtype=np.bool_, copy=True),
            ),
            gas_imbibition_flag=typing.cast(
                BooleanCellArray,
                np.array(state.gas_imbibition_flag, dtype=np.bool_, copy=True),
            ),
            water_reversal_saturation=typing.cast(
                CellArray, np.array(state.water_reversal_saturation, dtype=dtype, copy=True)
            ),
            gas_reversal_saturation=typing.cast(
                CellArray, np.array(state.gas_reversal_saturation, dtype=dtype, copy=True)
            ),
        )

    def snapshot(self, *, dtype: npt.DTypeLike = None) -> Hysteresis:
        """
        Return an independent immutable snapshot of the current history.

        :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
        :returns: An immutable snapshot of the current hysteresis history.
        """
        return load_hysteresis(self, dtype=dtype)

    def update(
        self,
        *,
        water_saturation: CellArray,
        gas_saturation: CellArray,
        previous_water_saturation: CellArray,
        previous_gas_saturation: CellArray,
    ) -> None:
        """
        Update the workspace in-place from current and previous saturations.

        :param water_saturation: Current water saturation per cell (fraction).
        :param gas_saturation: Current gas saturation per cell (fraction).
        :param previous_water_saturation: Previous water saturation per cell (fraction).
        :param previous_gas_saturation: Previous gas saturation per cell (fraction).
        """
        update_hysteresis_workspace(
            workspace=self,
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            previous_water_saturation=previous_water_saturation,
            previous_gas_saturation=previous_gas_saturation,
        )


def update_phase_hysteresis(
    new_saturation: CellArray,
    previous_saturation: CellArray,
    max_saturation: CellArray,
    imbibition_flag: BooleanCellArray,
    reversal_saturation: CellArray,
) -> None:
    """
    Update one phase's hysteresis arrays in place for every cell.

    The reversal array is updated before the historical maximum so a
    drainage reversal records the maximum from before this timestep.

    :param new_saturation: Current phase saturation per cell.
    :param previous_saturation: Previous phase saturation per cell.
    :param max_saturation: Historical maximum, updated in place.
    :param imbibition_flag: Imbibition flags, updated in place.
    :param reversal_saturation: Reversal saturations, updated in place.
    """
    increasing = new_saturation > previous_saturation
    decreasing = new_saturation < previous_saturation
    turned_to_drainage = imbibition_flag & decreasing
    turned_to_imbibition = ~imbibition_flag & increasing

    np.copyto(reversal_saturation, max_saturation, where=turned_to_drainage)
    np.copyto(reversal_saturation, previous_saturation, where=turned_to_imbibition)
    np.maximum(max_saturation, new_saturation, out=max_saturation)
    np.copyto(imbibition_flag, True, where=increasing)
    np.copyto(imbibition_flag, False, where=decreasing)


def update_hysteresis_workspace(
    workspace: HysteresisWorkspace,
    *,
    water_saturation: CellArray,
    gas_saturation: CellArray,
    previous_water_saturation: CellArray,
    previous_gas_saturation: CellArray,
) -> None:
    """
    Update a `HysteresisWorkspace` in-place from current and previous saturations.

    :param workspace: The hysteresis workspace to update.
    :param water_saturation: Current water saturation per cell (fraction).
    :param gas_saturation: Current gas saturation per cell (fraction).
    :param previous_water_saturation: Previous water saturation per cell (fraction).
    :param previous_gas_saturation: Previous gas saturation per cell (fraction).
    """
    update_phase_hysteresis(
        new_saturation=water_saturation,
        previous_saturation=previous_water_saturation,
        max_saturation=workspace.max_water_saturation,
        imbibition_flag=workspace.water_imbibition_flag,
        reversal_saturation=workspace.water_reversal_saturation,
    )
    update_phase_hysteresis(
        new_saturation=gas_saturation,
        previous_saturation=previous_gas_saturation,
        max_saturation=workspace.max_gas_saturation,
        imbibition_flag=workspace.gas_imbibition_flag,
        reversal_saturation=workspace.gas_reversal_saturation,
    )


class ReservoirWorkspace(typing.NamedTuple):
    """
    Per-cell reservoir primary unknowns, one row per cell, for the whole run.

    The reservoir state (`ReservoirState`) is decompiled from whatever this holds at a
    reported time.
    """

    pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Oil-phase reference pressure."""

    oil_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    water_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    gas_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    solution_gor: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rs."""

    oil_bubble_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    vaporized_oil_gas_ratio: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rv."""

    gas_dew_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    gas_solubility_in_water: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rsw."""

    water_bubble_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    @classmethod
    def from_state(cls, state: ReservoirState, *, dtype: npt.DTypeLike = None) -> Self:
        """
        Create a workspace seeded from an immutable reservoir state.

        :param state: The reservoir state to seed the workspace from.
        :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
        :returns: `ReservoirWorkspace` seeded from `state`.
        """
        return typing.cast(Self, build_reservoir_workspace(state=state, dtype=dtype))

    def snapshot(
        self,
        *,
        hysteresis: Hysteresis | HysteresisWorkspace | None = None,
        dtype: npt.DTypeLike = None,
    ) -> ReservoirState:
        """
        Return an independent snapshot of the current reservoir state.

        :param hysteresis: Optional hysteresis history to include in the state.
            If not given, the state will have no hysteresis history.
        :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
        :returns: An immutable snapshot of the current reservoir state.
        """
        return load_reservoir_state(self, hysteresis=hysteresis, dtype=dtype)


def build_reservoir_workspace(
    *, state: ReservoirState, dtype: npt.DTypeLike = None
) -> ReservoirWorkspace:
    """
    Builds a `ReservoirWorkspace` from a reservoir state's own primary unknowns.

    :param state: The reservoir state to seed the workspace from.
    :param n_cells: Number of grid cells.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `ReservoirWorkspace` seeded from `state`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return ReservoirWorkspace(
        pressure=state.pressure.astype(dtype, copy=True),
        oil_saturation=state.oil_saturation.astype(dtype, copy=True),
        water_saturation=state.water_saturation.astype(dtype, copy=True),
        gas_saturation=state.gas_saturation.astype(dtype, copy=True),
        solution_gor=state.solution_gor.astype(dtype, copy=True),
        oil_bubble_point_pressure=state.oil_bubble_point_pressure.astype(dtype, copy=True),
        vaporized_oil_gas_ratio=state.vaporized_oil_gas_ratio.astype(dtype, copy=True),
        gas_dew_point_pressure=state.gas_dew_point_pressure.astype(dtype, copy=True),
        gas_solubility_in_water=state.gas_solubility_in_water.astype(dtype, copy=True),
        water_bubble_point_pressure=state.water_bubble_point_pressure.astype(dtype, copy=True),
    )


def load_hysteresis(workspace: HysteresisWorkspace, *, dtype: npt.DTypeLike = None) -> Hysteresis:
    """
    Load an independent immutable snapshot of the current hysteresis history from a workspace.

    :param workspace: The hysteresis workspace to snapshot.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: An immutable snapshot of the current hysteresis history.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return Hysteresis(
        max_water_saturation=typing.cast(
            CellArray, np.array(workspace.max_water_saturation, dtype=dtype, copy=True)
        ),
        max_gas_saturation=typing.cast(
            CellArray, np.array(workspace.max_gas_saturation, dtype=dtype, copy=True)
        ),
        water_imbibition_flag=workspace.water_imbibition_flag.copy(),
        gas_imbibition_flag=workspace.gas_imbibition_flag.copy(),
        water_reversal_saturation=typing.cast(
            CellArray, np.array(workspace.water_reversal_saturation, dtype=dtype, copy=True)
        ),
        gas_reversal_saturation=typing.cast(
            CellArray, np.array(workspace.gas_reversal_saturation, dtype=dtype, copy=True)
        ),
    )


def load_reservoir_state(
    workspace: ReservoirWorkspace,
    *,
    hysteresis: Hysteresis | HysteresisWorkspace | None = None,
    dtype: npt.DTypeLike = None,
) -> ReservoirState:
    """
    Load an independent snapshot of the current reservoir state from a workspace.

    :param workspace: The reservoir workspace to snapshot.
    :param hysteresis: Optional hysteresis history to include in the state.
        If not given, the state will have no hysteresis history.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: An immutable snapshot of the current reservoir state.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    if isinstance(hysteresis, HysteresisWorkspace):
        hysteresis = hysteresis.snapshot(dtype=dtype)
    # return ReservoirState(
    #     pressure=workspace.pressure.astype(dtype, copy=True),
    #     oil_saturation=workspace.oil_saturation.astype(dtype, copy=True),
    #     water_saturation=workspace.water_saturation.astype(dtype, copy=True),
    #     gas_saturation=workspace.gas_saturation.astype(dtype, copy=True),
    #     solution_gor=workspace.solution_gor.astype(dtype, copy=True),
    #     oil_bubble_point_pressure=workspace.oil_bubble_point_pressure.astype(dtype, copy=True),
    #     vaporized_oil_gas_ratio=workspace.vaporized_oil_gas_ratio.astype(dtype, copy=True),
    #     gas_dew_point_pressure=workspace.gas_dew_point_pressure.astype(dtype, copy=True),
    #     gas_solubility_in_water=workspace.gas_solubility_in_water.astype(dtype, copy=True),
    #     water_bubble_point_pressure=workspace.water_bubble_point_pressure.astype(dtype, copy=True),
    #     hysteresis=hysteresis,
    # )
    ...
