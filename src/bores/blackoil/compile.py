"""Compiles a `BlackOilModel` into a `CompiledBlackOilModel`."""

import typing

import numpy.typing as npt

from bores.blackoil.fluids.model import BlackOil
from bores.blackoil.model import BlackOilModel
from bores.errors import BoundaryConditionCompilationError, WellCompilationError
from bores.reservoir.model import Reservoir
from bores.types import GridIntersectionMethod, Number, Orientation, UnitSystem
from bores.wells.compile import CompiledWellSystem, compile_well_system

__all__ = ["CompiledBlackOilModel", "compile_model"]


class CompiledBlackOilModel(typing.NamedTuple):
    """A `BlackOilModel` with its compilable components compiled."""

    reservoir: Reservoir
    """Unchanged from `BlackOilModel.reservoir`."""

    fluid: BlackOil
    """Unchanged from `BlackOilModel.fluid`."""

    wells: CompiledWellSystem | None
    """The compiled form of `BlackOilModel.wells`, or `None` if there were none."""

    boundary_conditions: None
    """Always `None`. `CompiledBoundaryConditions` isn't built yet."""

    unit_system: UnitSystem
    """Unchanged from `BlackOilModel.unit_system`."""


def compile_model(
    model: BlackOilModel,
    *,
    horizontal_tolerance: Number | None = None,
    intersection_method: GridIntersectionMethod = "aabb",
    search_radius: Number | None = None,
    dtype: npt.DTypeLike = None,
) -> CompiledBlackOilModel:
    """
    Compiles a `BlackOilModel`'s wells into a `CompiledBlackOilModel`.

    Permeability for well-index resolution is read from
    `model.reservoir.rock.absolute_permeability`.

    :param model: The model to compile.
    :param dtype: Forwarded to `compile_well_system`.
    :param resolve_kwargs: Forwarded to `compile_well_system`.
    :returns: The compiled model.
    :raises BoundaryConditionCompilationError: If `model.boundary_conditions` is set.
    :raises WellCompilationError: If well compilation fails.
    """
    model.validate()
    if model.boundary_conditions is not None:
        raise BoundaryConditionCompilationError(
            "`CompiledBoundaryConditions` does not exist yet. Compile a model with "
            "boundary_conditions=None."
        )

    compiled_wells = None
    if model.wells is not None:
        permeability = model.reservoir.rock.absolute_permeability
        try:
            compiled_wells = compile_well_system(
                wells=model.wells.wells,
                controls=model.wells.well_controls,
                grid=model.reservoir.grid,
                permeabilities={
                    Orientation.X: permeability.x,
                    Orientation.Y: permeability.y,
                    Orientation.Z: permeability.z,
                },
                group_controls=model.wells.group_controls,
                groups=model.wells.groups,
                horizontal_tolerance=horizontal_tolerance,
                intersection_method=intersection_method,
                search_radius=search_radius,
                dtype=dtype,
            )
        except Exception as exc:
            raise WellCompilationError(f"Failed to compile wells for {model!r}.") from exc

    return CompiledBlackOilModel(
        reservoir=model.reservoir,
        fluid=model.fluid,
        wells=compiled_wells,
        boundary_conditions=None,
        unit_system=model.unit_system,
    )
