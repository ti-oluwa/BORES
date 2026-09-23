"""Compiles a `BlackOilModel` into a `CompiledBlackOilModel`."""

import typing

import numpy.typing as npt

from bores.blackoil.fluids.model import BlackOil
from bores.blackoil.model import BlackOilModel
from bores.errors import ModelCompilationError
from bores.reservoir.boundary.compile import (
    CompiledBoundaryConditions,
    compile_boundary_conditions,
)
from bores.reservoir.model import Reservoir
from bores.types import GridIntersectionMethod, Number, Orientation, UnitSystem
from bores.wells.compile import CompiledWellSystem, compile_well_system

__all__ = ["CompiledBlackOilModel", "compile_model"]


class CompiledBlackOilModel(typing.NamedTuple):
    """A `BlackOilModel` with its compilable components compiled."""

    reservoir: Reservoir
    """The reservoir model."""

    fluid: BlackOil
    """The fluid model."""

    wells: CompiledWellSystem | None
    """The compiled `Wells` model, or `None` if the original model had no wells."""

    boundary_conditions: CompiledBoundaryConditions | None
    """The compiled boundary conditions, or `None` if the original model had none defined."""

    unit_system: UnitSystem
    """The model's unit system."""


def compile_model(
    model: BlackOilModel,
    *,
    horizontal_tolerance: Number | None = None,
    intersection_method: GridIntersectionMethod = "aabb",
    search_radius: Number | None = None,
    dtype: npt.DTypeLike = None,
) -> CompiledBlackOilModel:
    """
    Validate and compile a `BlackOilModel` into a `CompiledBlackOilModel`.

    This function validates the input model, compiles any boundary conditions and
    well data into optimized runtime structures, and returns the original
    reservoir and fluid models together with those compiled components.

    :param model: The black-oil model to validate and compile.
    :param horizontal_tolerance: Optional horizontal tolerance forwarded to
        `compile_well_system` during well-cell intersection checks.
    :param intersection_method: The grid-intersection strategy passed to
        `compile_well_system`.
    :param search_radius: Optional search radius forwarded to
        `compile_well_system` when resolving well-grid intersections.
    :param dtype: NumPy dtype used when constructing compiled boundary condition
        and well arrays.
    :returns: A compiled black-oil model containing the original reservoir and
        fluid plus any compiled wells and boundary conditions.
    :raises ModelCompilationError: If boundary condition or well syatem compilation fails.
    """
    model.validate()
    reservoir = model.reservoir
    compiled_boundary_conditions = None
    if model.boundary_conditions is not None:
        try:
            compiled_boundary_conditions = compile_boundary_conditions(
                boundary_conditions=model.boundary_conditions,
                reservoir=reservoir,
                dtype=dtype,
            )
        except Exception as exc:
            raise ModelCompilationError(
                f"Failed to compile boundary conditions for {model!r}."
            ) from exc

    compiled_wells = None
    if model.wells is not None:
        permeability = reservoir.rock.absolute_permeability
        wells = model.wells
        try:
            compiled_wells = compile_well_system(
                wells=wells.wells,
                controls=wells.well_controls,
                grid=reservoir.grid,
                permeabilities={
                    Orientation.X: permeability.x,
                    Orientation.Y: permeability.y,
                    Orientation.Z: permeability.z,
                },
                group_controls=wells.group_controls,
                groups=wells.groups,
                horizontal_tolerance=horizontal_tolerance,
                intersection_method=intersection_method,
                search_radius=search_radius,
                dtype=dtype,
            )
        except Exception as exc:
            raise ModelCompilationError(f"Failed to compile wells for {model!r}.") from exc

    return CompiledBlackOilModel(
        reservoir=reservoir,
        fluid=model.fluid,
        wells=compiled_wells,
        boundary_conditions=compiled_boundary_conditions,
        unit_system=model.unit_system,
    )
