"""Model state management."""

import logging
import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.datastructures import BottomHolePressures, FormationVolumeFactors, Rates
from bores.errors import ValidationError
from bores.grids.base import (
    CapillaryPressureGrids,
    RelativeMobilityGrids,
    RelPermGrids,
)
from bores.material_balance import MaterialBalanceErrors
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockPermeability,
    RockProperties,
    SaturationHistory,
)
from bores.precision import get_dtype
from bores.serialization import Serializable
from bores.timing import TimerState
from bores.types import NDimension
from bores.wells.base import Wells

logger = logging.getLogger(__name__)


__all__ = ["ModelState", "validate_state"]


@typing.final
@attrs.frozen(slots=True, eq=False, hash=False)
class ModelState(
    Serializable,
    typing.Generic[NDimension],
    dump_exclude={"_well_exists"},
    load_exclude={"_well_exists"},
):
    """A captured state of the reservoir model at a specific time step during a simulation run."""

    step: int
    """The time step index"""
    step_size: float
    """The time step size in seconds"""
    time: float
    """The simulation time reached in seconds. Time elapsed at this state in seconds"""
    model: ReservoirModel[NDimension]
    """The reservoir model at this state"""
    wells: Wells[NDimension]
    """The wells configuration at this state"""
    injection_rates: Rates[float, NDimension]
    """Holds sparse tensors for oil, water, and gas injection rates at this state in ft³/day"""
    production_rates: Rates[float, NDimension]
    """Holds sparse tensors for oil, water, and gas production rates at this state in ft³/day"""
    injection_formation_volume_factors: FormationVolumeFactors[float, NDimension]
    """Holds sparse tensors for oil, water, and gas injection formation volume factors at this state in ft³/SCF or bbls/STB"""
    production_formation_volume_factors: FormationVolumeFactors[float, NDimension]
    """Holds sparse tensors for oil, water, and gas production formation volume factors at this state in ft³/SCF or bbls/STB"""
    injection_bhps: BottomHolePressures[float, NDimension]
    """Sparse tensor holding injection well(s) cell perforation bottom hole pressure at this state in psi"""
    production_bhps: BottomHolePressures[float, NDimension]
    """Sparse tensor holding production well(s) cell perforation bottom hole pressure at this state in psi"""
    relative_permeabilities: RelPermGrids[NDimension]
    """Relative permeabilities at this state"""
    relative_mobilities: RelativeMobilityGrids[NDimension]
    """Relative mobilities at this state"""
    capillary_pressures: CapillaryPressureGrids[NDimension]
    """Capillary pressures at this state"""
    material_balance_errors: MaterialBalanceErrors
    """Material balance error for the model simulation at this time step"""
    timer_state: typing.Optional[TimerState] = None
    """Optional timer configuration state at this model state"""

    _well_exists: typing.Optional[bool] = attrs.field(default=None, init=False)

    def wells_exists(self) -> bool:
        """Check if there are any wells in this state."""
        if self._well_exists is None:
            object.__setattr__(self, "_well_exists", self.wells.exists())
        return self._well_exists  # type: ignore[return-value]

    # Quick access properties for commonly queried derived quantities
    @property
    def time_in_days(self) -> float:
        """Time elapsed at this state in days"""
        return self.time / 86400

    @property
    def time_in_weeks(self) -> float:
        """Time elapsed at this state in weeks"""
        return self.time / 7 * 86400

    @property
    def time_in_years(self) -> float:
        """Time elapsed at this state in years"""
        return self.time / c.DAYS_PER_YEAR * 86400

    @property
    def injection_fvfs(self) -> FormationVolumeFactors[float, NDimension]:
        """Formation volume factors for injection phases at this state in ft³/SCF or bbls/STB"""
        return self.injection_formation_volume_factors

    @property
    def production_fvfs(self) -> FormationVolumeFactors[float, NDimension]:
        """Formation volume factors for production phases at this state in ft³/SCF or bbls/STB"""
        return self.production_formation_volume_factors

    @property
    def average_pressure(self) -> float:
        """Average reservoir pressure at this state in psi, computed as the mean of the pressure grid."""
        return float(np.mean(self.model.fluid_properties.pressure_grid))

    @property
    def average_temperature(self) -> float:
        """Average reservoir temperature at this state in °F, computed as the mean of the temperature grid."""
        return float(np.mean(self.model.fluid_properties.temperature_grid))

    @property
    def average_water_saturation(self) -> float:
        """Average water saturation at this state, computed as the mean of the water saturation grid."""
        return float(np.mean(self.model.fluid_properties.water_saturation_grid))

    @property
    def average_oil_saturation(self) -> float:
        """Average oil saturation at this state, computed as the mean of the oil saturation grid."""
        return float(np.mean(self.model.fluid_properties.oil_saturation_grid))

    @property
    def average_gas_saturation(self) -> float:
        """Average gas saturation at this state, computed as the mean of the gas saturation grid."""
        return float(np.mean(self.model.fluid_properties.gas_saturation_grid))

    @property
    def mbe(self) -> MaterialBalanceErrors:
        """Model simulation material balance errors at this timestep."""
        return self.material_balance_errors


def _validate_array(
    model_shape: tuple[int, ...],
    grid: np.ndarray,
    field_name: str,
    dtype: typing.Optional[npt.DTypeLike],
) -> np.ndarray:
    if grid.shape != model_shape:
        raise ValidationError(
            f"{field_name} has shape {grid.shape}, expected {model_shape}."
        )
    if dtype is not None and np.issubdtype(grid.dtype, np.floating):
        grid = grid.astype(dtype, copy=False, order="C")
    return np.ascontiguousarray(grid)


def validate_state(
    state: ModelState[NDimension],
    dtype: typing.Optional[
        typing.Union[npt.DTypeLike, typing.Literal["global"]]
    ] = None,
) -> ModelState[NDimension]:
    """
    Validate state grids have matching shapes and optionally coerce to specified dtype.

    :param state: `ModelState` to validate
    :param dtype: Optional dtype to coerce all array fields to. If None, no coercion is performed.
    :return: Validated (and optionally coerced) `ModelState`
    """
    # Check that all grids have matching shapes
    model = state.model
    model_shape = model.grid_shape
    fluid_properties = model.fluid_properties
    rock_properties = model.rock_properties
    relative_mobilities = state.relative_mobilities
    relative_permeabilities = state.relative_permeabilities
    capillary_pressures = state.capillary_pressures
    thickness_grid = model.thickness_grid
    if thickness_grid.shape != model_shape:
        raise ValidationError(
            f"Thickness grid has shape {thickness_grid.shape}, expected {model_shape}."
        )

    if dtype == "global":
        dtype = get_dtype()

    # Validate and coerce fluid properties
    if dtype is not None:
        fluid_dict = {}
        for field in attrs.fields(fluid_properties.__class__):
            value = getattr(fluid_properties, field.name)
            if isinstance(value, np.ndarray):
                fluid_dict[field.name] = _validate_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Fluid property grid {field.name}",
                    dtype=dtype,
                )
            else:
                fluid_dict[field.name] = value
        fluid_properties = FluidProperties(**fluid_dict)
    else:
        for field in attrs.fields(fluid_properties.__class__):
            grid = getattr(fluid_properties, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Fluid property grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce rock properties
    if dtype is not None:
        rock_dict = {}
        for field in attrs.fields(rock_properties.__class__):
            value = getattr(rock_properties, field.name)
            if isinstance(value, np.ndarray):
                rock_dict[field.name] = _validate_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Rock property grid {field.name}",
                    dtype=dtype,
                )
            elif field.name == "absolute_permeability":
                perm = value
                rock_dict[field.name] = RockPermeability(
                    x=_validate_array(
                        model_shape=model_shape,
                        grid=perm.x,
                        field_name="Rock permeability x",
                        dtype=dtype,
                    ),
                    y=_validate_array(
                        model_shape=model_shape,
                        grid=perm.y,
                        field_name="Rock permeability y",
                        dtype=dtype,
                    ),
                    z=_validate_array(
                        model_shape=model_shape,
                        grid=perm.z,
                        field_name="Rock permeability z",
                        dtype=dtype,
                    ),
                )
            else:
                rock_dict[field.name] = value
        rock_properties = RockProperties(**rock_dict)
    else:
        for field in attrs.fields(rock_properties.__class__):
            grid = getattr(rock_properties, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Rock property grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce relative mobilities
    if dtype is not None:
        mobility_dict = {}
        for field in attrs.fields(relative_mobilities.__class__):
            value = getattr(relative_mobilities, field.name)
            if isinstance(value, np.ndarray):
                mobility_dict[field.name] = _validate_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Relative mobility grid {field.name}",
                    dtype=dtype,
                )
            else:
                mobility_dict[field.name] = value
        relative_mobilities = RelativeMobilityGrids(**mobility_dict)
    else:
        for field in attrs.fields(relative_mobilities.__class__):
            grid = getattr(relative_mobilities, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Relative mobility grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce relative permeabilities
    if dtype is not None:
        relperm_dict = {}
        for field in attrs.fields(relative_permeabilities.__class__):
            value = getattr(relative_permeabilities, field.name)
            if isinstance(value, np.ndarray):
                relperm_dict[field.name] = _validate_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Relative permeability grid {field.name}",
                    dtype=dtype,
                )
            else:
                relperm_dict[field.name] = value
        relative_permeabilities = RelPermGrids(**relperm_dict)
    else:
        for field in attrs.fields(relative_permeabilities.__class__):
            grid = getattr(relative_permeabilities, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Relative permeability grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce capillary pressures
    if dtype is not None:
        capillary_dict = {}
        for field in attrs.fields(capillary_pressures.__class__):
            value = getattr(capillary_pressures, field.name)
            if isinstance(value, np.ndarray):
                capillary_dict[field.name] = _validate_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Capillary pressure grid {field.name}",
                    dtype=dtype,
                )
            else:
                capillary_dict[field.name] = value
        capillary_pressures = CapillaryPressureGrids(**capillary_dict)
    else:
        for field in attrs.fields(capillary_pressures.__class__):
            grid = getattr(capillary_pressures, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Capillary pressure grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce thickness grid and saturation history
    if dtype is not None:
        thickness_grid = _validate_array(
            model_shape=model_shape,
            grid=thickness_grid,
            field_name="Thickness grid",
            dtype=dtype,
        )
        sat_hist = model.saturation_history
        saturation_history = SaturationHistory(
            max_water_saturation_grid=_validate_array(
                model_shape=model_shape,
                grid=sat_hist.max_water_saturation_grid,
                field_name="Max water saturation grid",
                dtype=dtype,
            ),
            max_gas_saturation_grid=_validate_array(
                model_shape=model_shape,
                grid=sat_hist.max_gas_saturation_grid,
                field_name="Max gas saturation grid",
                dtype=dtype,
            ),
            water_imbibition_flag_grid=_validate_array(
                model_shape=model_shape,
                grid=sat_hist.water_imbibition_flag_grid,
                field_name="Water imbibition flag grid",
                dtype=dtype,
            ),
            gas_imbibition_flag_grid=_validate_array(
                model_shape=model_shape,
                grid=sat_hist.gas_imbibition_flag_grid,
                field_name="Gas imbibition flag grid",
                dtype=dtype,
            ),
        )
        # Reconstruct model and model state with coerced data
        model = ReservoirModel(
            grid_shape=model.grid_shape,
            cell_dimension=model.cell_dimension,
            thickness_grid=thickness_grid,
            fluid_properties=fluid_properties,  # type:ignore[arg-type]
            rock_properties=rock_properties,  # type:ignore[arg-type]
            saturation_history=saturation_history,
            boundary_conditions=model.boundary_conditions,  # type: ignore
            dip_angle=model.dip_angle,
            dip_azimuth=model.dip_azimuth,
        )
        state = ModelState(
            step=state.step,
            step_size=state.step_size,
            time=state.time,
            model=model,
            wells=state.wells,
            injection_rates=state.injection_rates,
            production_rates=state.production_rates,
            injection_formation_volume_factors=state.injection_formation_volume_factors,
            production_formation_volume_factors=state.production_formation_volume_factors,
            injection_bhps=state.injection_bhps,
            production_bhps=state.production_bhps,
            relative_permeabilities=relative_permeabilities,
            relative_mobilities=relative_mobilities,
            capillary_pressures=capillary_pressures,
            material_balance_errors=state.material_balance_errors,
            timer_state=state.timer_state,
        )
    return state
