"""Base API for Black-oil physics based fluid."""

import logging
import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.blackoil.pseudo_pressure import (
    PseudoPressureTable,
    build_pseudo_pressure_table,
)
from bores.blackoil.pvt.tables import PVTTable, PVTTables
from bores.correlations import arrays
from bores.errors import ValidationError
from bores.types import FluidPhase, Number, UnitSystem

logger = logging.getLogger(__name__)

__all__ = ["Fluid"]


def _validate_pseudo_pressure_phase(
    instance: "Fluid",
    attribute: typing.Any,
    value: PseudoPressureTable | None,
) -> None:
    """Validator: pseudo-pressure table is only valid for gas phase."""
    if value is not None and instance.phase != FluidPhase.GAS:
        phase_str = (
            instance.phase.value if isinstance(instance.phase, FluidPhase) else str(instance.phase)
        )
        raise ValidationError(
            f"`pseudo_pressure` can only be set on gas phase fluids. "
            f"Fluid '{instance.name}' has phase '{phase_str}'. "
            f"Remove `pseudo_pressure` or change phase to gas."
        )


@attrs.frozen(slots=True)
class Fluid:
    """
    Black-oil physics based fluid.

    Carries phase identity and optional PVT / pseudo-pressure tables.
    """

    name: str
    """Human-readable fluid name."""

    phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """Fluid phase (oil, gas, or gas)."""

    pvt: PVTTable | None = None
    """
    Optional `PVTTable` for this fluid.

    When set, callers can retrieve fluid properties directly:

    ```python
    fluid.pvt.viscosity(pressure, temperature)
    fluid.pvt.formation_volume_factor(pressure, temperature)
    fluid.pvt.compressibility_factor(pressure, temperature)  # gas only
    ```

    `None` if no table was provided; property lookups fall back to
    correlations in that case (requires `specific_gravity` on the subclass).
    """

    pseudo_pressure: PseudoPressureTable | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(_validate_pseudo_pressure_phase),
    )
    """
    Optional pre-built gas pseudo-pressure table. **Gas phase only.**

    If `None`, `get_pseudo_pressure_table` builds a table automatically
    from `pvt` interpolators (preferred) or from correlations
    (requires `specific_gravity` on the `WellFluid` subclass).
    """

    def _build_pseudo_pressure_cache_key(
        self,
        temperature: Number,
        reference_pressure: Number | None,
        pressure_range: tuple[Number, Number] | None,
        points: int | None,
        number_of_base_points: int = 500,
        number_of_endpoint_extra_points: int = 20,
        pvt_tables: PVTTables | None = None,
        dtype: npt.DTypeLike = None,
        unit_system: UnitSystem = UnitSystem.FIELD,
    ) -> tuple[typing.Any, ...]:
        """Stable hashable cache key for pseudo-pressure table construction."""
        pvt_hash: tuple | None = None
        if self.pvt is not None:
            bounds = getattr(self.pvt, "_extrapolation_bounds", {})
            pressure_bound = bounds.get("pressure", (0.0, 0.0))
            temperature_bound = bounds.get("temperature", (0.0, 0.0))
            pvt_hash = (
                (round(pressure_bound[0], 2), round(pressure_bound[1], 2)),
                (round(temperature_bound[0], 2), round(temperature_bound[1], 2)),
                getattr(self.pvt, "interpolation_method", None),
                self.pvt.has("compressibility_factor"),
                self.pvt.has("viscosity"),
            )

        # Hash from global pvt_tables bundle gas slot (if used as fallback)
        global_pvt_hash: tuple | None = None
        if pvt_tables is not None:
            gas_table = getattr(pvt_tables, "gas", None)
            if gas_table is not None:
                bounds = getattr(gas_table, "_extrapolation_bounds", {})
                pressure_bound = bounds.get("pressure", (0.0, 0.0))
                temperature_bound = bounds.get("temperature", (0.0, 0.0))
                global_pvt_hash = (
                    (round(pressure_bound[0], 2), round(pressure_bound[1], 2)),
                    (round(temperature_bound[0], 2), round(temperature_bound[1], 2)),
                    getattr(gas_table, "interpolation_method", None),
                    gas_table.has("compressibility_factor"),
                    gas_table.has("viscosity"),
                )

        # Include scalar `specific_gravity` from subclass if available
        specific_gravity = getattr(self, "specific_gravity", None)
        molecular_weight = getattr(self, "molecular_weight", None)
        return (
            self.name,
            self.phase.value,  # type: ignore
            round(float(specific_gravity), 6) if specific_gravity is not None else None,
            round(float(molecular_weight), 6) if molecular_weight is not None else None,
            round(temperature, 2),
            round(reference_pressure, 2) if reference_pressure is not None else None,
            tuple(round(p, 2) for p in pressure_range) if pressure_range is not None else None,
            points,
            number_of_base_points,
            number_of_endpoint_extra_points,
            pvt_hash,
            global_pvt_hash,
            dtype,
            unit_system,
        )

    def get_pseudo_pressure_table(
        self,
        temperature: Number,
        reference_pressure: Number | None = None,
        pressure_range: tuple[Number, Number] | None = None,
        points: int | None = None,
        number_of_base_points: int = 500,
        number_of_endpoint_extra_points: int = 20,
        pvt_tables: PVTTables | None = None,
        dtype: npt.DTypeLike = None,
        unit_system: UnitSystem = UnitSystem.FIELD,
        use_cache: bool = True,
    ) -> PseudoPressureTable:
        """
        Return the gas pseudo-pressure table for this fluid.

        **Priority order:**

        1. Return `self.pseudo_pressure` if explicitly set.
        2. Build from `self.pvt` interpolators (`compressibility_factor`
           and `viscosity`) when the fluid carries its own PVT table.
        3. Build from `pvt_tables.gas` interpolators when a global
           `PVTTables` bundle is passed and its gas slot has the required
           interpolators. Useful when the fluid itself has no `pvt`
           but the simulator holds a shared bundle.
        4. Fall back to DAK Z-factor / Lee-Kesler viscosity correlations using
           `self.specific_gravity` (available on `WellFluid` subclasses).

        Raises `ValidationError` if:
        - `self.phase` is not `gas`.
        - None of the four sources above can supply Z-factor + viscosity functions.

        :param temperature: Reservoir temperature (°F).
        :param reference_pressure: Reference pressure (psi); defaults to 14.7.
        :param pressure_range: `(p_min, p_max)` for table construction (psi).
        :param points: Number of integration points (default 200).
        :param pvt_tables: Optional global `PVTTables`
            bundle. Its `gas` slot is consulted when `self.pvt` is
            `None` or lacks the required interpolators.
        :param use_cache: Use global pseudo-pressure table cache.
        :return: `PseudoPressureTable`.
        """
        if self.phase != FluidPhase.GAS:
            raise ValidationError(
                "`get_pseudo_pressure_table(...)` is only applicable for gas phase fluids. "
                f"Fluid '{self.name}' has phase '{self.phase.value}'."  # type: ignore
            )

        if self.pseudo_pressure is not None:
            logger.debug("Using custom pseudo-pressure table for fluid '%s'", self.name)
            return self.pseudo_pressure

        z_factor_func: typing.Callable | None = None  # type: ignore
        viscosity_func: typing.Callable | None = None  # type: ignore

        if self.pvt is not None:
            if self.pvt.has("compressibility_factor"):
                pvt_table = self.pvt
                temperature_ = temperature

                def z_factor_func(pressure: npt.NDArray) -> npt.NDArray:  # type: ignore[misc]
                    temperature_arr = np.full_like(pressure, temperature_)
                    result = pvt_table.compressibility_factor(pressure, temperature_arr)
                    return np.asarray(result, dtype=pressure.dtype)

                z_factor_func._supports_arrays = True  # type: ignore[attr-defined]

            if self.pvt.has("viscosity"):
                pvt_table = self.pvt
                temperature_ = temperature

                def viscosity_func(pressure: npt.NDArray) -> npt.NDArray:  # type: ignore[misc]
                    temperature_arr = np.full_like(pressure, temperature_)
                    result = pvt_table.viscosity(pressure, temperature_arr)
                    return np.asarray(result, dtype=pressure.dtype)

                viscosity_func._supports_arrays = True  # type: ignore[attr-defined]

        # Only fills in whichever function is still missing after previous check.
        if (z_factor_func is None or viscosity_func is None) and pvt_tables is not None:
            gas_table = getattr(pvt_tables, "gas", None)
            if gas_table is not None:
                if z_factor_func is None and gas_table.has("compressibility_factor"):
                    gas_table_ = gas_table
                    temperature_ = temperature

                    def z_factor_func(pressure: npt.NDArray) -> npt.NDArray:  # type: ignore[misc]
                        temperature_arr = np.full_like(pressure, temperature_)
                        result = gas_table_.compressibility_factor(pressure, temperature_arr)
                        return np.asarray(result, dtype=pressure.dtype)

                    z_factor_func._supports_arrays = True  # type: ignore[attr-defined]

                if viscosity_func is None and gas_table.has("viscosity"):
                    gas_table_ = gas_table
                    temperature_ = temperature

                    def viscosity_func(
                        pressure: npt.NDArray,
                    ) -> npt.NDArray:  # type: ignore[misc]
                        temperature_arr = np.full_like(pressure, temperature_)
                        result = gas_table_.viscosity(pressure, temperature_arr)
                        return np.asarray(result, dtype=pressure.dtype)

                    viscosity_func._supports_arrays = True  # type: ignore[attr-defined]

        # Fall back to correlations
        specific_gravity = getattr(self, "specific_gravity", None)
        molecular_weight = getattr(self, "molecular_weight", None)

        if z_factor_func is None:
            if specific_gravity is None:
                raise ValidationError(
                    f"Cannot build pseudo-pressure table for fluid '{self.name}': "
                    "neither `pvt` (with `compressibility_factor`) nor `specific_gravity` "
                    "is available. Provide one of the two, or set `pseudo_pressure` directly."
                )

            specific_gravity_ = specific_gravity

            def z_factor_func(pressure: npt.NDArray) -> npt.NDArray:  # type: ignore[misc]
                temperature_arr = np.full_like(pressure, temperature)
                specific_gravity_arr = np.full_like(pressure, specific_gravity_)
                return arrays.compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature_arr,
                    gas_gravity=specific_gravity_arr,
                )

            z_factor_func._supports_arrays = True  # type: ignore[attr-defined]

        if viscosity_func is None:
            if specific_gravity is None or molecular_weight is None:
                raise ValidationError(
                    f"Cannot build pseudo-pressure table for fluid '{self.name}': "
                    "neither `pvt` (with `viscosity`) nor `specific_gravity` + `molecular_weight` "
                    "are available."
                )

            specific_gravity_ = specific_gravity
            molecular_weight_ = molecular_weight

            def viscosity_func(pressure: npt.NDArray) -> npt.NDArray:  # type: ignore[misc]
                temperature_arr = np.full_like(pressure, temperature)
                specific_gravity_arr = np.full_like(pressure, specific_gravity_)
                z_factor_arr = arrays.compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature_arr,
                    gas_gravity=specific_gravity_arr,
                )
                density_arr = arrays.compute_gas_density(
                    pressure=pressure,
                    temperature=temperature_arr,
                    gas_gravity=specific_gravity_arr,
                    gas_compressibility_factor=z_factor_arr,
                )
                return arrays.compute_gas_viscosity(
                    temperature=temperature_arr,
                    gas_density=density_arr,
                    gas_molecular_weight=np.full_like(pressure, molecular_weight_),
                )

            viscosity_func._supports_arrays = True  # type: ignore[attr-defined]

        cache_key = None
        if use_cache:
            cache_key = self._build_pseudo_pressure_cache_key(
                temperature=temperature,
                reference_pressure=reference_pressure,
                pressure_range=pressure_range,
                points=points,
                number_of_base_points=number_of_base_points,
                number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                pvt_tables=pvt_tables,
                dtype=dtype,
                unit_system=unit_system,
            )
        return build_pseudo_pressure_table(
            z_factor_func=z_factor_func,  # type: ignore[arg-type]
            viscosity_func=viscosity_func,  # type: ignore[arg-type]
            reference_pressure=reference_pressure,
            pressure_range=pressure_range,
            points=points,
            number_of_base_points=number_of_base_points,
            number_of_endpoint_extra_points=number_of_endpoint_extra_points,
            cache_key=cache_key,
            dtype=dtype,
            unit_system=unit_system,
        )
