"""Base API for reservoir simulation fluids."""

import logging
import typing

import attrs
import numpy as np

from bores.correlations.arrays import (
    compute_gas_compressibility_factor as compute_gas_compressibility_factor_array,
)
from bores.correlations.arrays import (
    compute_gas_density as compute_gas_density_array,
)
from bores.correlations.arrays import (
    compute_gas_viscosity as compute_gas_viscosity_array,
)
from bores.errors import ValidationError
from bores.stores import StoreSerializable
from bores.tables.pseudo_pressure import (
    PseudoPressureTable,
    build_pseudo_pressure_table,
)
from bores.tables.pvt import PVTTable
from bores.types import FluidPhase

logger = logging.getLogger(__name__)

__all__ = ["Fluid"]


def _validate_pseudo_pressure_phase(
    instance: "Fluid",
    attribute: typing.Any,
    value: typing.Optional[PseudoPressureTable],
) -> None:
    """Validator: pseudo-pressure table is only valid for gas phase."""
    if value is not None and instance.phase != FluidPhase.GAS:
        phase_str = (
            instance.phase.value
            if isinstance(instance.phase, FluidPhase)
            else str(instance.phase)
        )
        raise ValidationError(
            f"`pseudo_pressure_table` can only be set on gas phase fluids. "
            f"Fluid '{instance.name}' has phase '{phase_str}'. "
            f"Remove `pseudo_pressure_table` or change phase to gas."
        )


@attrs.frozen
class Fluid(StoreSerializable):
    """
    Base class for all reservoir fluids.

    Carries phase identity and optional PVT / pseudo-pressure tables.
    """

    name: str
    """Human-readable fluid name."""

    phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """Fluid phase (oil, gas, or gas)."""

    pvt_table: typing.Optional[PVTTable] = None
    """
    Optional `PVTTable` for this fluid.

    When set, callers can retrieve fluid properties directly:

    ```python
    fluid.pvt_table.viscosity(pressure, temperature)
    fluid.pvt_table.formation_volume_factor(pressure, temperature)
    fluid.pvt_table.compressibility_factor(pressure, temperature)  # gas only
    ```

    `None` if no table was provided; property lookups fall back to
    correlations in that case (requires `specific_gravity` on the subclass).
    """

    pseudo_pressure_table: typing.Optional[PseudoPressureTable] = attrs.field(
        default=None,
        validator=attrs.validators.optional(_validate_pseudo_pressure_phase),
    )
    """
    Optional pre-built gas pseudo-pressure table.  **Gas phase only.**

    If `None`, `get_pseudo_pressure_table` builds a table automatically
    from `pvt_table` interpolators (preferred) or from correlations
    (requires `specific_gravity` on the `WellFluid` subclass).
    """

    def _build_pseudo_pressure_cache_key(
        self,
        temperature: float,
        reference_pressure: typing.Optional[float],
        pressure_range: typing.Optional[typing.Tuple[float, float]],
        points: typing.Optional[int],
        pvt_tables: typing.Optional[typing.Any] = None,
    ) -> typing.Tuple[typing.Any, ...]:
        """Stable hashable cache key for pseudo-pressure table construction."""
        pvt_hash: typing.Optional[tuple] = None
        if self.pvt_table is not None:
            bounds = getattr(self.pvt_table, "_extrapolation_bounds", {})
            p_b = bounds.get("pressure", (0.0, 0.0))
            t_b = bounds.get("temperature", (0.0, 0.0))
            pvt_hash = (
                (round(p_b[0], 2), round(p_b[1], 2)),
                (round(t_b[0], 2), round(t_b[1], 2)),
                getattr(self.pvt_table, "interpolation_method", None),
                self.pvt_table.exists("compressibility_factor"),
                self.pvt_table.exists("viscosity"),
            )

        # Hash from global pvt_tables bundle gas slot (if used as fallback)
        global_pvt_hash: typing.Optional[tuple] = None
        if pvt_tables is not None:
            gas_table = getattr(pvt_tables, "gas", None)
            if gas_table is not None:
                bounds = getattr(gas_table, "_extrapolation_bounds", {})
                p_b = bounds.get("pressure", (0.0, 0.0))
                t_b = bounds.get("temperature", (0.0, 0.0))
                global_pvt_hash = (
                    (round(p_b[0], 2), round(p_b[1], 2)),
                    (round(t_b[0], 2), round(t_b[1], 2)),
                    getattr(gas_table, "interpolation_method", None),
                    gas_table.exists("compressibility_factor"),
                    gas_table.exists("viscosity"),
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
            tuple(round(p, 2) for p in pressure_range)
            if pressure_range is not None
            else None,
            points,
            pvt_hash,
            global_pvt_hash,
        )

    def get_pseudo_pressure_table(
        self,
        temperature: float,
        reference_pressure: typing.Optional[float] = None,
        pressure_range: typing.Optional[typing.Tuple[float, float]] = None,
        points: typing.Optional[int] = None,
        pvt_tables: typing.Optional[typing.Any] = None,
        use_cache: bool = True,
    ) -> PseudoPressureTable:
        """
        Return the gas pseudo-pressure table for this fluid.

        **Priority order:**

        1. Return `self.pseudo_pressure_table` if explicitly set.
        2. Build from `self.pvt_table` interpolators (`compressibility_factor`
           and `viscosity`) when the fluid carries its own PVT table.
        3. Build from `pvt_tables.gas` interpolators when a global
           `PVTTables` bundle is passed and its gas slot has the required
           interpolators.  Useful when the fluid itself has no `pvt_table`
           but the simulator holds a shared bundle.
        4. Fall back to DAK Z-factor / Lee-Kesler viscosity correlations using
           `self.specific_gravity` (available on `WellFluid` subclasses).

        Raises `ValidationError` if:
        - `self.phase` is not ``gas``.
        - None of the four sources above can supply Z-factor + viscosity functions.

        :param temperature: Reservoir temperature (°F).
        :param reference_pressure: Reference pressure (psi); defaults to 14.7.
        :param pressure_range: ``(p_min, p_max)`` for table construction (psi).
        :param points: Number of integration points (default 200).
        :param pvt_tables: Optional global :class:`~bores.tables.pvt.PVTTables`
            bundle.  Its ``gas`` slot is consulted when ``self.pvt_table`` is
            ``None`` or lacks the required interpolators.
        :param use_cache: Use global pseudo-pressure table cache.
        :return: :class:`~bores.tables.pseudo_pressure.PseudoPressureTable`.
        """
        if self.phase != FluidPhase.GAS:
            raise ValidationError(
                "`get_pseudo_pressure_table(...)` is only applicable for gas phase fluids. "
                f"Fluid '{self.name}' has phase '{self.phase.value}'."  # type: ignore
            )

        if self.pseudo_pressure_table is not None:
            logger.debug(f"Using custom pseudo-pressure table for fluid '{self.name}'.")
            return self.pseudo_pressure_table

        z_factor_func: typing.Optional[typing.Callable] = None  # type: ignore
        viscosity_func: typing.Optional[typing.Callable] = None  # type: ignore

        if self.pvt_table is not None:
            if self.pvt_table.exists("compressibility_factor"):
                _pvt_table = self.pvt_table
                _temperature = temperature

                def z_factor_func(pressure: np.typing.NDArray) -> np.typing.NDArray:  # type: ignore[misc]
                    temperature_arr = np.full_like(pressure, _temperature)
                    result = _pvt_table.compressibility_factor(
                        pressure, temperature_arr
                    )
                    return np.asarray(result, dtype=pressure.dtype)

                z_factor_func._supports_arrays = True  # type: ignore[attr-defined]

            if self.pvt_table.exists("viscosity"):
                _pvt_table = self.pvt_table
                _temperature = temperature

                def viscosity_func(pressure: np.typing.NDArray) -> np.typing.NDArray:  # type: ignore[misc]
                    temperature_arr = np.full_like(pressure, _temperature)
                    result = _pvt_table.viscosity(pressure, temperature_arr)
                    return np.asarray(result, dtype=pressure.dtype)

                viscosity_func._supports_arrays = True  # type: ignore[attr-defined]

        # Only fills in whichever function is still missing after previous check.
        if (z_factor_func is None or viscosity_func is None) and pvt_tables is not None:
            gas_table = getattr(pvt_tables, "gas", None)
            if gas_table is not None:
                if z_factor_func is None and gas_table.exists("compressibility_factor"):
                    _gas_table = gas_table
                    _temperature = temperature

                    def z_factor_func(pressure: np.typing.NDArray) -> np.typing.NDArray:  # type: ignore[misc]
                        temperature_arr = np.full_like(pressure, _temperature)
                        result = _gas_table.compressibility_factor(
                            pressure, temperature_arr
                        )
                        return np.asarray(result, dtype=pressure.dtype)

                    z_factor_func._supports_arrays = True  # type: ignore[attr-defined]

                if viscosity_func is None and gas_table.exists("viscosity"):
                    _gas_table = gas_table
                    _temperature = temperature

                    def viscosity_func(
                        pressure: np.typing.NDArray,
                    ) -> np.typing.NDArray:  # type: ignore[misc]
                        temperature_arr = np.full_like(pressure, _temperature)
                        result = _gas_table.viscosity(pressure, temperature_arr)
                        return np.asarray(result, dtype=pressure.dtype)

                    viscosity_func._supports_arrays = True  # type: ignore[attr-defined]

        # Fall back to correlations
        specific_gravity = getattr(self, "specific_gravity", None)
        molecular_weight = getattr(self, "molecular_weight", None)

        if z_factor_func is None:
            if specific_gravity is None:
                raise ValidationError(
                    f"Cannot build pseudo-pressure table for fluid '{self.name}': "
                    "neither `pvt_table` (with `compressibility_factor`) nor `specific_gravity` "
                    "is available. Provide one of the two, or set `pseudo_pressure_table` directly."
                )

            _specific_gravity = float(specific_gravity)

            def z_factor_func(pressure: np.typing.NDArray) -> np.typing.NDArray:  # type: ignore[misc]
                temperature_arr = np.full_like(pressure, temperature)
                specific_gravity_arr = np.full_like(pressure, _specific_gravity)
                return compute_gas_compressibility_factor_array(
                    pressure=pressure,
                    temperature=temperature_arr,
                    gas_gravity=specific_gravity_arr,
                )

            z_factor_func._supports_arrays = True  # type: ignore[attr-defined]

        if viscosity_func is None:
            if specific_gravity is None or molecular_weight is None:
                raise ValidationError(
                    f"Cannot build pseudo-pressure table for fluid '{self.name}': "
                    "neither `pvt_table` (with `viscosity`) nor `specific_gravity` + `molecular_weight` "
                    "are available."
                )

            _specific_gravity = float(specific_gravity)
            _molecular_weight = float(molecular_weight)

            def viscosity_func(pressure: np.typing.NDArray) -> np.typing.NDArray:  # type: ignore[misc]
                temperature_arr = np.full_like(pressure, temperature)
                specific_gravity_arr = np.full_like(pressure, _specific_gravity)
                z_factor_arr = compute_gas_compressibility_factor_array(
                    pressure=pressure,
                    temperature=temperature_arr,
                    gas_gravity=specific_gravity_arr,
                )
                density_arr = compute_gas_density_array(
                    pressure=pressure,
                    temperature=temperature_arr,
                    gas_gravity=specific_gravity_arr,
                    gas_compressibility_factor=z_factor_arr,
                )
                return compute_gas_viscosity_array(
                    temperature=temperature_arr,
                    gas_density=density_arr,
                    gas_molecular_weight=np.full_like(pressure, _molecular_weight),
                )

            viscosity_func._supports_arrays = True  # type: ignore[attr-defined]

        cache_key = None
        if use_cache:
            cache_key = self._build_pseudo_pressure_cache_key(
                temperature=temperature,
                reference_pressure=reference_pressure,
                pressure_range=pressure_range,
                points=points,
                pvt_tables=pvt_tables,
            )

        return build_pseudo_pressure_table(
            z_factor_func=z_factor_func,  # type: ignore[arg-type]
            viscosity_func=viscosity_func,  # type: ignore[arg-type]
            reference_pressure=reference_pressure,
            pressure_range=pressure_range,
            points=points,
            cache_key=cache_key,
        )

