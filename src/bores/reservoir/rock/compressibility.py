"""
Rock compressibility tables for reservoir simulation.

Supports both the simple `ROCK` keyword (single compressibility scalar per
rock region) and the tabulated `ROCKTAB` keyword (pore-volume and
transmissibility multiplier as explicit functions of pressure).
"""

import logging
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator, interp1d
from typing_extensions import Self

from bores.constants import (
    UnitConversionTable,
    build_unit_conversion_table,
    c,
    get_conversion_factors,
)
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.stores import StoreSerializable
from bores.types import (
    CellArray,
    FloatArray,
    IntCellArray,
    InterpolationMethod,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    TableQuery,
    TableResult,
    UnitSystem,
)
from bores.utils import scale

logger = logging.getLogger(__name__)

__all__ = [
    "RockCompressibility",
    "RockCompressibilityTable",
    "RockCompressibilityTables",
    "load_rock_compressibility_tables",
]


@attrs.frozen(slots=True)
class RockCompressibility:
    reference_pressure: CellArray
    """
    Shape (n_cells,) - reference pressure at which each cell's pore volume equals the
    geometrically calculated value.

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    """

    compressibility: CellArray
    """
    Shape (n_cells,) - formation compressibility.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Used in the pore-volume accumulation term: dPV/dP = PV · cr.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all quantities on this object are expressed.
    """

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `RockCompressibility` with all quantities rescaled
        to *target*.

        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `RockCompressibility` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            reference_pressure=scale(self.reference_pressure, factors["pressure"]),
            compressibility=scale(self.compressibility, factors["compressibility"]),
            unit_system=target,
        )


class RockCompressibilityTable(StoreSerializable):
    """
    Pore-volume and transmissibility multiplier table for a single rock region.

    Wraps a pressure-indexed table of:

    - `pore_volume_multiplier(P)` - PV(P) / PV(P_ref), dimensionless.
    - `transmissibility_multiplier(P)` - T(P) / T(P_ref), dimensionless.

    Both multipliers are built from either a `ROCKTAB` table (explicit
    tabulation) or a synthetic table derived from a `ROCK` scalar record:

    `PV_mult(P) = exp(cr · (P - P_ref))`

    The same interpolation API is provided regardless of which deck keyword
    was present.

    **Unit system**

    Pressure values are stored in the table's `unit_system`. Query methods
    expect pressure in the same system and return values in the same system.
    Use `convert(target)` to produce a copy in a different unit system.
    When building from a deck file the unit system is taken from
    `DeckFile.unit_system` automatically.

    **dtype**

    All stored arrays and all returned scalars / arrays use `dtype`.
    Defaults to `get_dtype()` when not specified.
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        pressures: FloatArray[OneDimension],
        pore_volume_multipliers: FloatArray[OneDimension],
        transmissibility_multipliers: FloatArray[OneDimension],
        reference_pressure: float,
        *,
        interpolation_method: InterpolationMethod = "linear",
        unit_system: UnitSystem = UnitSystem.FIELD,
        dtype: npt.DTypeLike = None,
        validate: bool = True,
    ) -> None:
        """
        Build the rock compressibility table from raw pressure and multiplier arrays.

        :param pressures: 1-D array of pressures, strictly increasing.
            Units must match `unit_system`.
        :param pore_volume_multipliers: 1-D array of PV multipliers at each
            pressure point (dimensionless, must be positive).
        :param transmissibility_multipliers: 1-D array of transmissibility
            multipliers at each pressure point (dimensionless, must be positive).
            Pass `np.ones_like(pressures)` when built from `ROCK` (no `ROCKTAB`
            transmissibility column).
        :param reference_pressure: Pressure at which both multipliers equal 1.0.
            Units must match `unit_system`.
        :param interpolation_method: `"linear"` or `"cubic"` (PCHIP). Cubic
            preserves monotonicity; linear is faster.
        :param unit_system: Unit system for all pressure inputs and outputs.
            Defaults to `FIELD` (psi).
        :param dtype: Array dtype. Defaults to `get_dtype()`.
        :param validate: Run physical-consistency checks on construction.
        """
        if interpolation_method not in ("linear", "cubic"):
            raise ValidationError(
                f"Invalid `interpolation_method` {interpolation_method!r}. "
                "Must be 'linear' or 'cubic'."
            )

        self.dtype = np.dtype(dtype if dtype is not None else get_dtype())
        self.pressures = typing.cast(NumberArray, pressures.astype(self.dtype, copy=False))
        self.pore_volume_multipliers = typing.cast(
            NumberArray, pore_volume_multipliers.astype(self.dtype, copy=False)
        )
        self.transmissibility_multipliers = typing.cast(
            NumberArray, np.asarray(transmissibility_multipliers, dtype=self.dtype)
        )
        self.reference_pressure = typing.cast(
            Number,
            self.dtype.type(reference_pressure),  # type: ignore
        )
        self.interpolation_method: InterpolationMethod = interpolation_method
        self.unit_system = unit_system

        if validate:
            self._validate()

        self._build_interpolants()
        logger.debug(
            f"{self.__class__.__name__} init: n_p=%d, method=%row, unit_system=%row, "
            "p_range=[%.1f, %.1f], p_ref=%.1f",
            len(self.pressures),
            interpolation_method,
            unit_system.value,
            float(self.pressures[0]),
            float(self.pressures[-1]),
            self.reference_pressure,
        )

    def _validate(self) -> None:
        """Run physical-consistency checks on the table arrays."""
        if len(self.pressures) < 2:
            raise ValidationError(
                "Rock compressibility table requires at least 2 pressure points; "
                f"got {len(self.pressures)}."
            )
        if not np.all(np.diff(self.pressures) > 0):
            raise ValidationError(
                "Rock compressibility table pressures must be strictly increasing."
            )
        if len(self.pore_volume_multipliers) != len(self.pressures):
            raise ValidationError(
                f"`pore_volume_multipliers` length {len(self.pore_volume_multipliers)} "
                f"must match `pressures` length {len(self.pressures)}."
            )
        if len(self.transmissibility_multipliers) != len(self.pressures):
            raise ValidationError(
                f"`transmissibility_multipliers` length "
                f"{len(self.transmissibility_multipliers)} must match `pressures` "
                f"length {len(self.pressures)}."
            )
        if np.any(self.pore_volume_multipliers <= 0):  # type: ignore
            raise ValidationError("All `pore_volume_multipliers` must be positive.")
        if np.any(self.transmissibility_multipliers <= 0):  # type: ignore
            raise ValidationError("All `transmissibility_multipliers` must be positive.")

        min_pressure = self.pressures[0]
        max_pressure = self.pressures[-1]
        if not (min_pressure <= self.reference_pressure <= max_pressure):
            warnings.warn(
                f"`reference_pressure` {self.reference_pressure:.2f} lies outside "
                f"the table pressure range [{min_pressure:.2f}, {max_pressure:.2f}]. "
                "Multipliers at reference pressure will be extrapolated.",
                UserWarning,
                stacklevel=4,
            )

    def _build_interpolants(self) -> None:
        """Build and cache scipy interpolatants for both multiplier columns."""
        pressures = self.pressures
        pore_volume_multiplier = self.pore_volume_multipliers
        transmissibility_multipliers = self.transmissibility_multipliers

        if self.interpolation_method == "cubic":
            pore_volume_interp = PchipInterpolator(
                pressures, pore_volume_multiplier, extrapolate=True
            )
            transmissibility_interp = PchipInterpolator(
                pressures, transmissibility_multipliers, extrapolate=True
            )
            pore_volume_dp_interp = pore_volume_interp.derivative(1)
            transmissibility_dp_interp = transmissibility_interp.derivative(1)
        else:
            pore_volume_interp = interp1d(
                pressures,
                pore_volume_multiplier,
                kind="linear",
                bounds_error=False,
                fill_value=(pore_volume_multiplier[0], pore_volume_multiplier[-1]),
            )
            transmissibility_interp = interp1d(
                pressures,
                transmissibility_multipliers,
                kind="linear",
                bounds_error=False,
                fill_value=(
                    transmissibility_multipliers[0],
                    transmissibility_multipliers[-1],
                ),
            )
            # Use PCHIP derivative even for the linear primal - the derivative
            # of a linear interpolant is a step function which is less useful
            # for the Jacobian than a smooth approximation from the same data.
            pore_volume_dp_interp = PchipInterpolator(
                pressures, pore_volume_multiplier
            ).derivative(1)
            transmissibility_dp_interp = PchipInterpolator(
                pressures, transmissibility_multipliers
            ).derivative(1)

        self._pore_volume_interp = pore_volume_interp
        self._transmissibility_interp = transmissibility_interp
        self._pore_volume_dp_interp = pore_volume_dp_interp
        self._transmissibility_dp_interp = transmissibility_dp_interp

    def _query(
        self,
        interp: typing.Any,
        pressure: TableQuery[NDimension],
        *,
        clip_min: float | None = None,
    ) -> TableResult[NDimension]:
        """
        Evaluate *interp* at *pressure* and return a typed scalar or array.

        :param interp: scipy interpolator with callable interface.
        :param pressure: Scalar or array of pressures in `self.unit_system`.
        :param clip_min: When given, clip result to `[clip_min, inf)`.
        :returns: Interpolated value(s) cast to `self.dtype`.
        """
        is_scalar = np.isscalar(pressure)
        dtype = self.dtype
        pressure_arr = np.atleast_1d(pressure)
        raw = interp(pressure_arr.ravel() if pressure_arr.ndim > 0 else pressure_arr)

        if clip_min is not None:
            raw = np.clip(raw, clip_min, None, dtype=dtype)

        if is_scalar or (isinstance(raw, np.ndarray) and raw.size == 1):
            return typing.cast(
                Number, dtype.type(raw.item() if isinstance(raw, np.ndarray) else raw)
            )
        return typing.cast(FloatArray[NDimension], raw.astype(dtype, copy=False))

    def pore_volume_multiplier(self, pressure: TableQuery[NDimension]) -> TableResult[NDimension]:
        """
        Return the pore-volume multiplier at *pressure*.

        `PV_mult(P) = PV(P) / PV(P_ref)`, dimensionless.

        Values outside the table range are extrapolated using the end-point
        slope of the interpolant (PCHIP) or held constant (linear). Result
        is clipped to `[0, ∞)` since a negative PV multiplier is unphysical.

        :param pressure: Scalar or array of pressures in `self.unit_system`.
        :returns: PV multiplier(s), dimensionless, in `self.dtype`.
        """
        return self._query(self._pore_volume_interp, pressure, clip_min=0.0)

    def pore_volume_multiplier_dp(
        self, pressure: TableQuery[NDimension]
    ) -> TableResult[NDimension]:
        """
        Return `∂(PV_mult)/∂P` at *pressure*.

        Used in the Jacobian accumulation term:
        `∂(PV)/∂P = PV_ref · ∂(PV_mult)/∂P`.

        Units: `[unit_system pressure]⁻¹` - e.g. psi⁻¹ in `FIELD`.

        :param pressure: Scalar or array of pressures in `self.unit_system`.
        :returns: Derivative of PV multiplier with respect to pressure,
            in `self.dtype`.
        """
        return self._query(self._pore_volume_dp_interp, pressure)

    def transmissibility_multiplier(
        self, pressure: TableQuery[NDimension]
    ) -> TableResult[NDimension]:
        """
        Return the transmissibility multiplier at *pressure*.

        `T_mult(P) = T(P) / T(P_ref)`, dimensionless.

        For tables built from `ROCK` (no `ROCKTAB` transmissibility column),
        this returns 1.0 everywhere. Result is clipped to `[0, ∞)`.

        :param pressure: Scalar or array of pressures in `self.unit_system`.
        :returns: Transmissibility multiplier(s), dimensionless, in `self.dtype`.
        """
        return self._query(self._transmissibility_interp, pressure, clip_min=0.0)

    def transmissibility_multiplier_dp(
        self, pressure: TableQuery[NDimension]
    ) -> TableResult[NDimension]:
        """
        Return `∂(T_mult)/∂P` at *pressure*.

        For tables built from `ROCK`, this is effectively 0.0 everywhere
        since `T_mult` is identically 1.0.

        Units: `[unit_system pressure]⁻¹`.

        :param pressure: Scalar or array of pressures in `self.unit_system`.
        :returns: Derivative of transmissibility multiplier with respect to
            pressure, in `self.dtype`.
        """
        return self._query(self._transmissibility_dp_interp, pressure)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `RockCompressibilityTable` with pressures rescaled to
        *target*.

        Multiplier values are dimensionless and are copied unchanged. Only
        the pressure axis (and therefore `reference_pressure`) are rescaled.

        :param target: Target `UnitSystem`.
        :returns: New `RockCompressibilityTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        return self.__class__(
            pressures=(self.pressures * pressure_factor).astype(self.dtype, copy=False),  # type: ignore[arg-type, operator]
            pore_volume_multipliers=self.pore_volume_multipliers.copy(),  # type: ignore[arg-type]
            transmissibility_multipliers=self.transmissibility_multipliers.copy(),  # type: ignore[arg-type]
            reference_pressure=self.reference_pressure * pressure_factor,  # type: ignore[arg-type, operator]
            interpolation_method=self.interpolation_method,
            unit_system=target,
            dtype=self.dtype,
            validate=False,  # already validated at construction
        )

    def get_compressibility(
        self,
        pressure: CellArray,
        unit_system: UnitSystem | None = None,
    ) -> RockCompressibility:
        """
        Derive per-cell `RockCompressibility` scalars from this table.

        Evaluates an effective compressibility at each cell's current
        pressure using:

        `cr_eff(P) = (1 / PV_mult(P)) · ∂(PV_mult)/∂P`

        This is the instantaneous compressibility implied by the table -
        equivalent to the `ROCK` scalar `compressibility` but pressure-dependent.

        :param pressure: Shape `(n_cells,)` - current pressures.
            Units must match `self.unit_system`.
        :param unit_system: Unit system for the returned `RockCompressibility`.
            Defaults to `self.unit_system`.
        :returns: `RockCompressibility` with per-cell effective compressibility.
        """
        dtype = self.dtype
        target_unit_system = unit_system if unit_system is not None else self.unit_system

        pore_volume_multiplier = self._pore_volume_interp(pressure)
        dpv_dp = self._pore_volume_dp_interp(pressure)

        effective_compressibility = np.where(
            pore_volume_multiplier > 0.0, dpv_dp / pore_volume_multiplier, 0.0
        ).astype(dtype)
        reference_pressure = np.full_like(pressure, self.reference_pressure, dtype=dtype)

        if target_unit_system != self.unit_system:
            factors = get_conversion_factors(self.unit_system, target_unit_system)
            reference_pressure *= factors["pressure"]  # type: ignore[operator]
            effective_compressibility *= factors["compressibility"]  # type: ignore[operator]

        return RockCompressibility(
            reference_pressure=typing.cast(
                CellArray, reference_pressure.astype(dtype, copy=False)
            ),
            compressibility=typing.cast(
                CellArray, effective_compressibility.astype(dtype, copy=False)
            ),
            unit_system=target_unit_system,
        )

    def __dump__(self) -> dict[str, typing.Any]:
        return {
            "pressures": self.pressures.tolist(),
            "pore_volume_multipliers": self.pore_volume_multipliers.tolist(),
            "transmissibility_multipliers": self.transmissibility_multipliers.tolist(),
            "reference_pressure": self.reference_pressure,
            "interpolation_method": self.interpolation_method,
            "unit_system": self.unit_system.value,
            "dtype": np.dtype(self.dtype).str,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            pressures=typing.cast(FloatArray[OneDimension], np.atleast_1d(data["pressures"])),
            pore_volume_multipliers=typing.cast(
                FloatArray[OneDimension], np.atleast_1d(data["pore_volume_multipliers"])
            ),
            transmissibility_multipliers=typing.cast(
                FloatArray[OneDimension],
                np.atleast_1d(data["transmissibility_multipliers"]),
            ),
            reference_pressure=float(data["reference_pressure"]),
            interpolation_method=data.get("interpolation_method", "linear"),
            unit_system=UnitSystem(data.get("unit_system", "field")),
            dtype=np.dtype(data["dtype"]) if "dtype" in data else None,
            validate=False,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_p={len(self.pressures)}, "
            f"p_range=[{float(self.pressures[0]):.1f}, {float(self.pressures[-1]):.1f}], "
            f"p_ref={self.reference_pressure:.1f}, "
            f"method={self.interpolation_method!r}, "
            f"unit_system={self.unit_system.value!r}"
            f")"
        )

    @classmethod
    def from_rock_record(
        cls,
        reference_pressure: float,
        rock_compressibility: float,
        *,
        n_pressure_points: int = 50,
        pressure_range_factor: float = 10.0,
        interpolation_method: InterpolationMethod = "linear",
        unit_system: UnitSystem = UnitSystem.FIELD,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `RockCompressibilityTable` from a `ROCK` scalar record.

        Synthesises a pressure table using the standard exponential model:

        `PV_mult(P) = exp(cr · (P - P_ref))`

        The pressure grid spans
        `[max(P_atm, P_ref / pressure_range_factor), P_ref x pressure_range_factor]`
        so the reference pressure sits well inside the table bounds.
        `P_atm` is taken as the atmospheric pressure in `unit_system` units
        (14.696 psi, 1.01325 bar, 1 atm, or 101 325 Pa).

        The transmissibility column is set to all-ones since `ROCK` does not
        specify transmissibility multipliers.

        :param reference_pressure: Reference pressure from `ROCK` column 1.
            Units must match `unit_system`.
        :param rock_compressibility: Formation compressibility cr (psi⁻¹ in
            `FIELD`, bar⁻¹ in `METRIC`, etc.) from `ROCK` column 2.
        :param n_pressure_points: Points in the synthetic pressure grid.
        :param pressure_range_factor: Multiplier defining the grid extent above
            and below `reference_pressure`.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param unit_system: Unit system for pressures.
        :param dtype: Array dtype.
        :returns: `RockCompressibilityTable` with all-ones transmissibility.
        :raises ValidationError: If inputs are physically invalid.
        """
        if reference_pressure <= 0:
            raise ValidationError(
                f"`reference_pressure` must be positive; got {reference_pressure}."
            )
        if rock_compressibility < 0:
            raise ValidationError(
                f"`rock_compressibility` must be non-negative; got {rock_compressibility}."
            )

        # Atmospheric pressure floor in the given unit system
        if unit_system == UnitSystem.FIELD:
            atm_pressure = c.STANDARD_PRESSURE_PSI
        elif unit_system == UnitSystem.METRIC:
            atm_pressure = c.STANDARD_PRESSURE_BAR
        elif unit_system == UnitSystem.LAB:
            atm_pressure = c.STANDARD_PRESSURE_ATM
        else:
            atm_pressure = c.STANDARD_PRESSURE_PASCAL

        min_pressure = max(atm_pressure, reference_pressure / pressure_range_factor)
        max_pressure = reference_pressure * pressure_range_factor

        dtype = np.dtype(dtype if dtype is not None else get_dtype())
        pressures = np.linspace(min_pressure, max_pressure, n_pressure_points, dtype=dtype)
        delta_p = pressures - reference_pressure
        pore_volume_multipliers = np.exp(rock_compressibility * delta_p).astype(dtype)
        transmissibility_multipliers = np.ones(n_pressure_points, dtype=dtype)

        return cls(
            pressures=typing.cast(FloatArray[OneDimension], pressures),
            pore_volume_multipliers=typing.cast(FloatArray[OneDimension], pore_volume_multipliers),
            transmissibility_multipliers=typing.cast(
                FloatArray[OneDimension], transmissibility_multipliers
            ),
            reference_pressure=float(reference_pressure),
            interpolation_method=interpolation_method,
            unit_system=unit_system,
            dtype=dtype,
        )

    @classmethod
    def from_rocktab_records(
        cls,
        rocktab_records: list[dict[str, float]],
        *,
        interpolation_method: InterpolationMethod = "linear",
        unit_system: UnitSystem = UnitSystem.FIELD,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `RockCompressibilityTable` from parsed `ROCKTAB` records.

        `ROCKTAB` provides explicit `(pressure, pore_volume_multiplier, transimissibility_multiplier)` rows.
        The reference pressure is the pressure where `pore_volume_multiplier` is closest to
        1.0 - per the Eclipse spec, exactly one row should have `pore_volume_multiplier = 1`.

        :param rocktab_records: List of row dicts with keys `"pressure"`,
            `"pore_volume_multiplier"`, and optionally `"transimissibility_multiplier"` (default 1.0 when absent).
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param unit_system: Unit system for the pressures in *rocktab_records*.
        :param dtype: Array dtype.
        :returns: `RockCompressibilityTable`.
        :raises ValidationError: If fewer than 2 rows or non-monotone pressures.
        """
        if len(rocktab_records) < 2:
            raise ValidationError(f"ROCKTAB requires at least 2 rows; got {len(rocktab_records)}.")

        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        rows = sorted(rocktab_records, key=lambda row: row["pressure"])

        pressures = np.array([row["pressure"] for row in rows], dtype=dtype)
        pore_volume_multipliers = np.array(
            [row["pore_volume_multiplier"] for row in rows], dtype=dtype
        )
        transmissibility_multipliers = np.array(
            [row.get("transimissibility_multiplier", 1.0) for row in rows], dtype=dtype
        )

        # Reference pressure: row where pore_volume_multiplier is nearest to 1.0
        reference_idx = int(np.argmin(np.abs(pore_volume_multipliers - 1.0)))
        reference_pressure = float(pressures[reference_idx])
        reference_pore_volume = float(pore_volume_multipliers[reference_idx])

        if abs(reference_pore_volume - 1.0) > 1e-3:
            warnings.warn(
                f"`ROCKTAB` `pore_volume_multiplier` does not reach 1.0 (closest value is "
                f"{reference_pore_volume:.4f} at {reference_pressure:.2f}). "
                "Reference pressure may be inaccurate.",
                UserWarning,
                stacklevel=2,
            )

        return cls(
            pressures=typing.cast(FloatArray[OneDimension], pressures),
            pore_volume_multipliers=typing.cast(FloatArray[OneDimension], pore_volume_multipliers),
            transmissibility_multipliers=typing.cast(
                FloatArray[OneDimension], transmissibility_multipliers
            ),
            reference_pressure=reference_pressure,
            interpolation_method=interpolation_method,
            unit_system=unit_system,
            dtype=dtype,
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        region_index: int = 0,
        *,
        n_pressure_points: int = 50,
        pressure_range_factor: float = 10,
        interpolation_method: InterpolationMethod = "linear",
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `RockCompressibilityTable` for one region from a `DeckFile`.

        `ROCKTAB` takes precedence over `ROCK`. The unit system is read from
        `deck_file.unit_system` automatically.

        :param deck_file: Parsed `DeckFile`.
        :param region_index: 0-based region index (default 0 -> region 1).
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param dtype: Array dtype.
        :returns: `RockCompressibilityTable` for the specified region.
        :raises ValidationError: If neither `ROCK` nor `ROCKTAB` is found, or
            the region index is out of range.
        """
        unit_system = deck_file.unit_system
        rocktab_all = deck_file.get("ROCKTAB")
        rock_all = deck_file.get("ROCK")

        if rocktab_all is not None and region_index < len(rocktab_all):
            records = rocktab_all[region_index]
            if records:
                return cls.from_rocktab_records(
                    rocktab_records=records,
                    interpolation_method=interpolation_method,
                    unit_system=unit_system,
                    dtype=dtype,
                )

        if rock_all is not None and region_index < len(rock_all):
            region_rows = rock_all[region_index]
            if region_rows:
                record = region_rows[0]
                return cls.from_rock_record(
                    reference_pressure=record["reference_pressure"],
                    rock_compressibility=record["compressibility"],
                    n_pressure_points=n_pressure_points,
                    pressure_range_factor=pressure_range_factor,
                    interpolation_method=interpolation_method,
                    unit_system=unit_system,
                    dtype=dtype,
                )

        raise ValidationError(
            f"No `ROCK` or `ROCKTAB` data found for region index {region_index} "
            "in the provided DeckFile."
        )


class RockCompressibilityTables(StoreSerializable):
    """
    Multi-region rock compressibility tables keyed by 1-based `ROCKNUM` index.

    All tables in a `RockCompressibilityTables` instance (should) share the same
    `unit_system` and `dtype` (enforced at construction by `from_deck`
    and `load_rock_compressibility_tables`).
    """

    __abstract_serializable__ = True
    __slots__ = ("tables", "unit_system")

    def __init__(
        self,
        tables: dict[int, RockCompressibilityTable],
        *,
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Build a `RockCompressibilityTables` from a pre-built regions dict.

        :param tables: Mapping from 1-based `ROCKNUM` index to table.
        :param unit_system: Expected unit system for all tables. If omitted,
            it is inferred from the first table and every other table is
            required to match it.
        :raises ValidationError: If *tables* is empty, or if any table's
            unit system does not match *unit_system* (explicit or inferred).
        """
        if not tables:
            raise ValidationError("`tables` must contain at least one entry.")

        expected_unit_system = unit_system or next(iter(tables.values())).unit_system
        mismatched = {
            rocknum: table.unit_system
            for rocknum, table in tables.items()
            if table.unit_system != expected_unit_system
        }
        if mismatched:
            raise ValidationError(
                f"All tables must share `{expected_unit_system.value!r}` as "
                f"`{self.__class__.__name__}.unit_system`; mismatches "
                f"(rocknum -> unit_system): "
                f"{ {k: v.value for k, v in mismatched.items()} }."
            )
        self.tables = tables
        self.unit_system = expected_unit_system

    @property
    def n_regions(self) -> int:
        """Number of rock compressibility regions."""
        return len(self.tables)

    def region(self, rocknum: int) -> RockCompressibilityTable:
        """
        Return the `RockCompressibilityTable` for a given 1-based region index.

        :param rocknum: 1-based `ROCKNUM` region index.
        :returns: `RockCompressibilityTable` for that region.
        :raises KeyError: If the region index does not exist.
        """
        table = self.tables.get(rocknum)
        if table is None:
            available = sorted(self.tables.keys())
            raise KeyError(
                f"Rock compressibility region {rocknum} not found. Available regions: {available}."
            )
        return table

    @classmethod
    def from_one(cls, table: RockCompressibilityTable) -> Self:
        """
        Wrap a single `RockCompressibilityTable` as region 1.

        :param table: `RockCompressibilityTable` instance.
        :returns: `RockCompressibilityTables` with one entry at key 1.
        """
        return cls(tables={1: table})

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        interpolation_method: InterpolationMethod = "linear",
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build all rock compressibility regions' tables from a `DeckFile`.

        `ROCKTAB` takes precedence over `ROCK` for any region that has a
        `ROCKTAB` entry; remaining regions fall back to `ROCK`. The unit
        system is read from `deck_file.unit_system`.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param dtype: Array dtype; defaults to `get_dtype()`.
        :returns: `RockCompressibilityTables` keyed by 1-based `ROCKNUM` index.
        :raises ValidationError: If neither `ROCK` nor `ROCKTAB` is found.
        """
        tables = load_rock_compressibility_tables(
            deck_file=deck_file,
            interpolation_method=interpolation_method,
            dtype=dtype,
        )
        return cls(tables=tables)

    def get_compressibility(
        self,
        pressure: CellArray,
        rock_region: IntCellArray | None = None,
        unit_system: UnitSystem | None = None,
        dtype: npt.DTypeLike = None,
    ) -> RockCompressibility:
        """
        Derive per-cell `RockCompressibility` from this `RockCompressibilityTables`.

        For each cell, evaluates the effective compressibility at the cell's
        current pressure from its region's table:

        `cr_eff(P) = (1 / PV_mult(P)) · ∂(PV_mult)/∂P`

        :param pressure: Shape `(n_cells,)` - current pressures.
            Units must match the tables' `unit_system`.
        :param rock_region: Shape `(n_cells,)` int array of 1-based `ROCKNUM`
            values. When `None`, region 1 is used for all cells.
        :param unit_system: Unit system for the returned `RockCompressibility`.
            Defaults to the unit system of region 1.
        :returns: Per-cell `RockCompressibility`.
        """
        n_cells = len(pressure)
        first_table = self.tables[next(iter(self.tables))]
        dtype = np.dtype(first_table.dtype) if dtype is None else dtype
        target_unit_system = unit_system if unit_system is not None else first_table.unit_system
        effective_compressibility = np.empty(n_cells, dtype=dtype)
        reference_pressure = np.empty(n_cells, dtype=dtype)

        if rock_region is None:
            table = self.region(1)
            pore_volume_multiplier = table._pore_volume_interp(pressure)
            dpv_dp = table._pore_volume_dp_interp(pressure)
            effective_compressibility[:] = np.where(
                pore_volume_multiplier > 0.0, dpv_dp / pore_volume_multiplier, 0.0
            )
            reference_pressure[:] = table.reference_pressure

            if target_unit_system != table.unit_system:
                factors = get_conversion_factors(table.unit_system, target_unit_system)
                reference_pressure *= factors["pressure"]  # type: ignore
                effective_compressibility *= factors["compressibility"]  # type: ignore
        else:
            unit_conversion_table = build_unit_conversion_table()
            for rocknum in np.unique(rock_region):
                mask = rock_region == rocknum
                table = self.region(int(rocknum))
                region_pressures = pressure[mask]
                pore_volume_multiplier = table._pore_volume_interp(region_pressures)
                dpv_dp = table._pore_volume_dp_interp(region_pressures)
                effective_compressibility = np.where(
                    pore_volume_multiplier > 0.0, dpv_dp / pore_volume_multiplier, 0.0
                )
                reference_pressure = np.full(mask.sum(), table.reference_pressure)

                if target_unit_system != table.unit_system:
                    factors = get_conversion_factors(
                        table.unit_system,
                        target_unit_system,
                        table=unit_conversion_table,
                    )
                    reference_pressure *= factors["pressure"]
                    effective_compressibility *= factors["compressibility"]

                effective_compressibility[mask] = effective_compressibility
                reference_pressure[mask] = reference_pressure

        return RockCompressibility(
            reference_pressure=typing.cast(
                CellArray, reference_pressure.astype(dtype, copy=False)
            ),
            compressibility=typing.cast(
                CellArray, effective_compressibility.astype(dtype, copy=False)
            ),
            unit_system=target_unit_system,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `RockCompressibilityTables` with all region tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `RockCompressibilityTables` in *target* units.
        """
        return self.__class__(
            tables={
                rocknum: tables.convert(target, table=table)
                for rocknum, tables in self.tables.items()
            }
        )

    def __getitem__(self, key: int) -> RockCompressibilityTable:
        return self.region(key)

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def __contains__(self, key: object) -> bool:
        return key in self.tables

    def __dump__(self) -> dict[str, typing.Any]:
        return {"tables": {str(rocknum): table.dump() for rocknum, table in self.tables.items()}}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        tables = {
            int(rocknum): RockCompressibilityTable.load(table_data)
            for rocknum, table_data in data["tables"].items()
        }
        return cls(tables=tables)

    def __repr__(self) -> str:
        region_keys = sorted(self.tables.keys())
        return f"{self.__class__.__name__}(n_regions={self.n_regions}, regions={region_keys})"


def load_rock_compressibility_tables(
    deck_file: DeckFile,
    *,
    interpolation_method: InterpolationMethod = "linear",
    dtype: npt.DTypeLike = None,
) -> dict[int, RockCompressibilityTable]:
    """
    Build a `{rocknum: RockCompressibilityTable}` dict from a `DeckFile`.

    `ROCKTAB` takes precedence over `ROCK`. For any region that has a
    `ROCKTAB` entry, `ROCKTAB` is used; remaining regions fall back to
    `ROCK`. The unit system is taken from `deck_file.unit_system` for all
    regions.

    :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
    :param interpolation_method: `"linear"` or `"cubic"`.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: Dict keyed by 1-based `ROCKNUM` index.
    :raises ValidationError: If neither `ROCK` nor `ROCKTAB` is found or
        no valid regions could be built.
    """
    unit_system = deck_file.unit_system
    rocktab_all: list | None = deck_file.get("ROCKTAB")
    rock_all: list | None = deck_file.get("ROCK")

    if rocktab_all is None and rock_all is None:
        raise ValidationError(
            "No rock compressibility keyword found in DeckFile. Expected `ROCK` or `ROCKTAB`."
        )

    n_regions = max(len(x) for x in [rocktab_all, rock_all] if x is not None)
    tables: dict[int, RockCompressibilityTable] = {}

    for region_idx in range(n_regions):
        rocknum = region_idx + 1  # 1-based

        # ROCKTAB takes precedence
        if rocktab_all is not None and region_idx < len(rocktab_all):
            records = rocktab_all[region_idx]
            if records:
                tables[rocknum] = RockCompressibilityTable.from_rocktab_records(
                    rocktab_records=records,
                    interpolation_method=interpolation_method,
                    unit_system=unit_system,
                    dtype=dtype,
                )
                logger.debug(
                    "Rock region %d: built from `ROCKTAB` (%d rows, %s)",
                    rocknum,
                    len(records),
                    unit_system.value,
                )
                continue

        # Fall back to ROCK
        if rock_all is not None and region_idx < len(rock_all):
            region_rows = rock_all[region_idx]
            if region_rows:
                record = region_rows[0]
                tables[rocknum] = RockCompressibilityTable.from_rock_record(
                    reference_pressure=float(record["reference_pressure"]),
                    rock_compressibility=float(record["compressibility"]),
                    interpolation_method=interpolation_method,
                    unit_system=unit_system,
                    dtype=dtype,
                )
                logger.debug(
                    "Rock region %d: built from `ROCK` (p_ref=%.2f, cr=%.3e, %s)",
                    rocknum,
                    float(record["reference_pressure"]),
                    float(record["compressibility"]),
                    unit_system.value,
                )
                continue

        warnings.warn(
            f"Rock region {rocknum} has no `ROCK` or `ROCKTAB` data - skipping.",
            UserWarning,
            stacklevel=2,
        )

    if not tables:
        raise ValidationError(
            "No valid rock compressibility regions could be built from the `DeckFile`."
        )
    return tables
