"""PVT (Pressure-Volume-Temperature) property tables for reservoir simulation."""

import logging
import typing
import warnings
from os import PathLike

import attrs
import numpy as np
from scipy.interpolate import (  # type: ignore[import-untyped]
    RectBivariateSpline,
    RegularGridInterpolator,
    interp1d,
)
from typing_extensions import Self

from bores._precision import get_dtype
from bores.constants import c
from bores.correlations.core import compute_gas_gravity, compute_oil_api_gravity
from bores.errors import ValidationError
from bores.grids.pvt import (
    build_estimated_solution_gas_to_oil_ratio_grid,
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_molecular_weight_grid,
    build_gas_solubility_in_water_grid,
    build_gas_viscosity_grid,
    build_live_oil_density_grid,
    build_oil_api_gravity_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_formation_volume_factor_grid,
    build_oil_viscosity_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.stores import StoreSerializable
from bores.types import (
    FluidPhase,
    NDimensionalGrid,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    TwoDimensionalGrid,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GAS_CLAMPS",
    "DEFAULT_OIL_CLAMPS",
    "DEFAULT_WATER_CLAMPS",
    "PVTData",
    "PVTDataSet",
    "PVTTable",
    "PVTTables",
    "build_gas_pvt_data",
    "build_oil_pvt_data",
    "build_pvt_dataset",
    "build_water_pvt_data",
]

InterpolationMethod = typing.Literal["linear", "cubic"]
_INTERPOLATION_DEGREES = {"linear": 1, "cubic": 3}

QueryType = typing.Union[NDimensionalGrid, typing.List[typing.Any], float, np.floating]


DEFAULT_OIL_CLAMPS: typing.Dict[str, typing.Tuple[float, float]] = {
    "viscosity": (1e-6, 1e4),
    "density": (1.0, 80.0),
    "formation_volume_factor": (0.5, 5.0),
    "compressibility": (0.0, 1e-1),
    "solution_gor": (0.0, 5000.0),
    "specific_gravity": (0.5, 1.2),
    "molecular_weight": (10.0, 600.0),
}

DEFAULT_GAS_CLAMPS: typing.Dict[str, typing.Tuple[float, float]] = {
    "viscosity": (1e-6, 1e2),
    "density": (0.001, 50.0),
    "formation_volume_factor": (1e-6, 100.0),
    "compressibility": (0.0, 1e-1),
    "compressibility_factor": (0.1, 3.0),
    "solubility_in_water": (0.0, 500.0),
    "specific_gravity": (0.5, 2.0),
    "molecular_weight": (2.0, 100.0),
}

DEFAULT_WATER_CLAMPS: typing.Dict[str, typing.Tuple[float, float]] = {
    "viscosity": (1e-6, 10.0),
    "density": (30.0, 80.0),
    "formation_volume_factor": (0.9, 2.0),
    "compressibility": (0.0, 1e-2),
    "specific_gravity": (0.9, 1.3),
    "molecular_weight": (18.0, 20.0),
}

PHASE_DEFAULT_CLAMPS: typing.Dict[
    FluidPhase, typing.Dict[str, typing.Tuple[float, float]]
] = {
    FluidPhase.OIL: DEFAULT_OIL_CLAMPS,
    FluidPhase.GAS: DEFAULT_GAS_CLAMPS,
    FluidPhase.WATER: DEFAULT_WATER_CLAMPS,
}


@attrs.frozen
class PVTData(StoreSerializable):
    """
    Raw PVT table data for a single fluid phase.

    Phase-tagged container for tabulated fluid properties. All table arrays are
    optional - only `phase`, `pressures`, and `temperatures` are required.
    `PVTTable` validates which fields are meaningful for the phase at initialization time.

    **Table shapes:**
    - Oil / gas: 2D arrays with shape `(n_pressures, n_temperatures)`
    - Water: 3D arrays with shape `(n_pressures, n_temperatures, n_salinities)`

    **Oil-specific fields:** `bubble_point_pressures`, `solution_gas_to_oil_ratios`,
    `solution_gor_table`

    **Gas-specific fields:** `compressibility_factor_table`, `solubility_in_water_table`

    **Water-specific fields:** `salinities`, `bubble_point_pressure_table`,
    `gas_free_fvf_table`
    """

    phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """Fluid phase this data describes."""

    # Base coordinate grids
    pressures: OneDimensionalGrid
    """1D array of pressures (psi), strictly increasing."""

    temperatures: OneDimensionalGrid
    """1D array of temperatures (°F), strictly increasing."""

    # Water-only coordinate grid
    salinities: typing.Optional[OneDimensionalGrid] = None
    """1D array of salinities (ppm NaCl), strictly increasing. Water phase only."""

    # Oil-only coordinate / meta grids
    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None
    """
    Bubble point pressures (psi). Oil phase only.
    - 1D shape `(n_t,)`    → Pb(T)
    - 2D shape `(n_rs, n_t)` → Pb(Rs, T); requires `solution_gas_to_oil_ratios`
    """

    solution_gas_to_oil_ratios: typing.Optional[OneDimensionalGrid] = None
    """
    1D array of Rs values (SCF/STB) used as the first axis of a 2D
    `bubble_point_pressures` table. Required when `bubble_point_pressures`
    is 2D. Oil phase only.
    """

    # Shared property tables (2D for oil/gas, 3D for water)
    viscosity_table: typing.Optional[
        typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid]
    ] = None
    """Viscosity μ(P,T) in cP. 2D for oil/gas, 3D for water."""

    density_table: typing.Optional[
        typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid]
    ] = None
    """Density ρ(P,T) in lbm/ft³. 2D for oil/gas, 3D for water."""

    formation_volume_factor_table: typing.Optional[
        typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid]
    ] = None
    """
    Formation volume factor B(P,T) in bbl/STB (oil/water) or ft³/SCF (gas).
    2D for oil/gas, 3D for water.
    """

    compressibility_table: typing.Optional[
        typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid]
    ] = None
    """Compressibility c(P,T) in psi⁻¹. 2D for oil/gas, 3D for water."""

    specific_gravity_table: typing.Optional[
        typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid]
    ] = None
    """Specific gravity γ(P,T) (dimensionless). 2D for oil/gas, 3D for water."""

    molecular_weight_table: typing.Optional[
        typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid]
    ] = None
    """Molecular weight M(P,T) in lbm/lb-mol. 2D for oil/gas, 3D for water."""

    # Oil-only property tables (2D)
    solution_gor_table: typing.Optional[TwoDimensionalGrid] = None
    """Solution GOR Rs(P,T) in SCF/STB. Oil phase only."""

    # Gas-only property tables
    compressibility_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Z-factor z(P,T), dimensionless. Gas phase only."""

    solubility_in_water_table: typing.Optional[ThreeDimensionalGrid] = None
    """
    Gas solubility in water Rsw(P,T,S) in SCF/STB.
    Gas phase only. 3D shape `(n_p, n_t, n_s)` - requires `salinities`.
    """

    # Water-only property tables (3D)
    bubble_point_pressure_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water bubble point pressure water bubble point pressure(P,T,S) in psi. Water phase only."""

    gas_free_fvf_table: typing.Optional[TwoDimensionalGrid] = None
    """
    Gas-free water FVF gas-free water formation volume factor(P,T) in bbl/STB. Water phase only.
    Used internally for water formation volume factor and water compressibility calculations; not exposed on PVTTable.
    """

    def __attrs_post_init__(self) -> None:
        self._ensure_dtype()
        self._warn_phase_mismatches()

    def _ensure_dtype(self) -> None:
        """Convert all array fields to the global simulation dtype."""
        dtype = get_dtype()
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)
            if (
                value is not None
                and isinstance(value, np.ndarray)
                and value.dtype != dtype
            ):
                object.__setattr__(self, field.name, value.astype(dtype, copy=False))

    def _warn_phase_mismatches(self) -> None:
        """Emit warnings for fields that will be silently ignored for this phase."""
        phase = typing.cast(FluidPhase, self.phase)
        if phase == FluidPhase.GAS and self.solution_gor_table is not None:
            warnings.warn(
                f"{self.__class__.__name__}: `solution_gor_table` is oil-only and will be ignored "
                "for GAS phase.",
                UserWarning,
                stacklevel=3,
            )
        if phase == FluidPhase.OIL and self.compressibility_factor_table is not None:
            warnings.warn(
                f"{self.__class__.__name__}: `compressibility_factor_table` is gas-only and will be "
                "ignored for OIL phase.",
                UserWarning,
                stacklevel=3,
            )
        if phase == FluidPhase.WATER and self.bubble_point_pressures is not None:
            warnings.warn(
                f"{self.__class__.__name__}: `bubble_point_pressures` is oil-only. For water bubble "
                "point use `bubble_point_pressure_table` (3D). Field will be ignored.",
                UserWarning,
                stacklevel=3,
            )
        if self.salinities is not None and phase == FluidPhase.OIL:
            warnings.warn(
                f"{self.__class__.__name__}: `salinities` is water-only and will be ignored for "
                f"{phase.value.upper()} phase.",
                UserWarning,
                stacklevel=3,
            )
        if (
            self.bubble_point_pressures is not None
            and isinstance(self.bubble_point_pressures, np.ndarray)
            and self.bubble_point_pressures.ndim == 2
            and self.solution_gas_to_oil_ratios is None
        ):
            raise ValidationError(
                f"{self.__class__.__name__}: 2D `bubble_point_pressures` requires "
                "`solution_gas_to_oil_ratios` to be provided."
            )


class PVTTable(StoreSerializable):
    """
    Phase-aware PVT property lookup using pre-built interpolators.

    Accepts a `PVTData` instance and builds fast scipy interpolators
    for all non-None tables. The public interface exposes generic method names
    that apply to all phases; methods that are not applicable to a phase (or
    whose underlying table was not supplied) return `None`.

    **Phase dispatch summary:**

    +---------------------------------+-----+-----+-------+
    | Method                          | Oil | Gas | Water |
    +=================================+=====+=====+=======+
    | viscosity                       | ✓   | ✓   | ✓     |
    | density                         | ✓   | ✓   | ✓     |
    | formation_volume_factor         | ✓   | ✓   | ✓     |
    | compressibility                 | ✓   | ✓   | ✓     |
    | specific_gravity                | ✓   | ✓   | ✓     |
    | molecular_weight                | ✓   | ✓   | ✓     |
    | bubble_point_pressure           | ✓   | ✗   | ✓     |
    | is_saturated                    | ✓   | ✗   | ✗     |
    | solution_gor                    | ✓   | ✗   | ✗     |
    | compressibility_factor          | ✗   | ✓   | ✗     |
    | solubility_in_water             | ✗   | ✓   | ✗     |
    +---------------------------------+-----+-----+-------+

    **Extrapolation:** Outside table bounds the interpolators extrapolate
    linearly/cubically. Set `warn_on_extrapolation=True` to log warnings.

    **Performance:**  Uses `RectBivariateSpline` for 2D (oil/gas) - 10-50x
    faster than `RegularGridInterpolator`, and uses `RegularGridInterpolator`
    for 3D (water).
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        data: typing.Union[PVTData, PathLike[str], str],
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        clamps: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None,
    ) -> None:
        """
        Build a pvt table from raw tabulated data.

        :param data: Phase-tagged raw PVT data or path to PVT data file.
        :param interpolation_method: `"linear"` (default) or `"cubic"`.
            Cubic requires ≥ 4 points along each axis.
        :param validate: Run physical-consistency checks on `data`.
        :param warn_on_extrapolation: Log a warning when queries fall outside
            the table pressure/temperature/salinity bounds.
        :param clamps: Override default per-property clamp ranges.
            Keys use generic names (e.g. `"viscosity"`, not `"oil_viscosity"`).
            Merged on top of the phase-appropriate defaults.
        """
        if interpolation_method not in _INTERPOLATION_DEGREES:
            raise ValidationError(
                f"Invalid interpolation_method {interpolation_method!r}. "
                f"Must be one of: {list(_INTERPOLATION_DEGREES.keys())}"
            )

        if not isinstance(data, PVTData):
            data = PVTData.from_file(data)  # type: ignore[assignment]

        data = typing.cast(PVTData, data)
        self._phase = data.phase
        self.interpolation_method: InterpolationMethod = interpolation_method
        self.validate = validate
        self.warn_on_extrapolation = warn_on_extrapolation

        # Merge clamps
        self.clamps: typing.Dict[str, typing.Tuple[float, float]] = {
            **PHASE_DEFAULT_CLAMPS[data.phase],  # type: ignore
            **(clamps or {}),
        }

        # Cubic needs ≥ 4 points
        if interpolation_method != "linear":
            if len(data.pressures) < 4:
                raise ValidationError(
                    f"At least 4 pressure points required for {interpolation_method!r} "
                    f"interpolation, got {len(data.pressures)}."
                )
            if len(data.temperatures) < 4:
                raise ValidationError(
                    f"At least 4 temperature points required for {interpolation_method!r} "
                    f"interpolation, got {len(data.temperatures)}."
                )
            if data.salinities is not None and len(data.salinities) < 4:
                raise ValidationError(
                    f"At least 4 salinity points required for {interpolation_method!r} "
                    f"interpolation, got {len(data.salinities)}."
                )

        self._validate_data(data)
        if validate:
            self._check_physical_consistency(data)

        # Store extrapolation bounds
        self._extrapolation_bounds: typing.Dict[str, typing.Tuple[float, float]] = {
            "pressure": (float(data.pressures[0]), float(data.pressures[-1])),
            "temperature": (float(data.temperatures[0]), float(data.temperatures[-1])),
        }
        if data.salinities is not None:
            self._extrapolation_bounds["salinity"] = (
                float(data.salinities[0]),
                float(data.salinities[-1]),
            )

        # Oil bubble-point dimensionality
        self._pb_ndim: typing.Optional[int] = None
        if data.phase == FluidPhase.OIL and data.bubble_point_pressures is not None:
            self._pb_ndim = data.bubble_point_pressures.ndim

        # Water default salinity (used when salinity arg is None)
        self.default_salinity: typing.Optional[float] = (
            float(data.salinities[0]) if data.salinities is not None else None
        )

        self._data = data
        self._interpolators: typing.Dict[str, typing.Any] = {}
        self._build_interpolators(data)

        logger.debug(
            f"PVTTable initialized: phase={self.phase.value.upper()}, "
            f"P ∈ [{data.pressures[0]:.4f}, {data.pressures[-1]:.4f}] psi, "
            f"T ∈ [{data.temperatures[0]:.4f}, {data.temperatures[-1]:.4f}] °F, "
            f"interpolation_method={interpolation_method!r}, "
            f"interpolators={len(self._interpolators)}"
        )

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """
        Serialize this `PVTTable` to a dictionary.

        :param recurse: Passed through to nested `StoreSerializable` objects.
        :return: JSON-compatible dictionary representation.
        """
        return {
            "data": self._data.dump(recurse),
            "interpolation_method": self.interpolation_method,
            "validate": self.validate,
            "warn_on_extrapolation": self.warn_on_extrapolation,
            # Only store clamps that differ from the phase defaults so the
            # payload stays compact and phase-default changes propagate forward.
            "clamps": {
                k: list(v)
                for k, v in self.clamps.items()
                if v != PHASE_DEFAULT_CLAMPS[self.phase].get(k)
            },
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "PVTTable":
        """
        Deserialize a `PVTTable` from a dictionary produced by `__dump__`.

        Rebuilds all scipy interpolators from the stored `PVTData`.

        :param data: Dictionary as returned by `__dump__`.
        :return: Fully initialised `PVTTable` instance.
        """
        pvt_data = PVTData.load(data["data"])
        clamps = {k: tuple(v) for k, v in data.get("clamps", {}).items()} or None
        return cls(
            data=pvt_data,
            interpolation_method=data.get("interpolation_method", "linear"),
            validate=data.get("validate", True),
            warn_on_extrapolation=data.get("warn_on_extrapolation", False),
            clamps=clamps or None,
        )

    def _validate_data(self, data: PVTData) -> None:
        """Validate grid monotonicity, shapes, and phase-specific field rules."""
        pressures = data.pressures
        temperatures = data.temperatures
        salinities = data.salinities
        n_p = len(pressures)
        n_t = len(temperatures)
        n_s = len(salinities) if salinities is not None else None

        if pressures.ndim != 1:
            raise ValidationError("`pressures` must be 1-dimensional.")
        if temperatures.ndim != 1:
            raise ValidationError("`temperatures` must be 1-dimensional.")
        if not np.all(np.diff(pressures) > 0):
            raise ValidationError(
                "`pressures` must be strictly monotonically increasing."
            )
        if not np.all(np.diff(temperatures) > 0):
            raise ValidationError(
                "`temperatures` must be strictly monotonically increasing."
            )
        if salinities is not None:
            if salinities.ndim != 1:
                raise ValidationError("`salinities` must be 1-dimensional.")
            if not np.all(np.diff(salinities) > 0):
                raise ValidationError(
                    "`salinities` must be strictly monotonically increasing."
                )

        # Bubble point shape (oil only)
        bubble_point_pressures = data.bubble_point_pressures
        if bubble_point_pressures is not None:
            if bubble_point_pressures.ndim == 1:
                if len(bubble_point_pressures) != n_t:
                    raise ValidationError(
                        f"`bubble_point_pressures` 1D length {len(bubble_point_pressures)} must match "
                        f"n_temperatures={n_t}."
                    )
            elif bubble_point_pressures.ndim == 2:
                solution_gors = data.solution_gas_to_oil_ratios
                if solution_gors is None:
                    raise ValidationError(
                        "2D `bubble_point_pressures` requires `solution_gas_to_oil_ratios`."
                    )
                if not np.all(np.diff(solution_gors) > 0):
                    raise ValidationError(
                        "`solution_gas_to_oil_ratios` must be strictly monotonically increasing."
                    )
                if bubble_point_pressures.shape != (len(solution_gors), n_t):
                    raise ValidationError(
                        f"`bubble_point_pressures` shape {bubble_point_pressures.shape} must be "
                        f"({len(solution_gors)}, {n_t})."
                    )
            else:
                raise ValidationError("`bubble_point_pressures` must be 1D or 2D.")

        # Validate 2D table shapes
        _2d_fields = [
            "viscosity_table",
            "density_table",
            "formation_volume_factor_table",
            "compressibility_table",
            "specific_gravity_table",
            "molecular_weight_table",
            "solution_gor_table",
            "compressibility_factor_table",
            "gas_free_fvf_table",
        ]
        for fname in _2d_fields:
            arr = getattr(data, fname, None)
            if arr is not None and arr.ndim == 2 and arr.shape != (n_p, n_t):
                raise ValidationError(
                    f"`{fname}` shape {arr.shape} must be (n_pressures={n_p}, "
                    f"n_temperatures={n_t})."
                )

        # Validate 3D table shapes. Only valid if salinities present
        _3d_fields = [
            "solubility_in_water_table",
            "bubble_point_pressure_table",
        ]
        # Shared fields can be either 2D (oil/gas) or 3D (water)
        _shared_fields = [
            "viscosity_table",
            "density_table",
            "formation_volume_factor_table",
            "compressibility_table",
            "specific_gravity_table",
            "molecular_weight_table",
        ]
        all_3d = _3d_fields + (_shared_fields if data.phase == FluidPhase.WATER else [])
        for fname in all_3d:
            arr = getattr(data, fname, None)
            if arr is None:
                continue
            if arr.ndim == 3:
                if n_s is None:
                    raise ValidationError(
                        f"`{fname}` is 3D but `salinities` was not provided."
                    )
                if arr.shape != (n_p, n_t, n_s):
                    raise ValidationError(
                        f"`{fname}` shape {arr.shape} must be "
                        f"(n_p={n_p}, n_t={n_t}, n_s={n_s})."
                    )

        # `solubility_in_water` requires salinities
        if data.solubility_in_water_table is not None and salinities is None:
            raise ValidationError(
                "`solubility_in_water_table` is 3D and requires `salinities`."
            )

    def _check_physical_consistency(self, data: PVTData) -> None:
        """Phase-aware physical sanity checks on table values."""
        phase = typing.cast(FluidPhase, data.phase)

        if data.viscosity_table is not None and np.any(data.viscosity_table <= 0):
            raise ValidationError(
                f"{phase.value.upper()} viscosity must be positive everywhere."
            )
        if data.density_table is not None and np.any(data.density_table <= 0):
            raise ValidationError(
                f"{phase.value.upper()} density must be positive everywhere."
            )
        if data.formation_volume_factor_table is not None and np.any(
            data.formation_volume_factor_table <= 0
        ):
            raise ValidationError(
                f"{phase.value.upper()} formation volume factor must be positive everywhere."
            )
        if (
            phase == FluidPhase.GAS
            and data.compressibility_factor_table is not None
            and np.any(data.compressibility_factor_table <= 0)
        ):
            raise ValidationError("Gas Z-factor must be positive everywhere.")

        # Gas density < reasonable oil density upper bound (50 lbm/ft³)
        if (
            phase == FluidPhase.GAS
            and data.density_table is not None
            and np.any(data.density_table >= 50.0)
        ):
            warnings.warn(
                "Gas density table contains values ≥ 50 lbm/ft³, which is unusually "
                "high. Verify units (expected lbm/ft³).",
                UserWarning,
                stacklevel=3,
            )

        logger.debug(
            f"Physical consistency checks passed for {phase.value.upper()} phase."
        )

    def _build_interpolators(self, data: PVTData) -> None:
        """Build scipy interpolator objects for all non-None tables."""
        phase = typing.cast(FluidPhase, data.phase)
        pressures = data.pressures
        temperatures = data.temperatures
        salinities = data.salinities
        k = _INTERPOLATION_DEGREES[self.interpolation_method]

        def _register_2d(name: str, table: typing.Optional[np.typing.NDArray]) -> None:
            """Register a 2D RectBivariateSpline interpolator."""
            if table is None:
                return
            self._interpolators[name] = RectBivariateSpline(
                x=pressures, y=temperatures, z=table, kx=k, ky=k
            )

        def _register_3d(name: str, table: typing.Optional[np.typing.NDArray]) -> None:
            """Register a 3D RegularGridInterpolator."""
            if table is None or salinities is None:
                return
            self._interpolators[name] = RegularGridInterpolator(
                points=(pressures, temperatures, salinities),
                values=table,
                method=self.interpolation_method,
                bounds_error=False,
                fill_value=None,  # extrapolate
            )

        # Shared tables
        if phase == FluidPhase.WATER:
            # Water shared tables are 3D
            _register_3d("viscosity", data.viscosity_table)
            _register_3d("density", data.density_table)
            _register_3d("formation_volume_factor", data.formation_volume_factor_table)
            _register_3d("compressibility", data.compressibility_table)
            _register_3d("specific_gravity", data.specific_gravity_table)
            _register_3d("molecular_weight", data.molecular_weight_table)
        else:
            # Oil / gas shared tables are 2D
            _register_2d("viscosity", data.viscosity_table)
            _register_2d("density", data.density_table)
            _register_2d("formation_volume_factor", data.formation_volume_factor_table)
            _register_2d("compressibility", data.compressibility_table)
            _register_2d("specific_gravity", data.specific_gravity_table)
            _register_2d("molecular_weight", data.molecular_weight_table)

        # Oil-specific tables
        if phase == FluidPhase.OIL:
            _register_2d("solution_gor", data.solution_gor_table)

            pb = data.bubble_point_pressures
            if pb is not None:
                if pb.ndim == 1:
                    self._interpolators["bubble_point_pressure"] = interp1d(
                        x=temperatures,
                        y=pb,
                        kind=self.interpolation_method,
                        bounds_error=False,
                        fill_value="extrapolate",  # type: ignore[arg-type]
                    )
                else:  # 2D Pb(Rs, T)
                    rs = data.solution_gas_to_oil_ratios
                    assert rs is not None
                    self._interpolators["bubble_point_pressure"] = RectBivariateSpline(
                        x=rs, y=temperatures, z=pb, kx=k, ky=k
                    )

        # Gas-specific
        if phase == FluidPhase.GAS:
            _register_2d("compressibility_factor", data.compressibility_factor_table)
            _register_3d("solubility_in_water", data.solubility_in_water_table)

        # Water-specific
        if phase == FluidPhase.WATER:
            _register_3d("bubble_point_pressure", data.bubble_point_pressure_table)
            _register_2d("gas_free_fvf", data.gas_free_fvf_table)

        logger.debug(
            f"Built {len(self._interpolators)} interpolators for "
            f"{phase.value.upper()} phase."
        )

    @property
    def phase(self) -> FluidPhase:
        """The fluid phase this table describes."""
        return typing.cast(FluidPhase, self._phase)

    def exists(self, name: str) -> bool:
        """Return True if an interpolator for *name* was built."""
        return name in self._interpolators

    def _warn_extrapolation(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> None:
        if not self.warn_on_extrapolation:
            return

        p_arr = np.atleast_1d(pressure)
        t_arr = np.atleast_1d(temperature)
        p_min, p_max = self._extrapolation_bounds["pressure"]
        t_min, t_max = self._extrapolation_bounds["temperature"]
        if np.any(p_arr < p_min) or np.any(p_arr > p_max):
            logger.warning(
                f"Pressure extrapolation: queried P ∈ "
                f"[{p_arr.min():.1f}, {p_arr.max():.1f}] psi, "
                f"table range [{p_min:.1f}, {p_max:.1f}] psi"
            )
        if np.any(t_arr < t_min) or np.any(t_arr > t_max):
            logger.warning(
                f"Temperature extrapolation: queried T ∈ "
                f"[{t_arr.min():.1f}, {t_arr.max():.1f}] °F, "
                f"table range [{t_min:.1f}, {t_max:.1f}] °F"
            )
        if salinity is not None and "salinity" in self._extrapolation_bounds:
            s_arr = np.atleast_1d(salinity)
            s_min, s_max = self._extrapolation_bounds["salinity"]
            if np.any(s_arr < s_min) or np.any(s_arr > s_max):
                logger.warning(
                    f"Salinity extrapolation: queried S ∈ "
                    f"[{s_arr.min():.0f}, {s_arr.max():.0f}] ppm, "
                    f"table range [{s_min:.0f}, {s_max:.0f}] ppm"
                )

    def _pt_interpolate(
        self, name: str, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """2D interpolation via `RectBivariateSpline.ev` with clamping."""
        interp = self._interpolators.get(name)
        if interp is None:
            return None
        self._warn_extrapolation(pressure, temperature)

        is_scalar = isinstance(pressure, (int, float)) and isinstance(
            temperature, (int, float)
        )
        if is_scalar:
            result: typing.Union[float, np.typing.NDArray] = float(
                interp.ev(pressure, temperature)
            )
        else:
            p = np.atleast_1d(pressure)
            t = np.atleast_1d(temperature)
            if p.shape != t.shape:
                if p.size == 1:
                    p = np.full_like(t, p[0])
                elif t.size == 1:
                    t = np.full_like(p, t[0])
                else:
                    raise ValidationError(
                        f"Incompatible shapes: pressure {p.shape}, temperature {t.shape}"
                    )
            result = interp.ev(p, t)

        clamps = self.clamps
        if name in clamps:
            lo, hi = clamps[name]
            if isinstance(result, np.ndarray):
                np.clip(result, lo, hi, out=result)
            else:
                result = float(np.clip(result, lo, hi))

        return (
            float(result)
            if isinstance(result, np.ndarray) and result.size == 1
            else result
        )

    def _pts_interpolate(
        self,
        name: str,
        pressure: QueryType,
        temperature: QueryType,
        salinity: QueryType,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """3D interpolation via `RegularGridInterpolator` with clamping."""
        interp = self._interpolators.get(name)
        if interp is None:
            return None
        self._warn_extrapolation(pressure, temperature, salinity)

        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)
        s = np.atleast_1d(salinity)
        p, t, s = np.broadcast_arrays(p, t, s)
        points = np.column_stack([p.ravel(), t.ravel(), s.ravel()])
        result = interp(points).reshape(p.shape).astype(get_dtype(), copy=False)

        clamps = self.clamps
        if name in clamps:
            lo, hi = clamps[name]
            np.clip(result, lo, hi, out=result)

        return float(result) if result.size == 1 else result

    def _resolve_salinity(self, salinity: typing.Optional[QueryType]) -> QueryType:
        """Return salinity, falling back to default_salinity for water tables."""
        if salinity is not None:
            return salinity
        if self.default_salinity is not None:
            return self.default_salinity
        raise ValidationError(
            "Salinity is required for water phase PVT lookup but none was provided "
            "and no default salinity is available."
        )

    def viscosity(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get fluid viscosity μ (cP).

        **Water / gas phase:** direct table interpolation.

        **Oil phase:**
        - Saturated (P ≤ Pb): interpolates directly from table.
        - Undersaturated (P > Pb): applies the modified Beggs & Robinson
          undersaturated viscosity correlation:

          `μo = μob * (P / Pb)^X`

          where `X = 2.6 * P^1.187 * exp(-11.513 - 8.98e-5 * P)`
          (Beggs & Robinson 1975, Vazquez & Beggs 1980).

          Falls back to the saturated table value at P if no bubble point
          table is available.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only; uses default if None.
        :param solution_gor: Solution GOR (SCF/STB). Oil phase only; required when
            the bubble point table is 2D — i.e. `Pb(Rs, T)`, so the correct
            bubble point can be located before applying the undersaturated
            correction. Not needed for 1D `Pb(T)` tables.
        :return: Viscosity in cP, or None if table not provided.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_interpolate(
                "viscosity", pressure, temperature, self._resolve_salinity(salinity)
            )

        if self._phase == FluidPhase.GAS:
            return self._pt_interpolate("viscosity", pressure, temperature)

        # For oil, apply undersaturated Beggs & Robinson correction above Pb
        if "viscosity" not in self._interpolators:
            return None

        pb = self.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )
        if pb is None:
            return self._pt_interpolate("viscosity", pressure, temperature)

        p_arr = np.atleast_1d(pressure)
        t_arr = np.atleast_1d(temperature)
        pb_arr = np.atleast_1d(pb)
        p_arr, t_arr, pb_arr = np.broadcast_arrays(p_arr, t_arr, pb_arr)

        result = np.zeros_like(p_arr, dtype=float)
        saturated_mask = p_arr <= pb_arr
        unsaturated_mask = ~saturated_mask

        if np.any(saturated_mask):
            result[saturated_mask] = self._pt_interpolate(  # type: ignore[index]
                "viscosity", p_arr[saturated_mask], t_arr[saturated_mask]
            )

        if np.any(unsaturated_mask):
            # Viscosity at bubble point (μob)
            mu_ob = self._pt_interpolate(
                "viscosity", pb_arr[unsaturated_mask], t_arr[unsaturated_mask]
            )
            if mu_ob is None:
                # Fallback: use table directly at current pressure
                result[unsaturated_mask] = self._pt_interpolate(  # type: ignore[index]
                    "viscosity", p_arr[unsaturated_mask], t_arr[unsaturated_mask]
                )
            else:
                p_under = p_arr[unsaturated_mask]
                pb_under = pb_arr[unsaturated_mask]
                mu_ob_arr = np.asarray(mu_ob)
                # Modified Beggs & Robinson (1975) / Vazquez & Beggs (1980)
                # X = 2.6 * P^1.187 * exp(-11.513 - 8.98e-5 * P)
                X = 2.6 * (p_under**1.187) * np.exp(-11.513 - 8.98e-5 * p_under)
                X = np.clip(X, 0.0, 5.0)  # prevent runaway exponentiation
                pressure_ratio = np.clip(
                    p_under / pb_under,
                    1.0,
                    None,  # P > Pb by definition
                )
                result[unsaturated_mask] = np.clip(
                    mu_ob_arr * (pressure_ratio**X),
                    mu_ob_arr,  # viscosity must not decrease above Pb
                    mu_ob_arr * 100.0,  # sanity cap at 100x bubble-point value
                )

        clamps = self.clamps
        if "viscosity" in clamps:
            lo, hi = clamps["viscosity"]
            np.clip(result, lo, hi, out=result)

        return float(result) if result.size == 1 else result

    def density(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get fluid density ρ (lbm/ft³).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only; uses default if None.
        :return: Density in lbm/ft³, or None if table not provided.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_interpolate(
                "density", pressure, temperature, self._resolve_salinity(salinity)
            )
        return self._pt_interpolate("density", pressure, temperature)

    def formation_volume_factor(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get formation volume factor B (bbl/STB for oil/water, ft³/SCF for gas).

        **Oil phase:**
        - Saturated (P ≤ Pb): interpolates directly from table.
        - Undersaturated (P > Pb): applies McCain compressibility correction:
          ``Bo = Bob * exp(-co_avg * (P - Pb))``. Falls back to Bob if no
          compressibility table is available.

        **Gas / water phase:** direct table interpolation.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only; uses default if None.
        :param solution_gor: Solution GOR (SCF/STB). Required for 2D oil Pb table.
        :return: FVF or None if table not provided.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_interpolate(
                "formation_volume_factor",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )

        if self._phase == FluidPhase.GAS:
            return self._pt_interpolate(
                "formation_volume_factor", pressure, temperature
            )

        # Oil: handle Pb correction
        if "formation_volume_factor" not in self._interpolators:
            return None

        pb = self.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )
        if pb is None:
            # No Pb table
            return self._pt_interpolate(
                "formation_volume_factor", pressure, temperature
            )

        p_arr = np.atleast_1d(pressure)
        t_arr = np.atleast_1d(temperature)
        pb_arr = np.atleast_1d(pb)
        p_arr, t_arr, pb_arr = np.broadcast_arrays(p_arr, t_arr, pb_arr)

        result = np.zeros_like(p_arr, dtype=float)

        saturated_mask = p_arr <= pb_arr
        unsaturated_mask = ~saturated_mask

        if np.any(saturated_mask):
            result[saturated_mask] = self._pt_interpolate(  # type: ignore[index]
                "formation_volume_factor", p_arr[saturated_mask], t_arr[saturated_mask]
            )

        if np.any(unsaturated_mask):
            oil_fvf_at_pb = self._pt_interpolate(
                "formation_volume_factor",
                pb_arr[unsaturated_mask],
                t_arr[unsaturated_mask],
            )
            if "compressibility" in self._interpolators:
                co_pb = self._pt_interpolate(
                    "compressibility", pb_arr[unsaturated_mask], t_arr[unsaturated_mask]
                )
                co_p = self._pt_interpolate(
                    "compressibility", p_arr[unsaturated_mask], t_arr[unsaturated_mask]
                )
                co_avg = 0.5 * (np.asarray(co_pb) + np.asarray(co_p))
                result[unsaturated_mask] = np.asarray(oil_fvf_at_pb) * np.exp(
                    -co_avg * (p_arr[unsaturated_mask] - pb_arr[unsaturated_mask])
                )
            else:
                result[unsaturated_mask] = np.asarray(oil_fvf_at_pb)

        # Apply clamps
        clamps = self.clamps
        if "formation_volume_factor" in clamps:
            lo, hi = clamps["formation_volume_factor"]
            np.clip(result, lo, hi, out=result)

        return float(result) if result.size == 1 else result

    def compressibility(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get fluid compressibility c (psi⁻¹).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only; uses default if None.
        :return: Compressibility in psi⁻¹, or None if table not provided.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_interpolate(
                "compressibility",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_interpolate("compressibility", pressure, temperature)

    def specific_gravity(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get fluid specific gravity γ (dimensionless, water = 1 for oil/water,
        air = 1 for gas).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only; uses default if None.
        :return: Specific gravity or None if table not provided.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_interpolate(
                "specific_gravity",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_interpolate("specific_gravity", pressure, temperature)

    def molecular_weight(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get fluid molecular weight `M` (lbm/lb-mol).

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl). Water phase only; uses default if None.
        :return: Molecular weight or None if table not provided.
        """
        if self._phase == FluidPhase.WATER:
            return self._pts_interpolate(
                "molecular_weight",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )
        return self._pt_interpolate("molecular_weight", pressure, temperature)

    def bubble_point_pressure(
        self,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
        pressure: typing.Optional[QueryType] = None,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get bubble point pressure Pb (psi).

        **Oil phase:**
        - 1D table Pb(T): `temperature` only.
        - 2D table Pb(Rs, T): both `temperature` and `solution_gor` required.

        **Water phase:** water bubble point pressure(P, T, S) - `pressure` and `salinity` required
        (or defaults to `default_salinity` if salinity is None).

        **Gas phase:** returns `None`.

        :param temperature: Temperature (°F).
        :param solution_gor: Solution GOR (SCF/STB). Required for 2D oil Pb table.
        :param pressure: Pressure (psi). Required for water Pb table.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :return: Bubble point pressure in psi, or None for gas phase / missing table.
        """
        if self._phase == FluidPhase.GAS:
            return None

        if self._phase == FluidPhase.WATER:
            if pressure is None:
                raise ValidationError(
                    "Water bubble point pressure requires `pressure` argument."
                )
            return self._pts_interpolate(
                "bubble_point_pressure",
                pressure,
                temperature,
                self._resolve_salinity(salinity),
            )

        # Oil
        interp = self._interpolators.get("bubble_point_pressure")
        if interp is None:
            return None

        if self._pb_ndim == 1:
            result = interp(temperature)
            return float(result) if np.isscalar(temperature) else result

        # 2D Pb(Rs, T)
        if solution_gor is None:
            raise ValidationError(
                "2D bubble point table requires `solution_gor` (solution GOR) argument."
            )
        rs = np.atleast_1d(solution_gor)
        t = np.atleast_1d(temperature)
        if rs.shape != t.shape:
            if rs.size == 1:
                rs = np.full_like(t, rs[0])
            elif t.size == 1:
                t = np.full_like(rs, t[0])
            else:
                raise ValidationError(
                    "`solution_gor` and `temperature` must have compatible shapes."
                )
        result = interp.ev(rs, t)
        return float(result) if result.size == 1 else result

    def is_saturated(
        self,
        pressure: QueryType,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[bool, np.typing.NDArray]]:
        """
        Determine whether conditions are saturated (P ≤ Pb).

        **Oil and water phase only.** Returns `None` for gas.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param solution_gor: Solution GOR (SCF/STB). Required for 2D Pb tables.
        :return: Boolean mask (where True = saturated), or None for gas phase.
        """
        if self._phase == FluidPhase.GAS:
            return None

        pb = self.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )
        if pb is None:
            return None

        p_arr = np.atleast_1d(pressure)
        pb_arr = np.atleast_1d(pb)
        result = p_arr <= pb_arr
        return bool(result) if result.size == 1 else result

    def solution_gas_to_oil_ratio(
        self,
        pressure: QueryType,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get solution gas-oil ratio Rs (SCF/STB).

        **Oil phase only.** Rs is frozen at Rsb above the bubble point.
        Returns `None` for gas and water phases.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param solution_gor: Solution GOR (SCF/STB). Required for 2D oil Pb table.
        :return: Rs in SCF/STB, or None.
        """
        if self._phase != FluidPhase.OIL:
            return None
        if "solution_gor" not in self._interpolators:
            return None

        pb = self.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )
        if pb is None:
            return self._pt_interpolate("solution_gor", pressure, temperature)

        p_arr = np.atleast_1d(pressure)
        t_arr = np.atleast_1d(temperature)
        pb_arr = np.atleast_1d(pb)
        p_arr, t_arr, pb_arr = np.broadcast_arrays(p_arr, t_arr, pb_arr)

        result = np.zeros_like(p_arr, dtype=float)
        saturated_mask = p_arr <= pb_arr
        unsaturated_mask = ~saturated_mask

        if np.any(saturated_mask):
            result[saturated_mask] = self._pt_interpolate(
                "solution_gor", p_arr[saturated_mask], t_arr[saturated_mask]
            )
        if np.any(unsaturated_mask):
            # Rs frozen at Rsb above Pb
            result[unsaturated_mask] = self._pt_interpolate(
                "solution_gor", pb_arr[unsaturated_mask], t_arr[unsaturated_mask]
            )

        clamps = self.clamps
        if "solution_gor" in clamps:
            lo, hi = clamps["solution_gor"]
            np.clip(result, lo, hi, out=result)

        return float(result) if result.size == 1 else result

    def compressibility_factor(
        self,
        pressure: QueryType,
        temperature: QueryType,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get gas compressibility factor Z (dimensionless).

        **Gas phase only.** Returns `None` for oil and water phases.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :return: Z-factor or None.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pt_interpolate("compressibility_factor", pressure, temperature)

    def solubility_in_water(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.typing.NDArray]]:
        """
        Get gas solubility in water Rsw (SCF/STB).

        **Gas phase only.** Returns `None` for oil and water phases.

        :param pressure: Pressure (psi).
        :param temperature: Temperature (°F).
        :param salinity: Salinity (ppm NaCl); uses default if None.
        :return: Rsw in SCF/STB, or None.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self._pts_interpolate(
            "solubility_in_water",
            pressure,
            temperature,
            self._resolve_salinity(salinity),
        )


@attrs.frozen
class PVTDataSet(StoreSerializable):
    """
    Serializable bundle of raw `PVTData` for all three fluid phases.

    `PVTDataSet` is the persistence layer. It stores the raw tabulated
    data for oil, gas, and water and can be saved to / loaded from disk
    independently of any interpolator settings. Use it when you want to
    share or inspect PVT data without rebuilding tables, or as an
    intermediate step before constructing a `PVTTables` bundle.

    Typical workflow:

    ```python
    # Build once and persist
    oil_data, gas_data, water_data = build_pvt_dataset(pressures, temperatures, ...)
    dataset = PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)
    dataset.save("run/pvt_data.h5")

    # Load and build tables later (possibly with different interp settings)
    dataset = PVTDataSet.load("run/pvt_data.h5")
    tables = PVTTables.from_dataset(dataset, interpolation_method="cubic")
    ```

    See also `PVTTables.from_dataset(...)` and `PVTTables.from_files(...)`.
    """

    oil: typing.Optional[PVTData] = None
    """Raw PVT data for the oil phase."""

    gas: typing.Optional[PVTData] = None
    """Raw PVT data for the gas phase."""

    water: typing.Optional[PVTData] = None
    """Raw PVT data for the water phase."""

    @classmethod
    def from_files(
        cls,
        oil: typing.Optional[typing.Union[PathLike[str], str]] = None,
        gas: typing.Optional[typing.Union[PathLike[str], str]] = None,
        water: typing.Optional[typing.Union[PathLike[str], str]] = None,
    ) -> Self:
        """
        Load a `PVTDataSet` from individual per-phase files.

        Each path is optional. Pass only the phases you need. Files must
        have been created by saving a `PVTData` instance (e.g. via
        `pvt_data.save(path)`).

        :param oil: Path to serialized oil `PVTData`.
        :param gas: Path to serialized gas `PVTData`.
        :param water: Path to serialized water `PVTData`.
        :return: `PVTDataSet` with the requested phases populated.

        Example:

        ```python
        dataset = PVTDataSet.from_files(
            oil="data/oil_pvt.h5", gas="data/gas_pvt.h5"
        )
        ```
        """
        return cls(
            oil=PVTData.from_file(oil) if oil is not None else None,
            gas=PVTData.from_file(gas) if gas is not None else None,
            water=PVTData.from_file(water) if water is not None else None,
        )


@attrs.frozen
class PVTTables(StoreSerializable):
    """
    Bundle of phase-specific `PVTTable` instances.

    Holds one optional `PVTTable` per fluid phase. Access via:

    ```python
    pvt_tables.oil.viscosity(p, t)
    pvt_tables.gas.formation_volume_factor(p, t)
    pvt_tables.water.density(p, t, salinity=35000)
    ```

    ```python
    tables.save("run/pvt_tables.h5")
    tables2 = PVTTables.load("run/pvt_tables.h5")
    ```

    Build from a `PVTDataSet` or directly from per-phase files without touching `PVTData`:

    ```python
    # From a dataset
    tables = PVTTables.from_dataset(dataset, interpolation_method="cubic")

    # From individual files
    tables = PVTTables.from_files(
        oil="oil.h5",
        gas="gas.h5",
        water="water.h5",
        interpolation_method="linear",
    )
    ```
    """

    oil: typing.Optional[PVTTable] = None
    """PVT table for the oil phase."""

    gas: typing.Optional[PVTTable] = None
    """PVT table for the gas phase."""

    water: typing.Optional[PVTTable] = None
    """PVT table for the water phase."""

    @classmethod
    def from_dataset(
        cls,
        dataset: PVTDataSet,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        clamps: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None,
    ) -> Self:
        """
        Build a `PVTTables` bundle from a `PVTDataSet`.

        Constructs one `PVTTable` per non-`None` phase in the
        dataset, applying the same interpolation settings to all phases.

        :param dataset: Source `PVTDataSet` containing raw tabulated data.
        :param interpolation_method: `"linear"` (default) or `"cubic"`.
            Cubic requires ≥ 4 points along each axis.
        :param validate: Run physical-consistency checks on each phase's data.
        :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
        :param clamps: Property clamp overrides applied to all phases.
            Keys use generic names (e.g. `"viscosity"`). Merged on top of
            per-phase defaults.
        :return: `PVTTables` with interpolators ready for simulation.

        Example:

        ```python
        dataset = PVTDataSet.load("run/pvt_data.h5")
        tables  = PVTTables.from_dataset(dataset, interpolation_method="cubic")
        ```
        """
        kwargs = dict(  # noqa
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            clamps=clamps,
        )
        return cls(
            oil=PVTTable(dataset.oil, **kwargs) if dataset.oil is not None else None,  # type: ignore
            gas=PVTTable(dataset.gas, **kwargs) if dataset.gas is not None else None,  # type: ignore
            water=PVTTable(dataset.water, **kwargs)  # type: ignore
            if dataset.water is not None
            else None,
        )

    @classmethod
    def from_files(
        cls,
        oil: typing.Optional[typing.Union[PathLike[str], str]] = None,
        gas: typing.Optional[typing.Union[PathLike[str], str]] = None,
        water: typing.Optional[typing.Union[PathLike[str], str]] = None,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        clamps: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None,
    ) -> Self:
        """
        Build a `PVTTables` bundle directly from per-phase data files.

        A convenience wrapper around `from_dataset` that loads the
        `PVTData` files for you. Each path is optional. Pass only
        the phases you need.

        :param oil: Path to serialized oil `PVTData` file.
        :param gas: Path to serialized gas `PVTData` file.
        :param water: Path to serialized water `PVTData` file.
        :param interpolation_method: `"linear"` (default) or `"cubic"`.
        :param validate: Run physical-consistency checks on each phase's data.
        :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
        :param clamps: Property clamp overrides applied to all phases.
        :return: `PVTTables` with interpolators ready for simulation.

        Example:

        ```python
        tables = PVTTables.from_files(
            oil="data/oil_pvt.h5",
            gas="data/gas_pvt.h5",
            interpolation_method="linear",
        )
        ```
        """
        dataset = PVTDataSet.from_files(oil=oil, gas=gas, water=water)
        return cls.from_dataset(
            dataset,
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            clamps=clamps,
        )

    @property
    def dataset(self) -> PVTDataSet:
        """
        Return the underlying raw data as a `PVTDataSet`.

        Extracts the `PVTData` stored internally on each
        `PVTTable` slot. Useful when you want to persist the raw
        data separately from the interpolator settings, or inspect the
        tables without re-running the builders.

        :return: `PVTDataSet` containing the raw data for each
            non-`None` phase.

        Example:

        ```python
        # Save raw data independently of interpolator settings
        tables.dataset.save("run/pvt_data.h5")
        ```
        """
        return PVTDataSet(
            oil=self.oil._data if self.oil is not None else None,
            gas=self.gas._data if self.gas is not None else None,
            water=self.water._data if self.water is not None else None,
        )


def _resolve_gas(
    gas: typing.Optional[typing.Any],
    gas_gravity: typing.Optional[float],
) -> typing.Tuple[str, typing.Optional[float], typing.Optional["PVTTable"]]:
    """
    Resolve the `gas` argument used across all three builders.

    Returns `(gas_name, gas_gravity, gas_pvt_table)` where:
    - `gas_name` - String name for correlation lookup
    - `gas_gravity` - Scalar gravity (may be None if taken from `pvt_table`)
    - `gas_pvt_table` - `PVTTable` to read gas properties from directly (may be None)
    """
    from bores.fluids import Fluid  # type: ignore[import]

    gas_pvt_table: typing.Optional[PVTTable] = None
    gas_name: str = c.RESERVOIR_GAS

    if gas is None:
        pass
    elif isinstance(gas, str):
        gas_name = gas
    elif Fluid is not None and isinstance(gas, Fluid):
        gas_name = gas.name
        if gas.pvt_table is not None:
            gas_pvt_table = gas.pvt_table
    # else: unknown type - fall through to default

    # If `gas_gravity` not supplied, try to get a scalar from the `pvt_table`
    if gas_gravity is None and gas_pvt_table is not None:
        # Evaluate at a representative mid-range pressure for a scalar
        avg_pressure = float(np.mean(gas_pvt_table._extrapolation_bounds["pressure"]))
        avg_temperature = float(
            np.mean(gas_pvt_table._extrapolation_bounds["temperature"])
        )
        specific_gravity = gas_pvt_table.specific_gravity(avg_pressure, avg_temperature)
        gas_gravity = float(specific_gravity) if specific_gravity is not None else None

    if gas_gravity is None:
        gas_gravity = compute_gas_gravity(gas=gas_name)

    return gas_name, gas_gravity, gas_pvt_table


def _tables_from_gas_pvt_table(
    gas_pvt_table: PVTTable,
    pressure_grid: np.typing.NDArray,
    temperature_grid: np.typing.NDArray,
    salinity_grid: typing.Optional[np.typing.NDArray] = None,
) -> typing.Dict[str, typing.Optional[np.typing.NDArray]]:
    """
    Evaluate a gas `PVTTable` on a (P,T) meshgrid and return the resulting
    2D arrays keyed by table name. Used when a `Fluid` with a `pvt_table` is
    passed to a builder.
    """
    flat_pressure = pressure_grid.ravel()
    flat_temperature = temperature_grid.ravel()
    shape = pressure_grid.shape

    def _eval(method_name: str) -> typing.Optional[np.typing.NDArray]:
        method = getattr(gas_pvt_table, method_name, None)
        if method is None:
            return None
        result = method(flat_pressure, flat_temperature)
        if result is None:
            return None
        return np.asarray(result).reshape(shape).astype(get_dtype(), copy=False)

    return {
        "viscosity_table": _eval("viscosity"),
        "density_table": _eval("density"),
        "formation_volume_factor_table": _eval("formation_volume_factor"),
        "compressibility_table": _eval("compressibility"),
        "compressibility_factor_table": _eval("compressibility_factor"),
        "specific_gravity_table": _eval("specific_gravity"),
        "molecular_weight_table": _eval("molecular_weight"),
    }


def build_oil_pvt_data(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    oil_specific_gravity: float = 0.85,
    gas_gravity: typing.Optional[float] = None,
    estimated_solution_gor: typing.Optional[float] = None,
    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None,
    solution_gas_to_oil_ratios: typing.Optional[OneDimensionalGrid] = None,
    gas: typing.Optional[typing.Union[str, typing.Any]] = None,
    # Pre-computed tables (all optional, 2D shape (n_p, n_t))
    viscosity_table: typing.Optional[TwoDimensionalGrid] = None,
    density_table: typing.Optional[TwoDimensionalGrid] = None,
    formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None,
    compressibility_table: typing.Optional[TwoDimensionalGrid] = None,
    specific_gravity_table: typing.Optional[TwoDimensionalGrid] = None,
    molecular_weight_table: typing.Optional[TwoDimensionalGrid] = None,
    solution_gas_to_oil_ratio_table: typing.Optional[TwoDimensionalGrid] = None,
    **kwargs: typing.Any,
) -> PVTData:
    """
    Build a `PVTData` for the oil phase.

    Computes missing tables from empirical correlations (Standing bubble point,
    iterative oil compressibility andoil FVF bootstrap, Beggs & Robinson viscosity, etc.) and packages
    everything into a `PVTData(phase=OIL)`.

    :param pressures: 1D array of pressures (psi), strictly increasing.
    :param temperatures: 1D array of temperatures (°F), strictly increasing.
    :param oil_specific_gravity: Oil specific gravity (dimensionless, water=1).
    :param gas_gravity: Gas specific gravity (air=1). Derived from `gas` if not provided.
    :param estimated_solution_gor: Estimated solution gas-oil ration (SCF/STB) for 1D bubble point pressure correlation.
        If None, estimated from API gravity.
    :param bubble_point_pressures: Pre-computed bubble point pressures. 1D Pb(T) or 2D Pb(Rs,T).
    :param solution_gas_to_oil_ratios: Solution gas-to-oil ratio axis for 2D Pb table.
    :param gas: Gas identity. Either a string name (e.g. `"Methane"`)
        or a `Fluid` instance. If a `Fluid` with a `pvt_table`
        is supplied, gas properties are read directly from that table instead of
        being recomputed from correlations.
    :param viscosity_table: Pre-computed oil viscosity (n_p, n_t) in cP.
    :param density_table: Pre-computed oil density (n_p, n_t) in lbm/ft³.
    :param formation_volume_factor_table: Pre-computed oil formation volume factor (n_p, n_t) in bbl/STB.
    :param compressibility_table: Pre-computed oil compressibility (n_p, n_t) in psi⁻¹.
    :param specific_gravity_table: Pre-computed oil specific gravity (n_p, n_t).
    :param molecular_weight_table: Pre-computed oil molecular weight (n_p, n_t).
    :param solution_gas_to_oil_ratio_table: Pre-computed solution gas-oil ratio, `Rs(P,T)`, (n_p, n_t) in SCF/STB.
    :return: `PVTData` with `phase=OIL`.
    """
    if pressures.ndim != 1 or not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be a strictly increasing 1D array.")
    if temperatures.ndim != 1 or not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be a strictly increasing 1D array.")

    _, gas_gravity, gas_pvt_table = _resolve_gas(gas, gas_gravity)
    assert gas_gravity is not None

    dtype = get_dtype()
    n_p = len(pressures)
    n_t = len(temperatures)

    pressure_grid, temperature_grid = np.meshgrid(
        pressures, temperatures, indexing="ij"
    )

    # Gas tables needed for oil Oil Compressibility anf FVF calculation
    gas_fvf_grid: typing.Optional[np.typing.NDArray] = None
    if gas_pvt_table is not None:
        gas_tables = _tables_from_gas_pvt_table(
            gas_pvt_table, pressure_grid, temperature_grid
        )
        gas_fvf_grid = gas_tables.get("formation_volume_factor_table")

    if gas_fvf_grid is None:
        z_factor_grid = build_gas_compressibility_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
        )
        gas_fvf_grid = build_gas_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_compressibility_factor_grid=z_factor_grid,
        )

    if specific_gravity_table is None:
        specific_gravity_table = np.full((n_p, n_t), oil_specific_gravity, dtype=dtype)

    oil_api_gravity_table = build_oil_api_gravity_grid(
        oil_specific_gravity_grid=specific_gravity_table
    )

    if molecular_weight_table is None:
        # Approximate from API: molecular weight ≈ 6084 / (API - 5.9)  (Ahmed 2006)
        oil_api = compute_oil_api_gravity(oil_specific_gravity)
        molecular_weight = float(np.clip(6084.0 / max(oil_api - 5.9, 1.0), 10.0, 600.0))
        molecular_weight_table = np.full((n_p, n_t), molecular_weight, dtype=dtype)

    if bubble_point_pressures is None:
        gas_gravity_grid_1d = np.full(n_t, gas_gravity, dtype=dtype)
        oil_api_gravity_grid = build_oil_api_gravity_grid(
            np.full(n_t, oil_specific_gravity, dtype=dtype)
        )
        if solution_gas_to_oil_ratios is not None:
            # 2D Pb(Rs, T)
            n_rs = len(solution_gas_to_oil_ratios)
            bubble_point_pressures = np.zeros((n_rs, n_t), dtype=dtype)
            for i, solution_gor in enumerate(solution_gas_to_oil_ratios):
                bubble_point_pressures[i, :] = build_oil_bubble_point_pressure_grid(
                    gas_gravity_grid=gas_gravity_grid_1d,
                    oil_api_gravity_grid=oil_api_gravity_grid,
                    temperature_grid=temperatures,
                    solution_gas_to_oil_ratio_grid=np.full(
                        n_t, solution_gor, dtype=dtype
                    ),
                )
        else:
            # 1D Pb(T)
            oil_api = compute_oil_api_gravity(oil_specific_gravity)
            if estimated_solution_gor is not None:
                estimated_gor = estimated_solution_gor
            else:
                estimated_gor = float(
                    np.clip(10 ** (0.0125 * oil_api) * 50.0, 50.0, 2000.0)
                )
                warnings.warn(
                    f"No `estimated_solution_gor` provided. Estimating Rs = {estimated_gor:.1f} "
                    f"SCF/STB from API = {oil_api:.1f}°. Pass `estimated_solution_gor` "
                    "for best results.",
                    UserWarning,
                    stacklevel=2,
                )
            bubble_point_pressures = build_oil_bubble_point_pressure_grid(
                gas_gravity_grid=gas_gravity_grid_1d,
                oil_api_gravity_grid=oil_api_gravity_grid,
                temperature_grid=temperatures,
                solution_gas_to_oil_ratio_grid=np.full(n_t, estimated_gor, dtype=dtype),
            )

    # 2D Pb grid for property table construction
    if bubble_point_pressures.ndim == 1:
        bubble_point_pressure_grid = np.broadcast_to(
            bubble_point_pressures[np.newaxis, :], (n_p, n_t)
        ).copy()
    else:
        # Pb(Rs, T) - need Rs(P,T) to map each (P,T) to a Pb
        if solution_gas_to_oil_ratio_table is None:
            solution_gas_to_oil_ratio_table = build_estimated_solution_gas_to_oil_ratio_grid(
                pressure_grid=pressure_grid,  # type: ignore[arg-type]
                temperature_grid=temperature_grid,
                oil_api_gravity_grid=oil_api_gravity_table,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                max_iterations=20,
                tolerance=1e-4,
            )

        pb_interpolator = RectBivariateSpline(
            x=solution_gas_to_oil_ratios,  # type: ignore[arg-type]
            y=temperatures,
            z=bubble_point_pressures,
            kx=1,
            ky=1,
        )
        temperature_flat = np.broadcast_to(temperatures, (n_p, n_t))
        assert solution_gas_to_oil_ratio_table is not None
        bubble_point_pressure_grid = (
            pb_interpolator.ev(solution_gas_to_oil_ratio_table.ravel(), temperature_flat.ravel())
            .reshape(n_p, n_t)
            .astype(dtype)
        )

    gas_gravity_grid = np.full((n_p, n_t), gas_gravity, dtype=dtype)
    if solution_gas_to_oil_ratio_table is None:
        solution_gas_to_oil_ratio_table = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=bubble_point_pressure_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_table,
        )

    # Solve for oil FVF and compressibility iteratively
    needs_fvf = formation_volume_factor_table is None
    needs_compressibility = compressibility_table is None

    gor_at_bubble_point_pressure_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=bubble_point_pressure_grid,
        temperature_grid=temperature_grid,
        bubble_point_pressure_grid=bubble_point_pressure_grid,
        gas_gravity_grid=gas_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_table,
    )

    if needs_fvf and needs_compressibility:
        estimated_compressibility = np.full((n_p, n_t), 1e-5, dtype=dtype)
        max_delta = 0.0
        for iteration in range(10):
            formation_volume_factor_table = build_oil_formation_volume_factor_grid(
                pressure_grid=pressure_grid,  # type: ignore[arg-type]
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=bubble_point_pressure_grid,
                oil_specific_gravity_grid=specific_gravity_table,
                gas_gravity_grid=gas_gravity_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,  # type: ignore[arg-type]
                oil_compressibility_grid=estimated_compressibility,
            )
            new_compressibility = build_oil_compressibility_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=bubble_point_pressure_grid,
                oil_api_gravity_grid=oil_api_gravity_table,
                gas_gravity_grid=gas_gravity_grid,
                gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
                gas_formation_volume_factor_grid=gas_fvf_grid,
                oil_formation_volume_factor_grid=formation_volume_factor_table,  # type: ignore[arg-type]
            )
            max_delta = float(
                np.max(np.abs(new_compressibility - estimated_compressibility))
            )
            logger.debug(
                f"Bo/Co bootstrap iteration {iteration + 1}/10: "
                f"max ΔCo = {max_delta:.3e} psi⁻¹"
            )
            estimated_compressibility = new_compressibility
            if max_delta < 1e-7:
                logger.debug(
                    f"Bo/Co bootstrap converged in {iteration + 1} iteration(s)."
                )
                break
        else:
            warnings.warn(
                f"Bo/Co bootstrap did not converge within 10 iterations "
                f"(final max ΔCo = {max_delta:.3e} psi⁻¹).",
                UserWarning,
                stacklevel=2,
            )
        compressibility_table = estimated_compressibility  # type: ignore[assignment]

    elif needs_fvf:
        assert compressibility_table is not None
        formation_volume_factor_table = build_oil_formation_volume_factor_grid(
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=bubble_point_pressure_grid,
            oil_specific_gravity_grid=specific_gravity_table,
            gas_gravity_grid=gas_gravity_grid,
            solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,  # type: ignore[arg-type]
            oil_compressibility_grid=compressibility_table,
        )

    elif needs_compressibility:
        assert formation_volume_factor_table is not None
        compressibility_table = build_oil_compressibility_grid(
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=bubble_point_pressure_grid,
            oil_api_gravity_grid=oil_api_gravity_table,
            gas_gravity_grid=gas_gravity_grid,
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
            gas_formation_volume_factor_grid=gas_fvf_grid,
            oil_formation_volume_factor_grid=formation_volume_factor_table,
        )

    if density_table is None:
        density_table = build_live_oil_density_grid(
            oil_api_gravity_grid=oil_api_gravity_table,
            gas_gravity_grid=gas_gravity_grid,
            solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,  # type: ignore[arg-type]
            formation_volume_factor_grid=formation_volume_factor_table,  # type: ignore[arg-type]
        )

    if viscosity_table is None:
        viscosity_table = build_oil_viscosity_grid(  # type: ignore[assignment]
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=bubble_point_pressure_grid,
            oil_specific_gravity_grid=specific_gravity_table,
            solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,  # type: ignore[arg-type]
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
        )

    return PVTData(
        phase=FluidPhase.OIL,
        pressures=pressures,
        temperatures=temperatures,
        bubble_point_pressures=bubble_point_pressures,
        solution_gas_to_oil_ratios=solution_gas_to_oil_ratios,
        viscosity_table=viscosity_table,
        density_table=density_table,
        formation_volume_factor_table=formation_volume_factor_table,
        compressibility_table=compressibility_table,
        specific_gravity_table=specific_gravity_table,
        molecular_weight_table=molecular_weight_table,
        solution_gor_table=solution_gas_to_oil_ratio_table,
    )


def build_gas_pvt_data(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    gas_gravity: typing.Optional[float] = None,
    gas: typing.Optional[typing.Union[str, typing.Any]] = None,
    water_salinities: typing.Optional[OneDimensionalGrid] = None,
    # Pre-computed tables (all optional, 2D shape (n_p, n_t))
    viscosity_table: typing.Optional[TwoDimensionalGrid] = None,
    density_table: typing.Optional[TwoDimensionalGrid] = None,
    formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None,
    compressibility_table: typing.Optional[TwoDimensionalGrid] = None,
    compressibility_factor_table: typing.Optional[TwoDimensionalGrid] = None,
    specific_gravity_table: typing.Optional[TwoDimensionalGrid] = None,
    molecular_weight_table: typing.Optional[TwoDimensionalGrid] = None,
    solubility_in_water_table: typing.Optional[ThreeDimensionalGrid] = None,
    **kwargs: typing.Any,
) -> PVTData:
    """
    Build a `PVTData` for the gas phase.

    Computes missing tables from correlations (DAK Z-factor, Lee-Kesler viscosity,
    etc.) or reads them from a `Fluid`'s `pvt_table` if supplied.

    :param pressures: 1D array of pressures (psi), strictly increasing.
    :param temperatures: 1D array of temperatures (°F), strictly increasing.
    :param gas_gravity: Gas specific gravity (air=1). Derived from `gas` if None.
    :param gas: Gas identity. String name or `Fluid`. If a `Fluid` with a `pvt_table` is supplied,
        tables are populated from it directly.
    :param water_salinities: 1D salinity array (ppm) for Rsw(P,T,S) table.
        If None, `solubility_in_water_table` is not built.
    :param viscosity_table: Pre-computed gas viscosity (n_p, n_t) in cP.
    :param density_table: Pre-computed gas density (n_p, n_t) in lbm/ft³.
    :param formation_volume_factor_table: Pre-computed Bg (n_p, n_t) in ft³/SCF.
    :param compressibility_table: Pre-computed gas compressibility (n_p, n_t) in psi⁻¹.
    :param compressibility_factor_table: Pre-computed gas compressibility factor (n_p, n_t).
    :param specific_gravity_table: Pre-computed gas specific gravity (n_p, n_t).
    :param molecular_weight_table: Pre-computed gas molecular weight (n_p, n_t) in lbm/lb-mol.
    :param solubility_in_water_table: Pre-computed gas solubility in water (n_p, n_t, n_s) in SCF/STB.
    :return: `PVTData` with `phase=GAS`.
    """
    if pressures.ndim != 1 or not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be a strictly increasing 1D array.")
    if temperatures.ndim != 1 or not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be a strictly increasing 1D array.")

    gas_name, gas_gravity, gas_pvt_table = _resolve_gas(gas, gas_gravity)
    assert gas_gravity is not None

    dtype = get_dtype()
    n_p = len(pressures)
    n_t = len(temperatures)

    pressure_grid, temperature_grid = np.meshgrid(
        pressures, temperatures, indexing="ij"
    )

    # If a Fluid pvt_table is available, evaluate all tables from it
    if gas_pvt_table is not None:
        gas_tables = _tables_from_gas_pvt_table(
            gas_pvt_table, pressure_grid, temperature_grid
        )
        if viscosity_table is None:
            viscosity_table = gas_tables.get("viscosity_table")  # type: ignore[arg-type]
        if density_table is None:
            density_table = gas_tables.get("density_table")  # type: ignore[arg-type]
        if formation_volume_factor_table is None:
            formation_volume_factor_table = gas_tables.get(  # type: ignore[arg-type]
                "formation_volume_factor_table"
            )
        if compressibility_table is None:
            compressibility_table = gas_tables.get("compressibility_table")  # type: ignore[arg-type]
        if compressibility_factor_table is None:
            compressibility_factor_table = gas_tables.get(  # type: ignore[arg-type]
                "compressibility_factor_table"
            )
        if specific_gravity_table is None:
            specific_gravity_table = gas_tables.get("specific_gravity_table")  # type: ignore[arg-type]
        if molecular_weight_table is None:
            molecular_weight_table = gas_tables.get("molecular_weight_table")  # type: ignore[arg-type]

    gas_gravity_grid = np.full((n_p, n_t), gas_gravity, dtype=dtype)
    if specific_gravity_table is None:
        specific_gravity_table = gas_gravity_grid.copy()

    if molecular_weight_table is None:
        molecular_weight_table = build_gas_molecular_weight_grid(
            gas_gravity_grid=specific_gravity_table
        )

    if compressibility_factor_table is None:
        compressibility_factor_table = build_gas_compressibility_factor_grid(
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
        )

    if formation_volume_factor_table is None:
        formation_volume_factor_table = build_gas_formation_volume_factor_grid(  # type: ignore[asignment]
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            gas_compressibility_factor_grid=compressibility_factor_table,  # type: ignore[arg-type]
        )

    if compressibility_table is None:
        compressibility_table = build_gas_compressibility_grid(
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
            gas_compressibility_factor_grid=compressibility_factor_table,
        )

    if density_table is None:
        density_table = build_gas_density_grid(  # type: ignore[asignment]
            pressure_grid=pressure_grid,  # type: ignore[arg-type]
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
            gas_compressibility_factor_grid=compressibility_factor_table,  # type: ignore[arg-type]
        )

    if viscosity_table is None:
        viscosity_table = build_gas_viscosity_grid(  # type: ignore[aasignment]
            temperature_grid=temperature_grid,
            gas_density_grid=density_table,  # type: ignore[arg-type]
            gas_molecular_weight_grid=molecular_weight_table,
        )

    # Solubility in water (3D)
    if solubility_in_water_table is None and water_salinities is not None:
        pressures_3d, temperatures_3d, salinities_3d = np.meshgrid(
            pressures, temperatures, water_salinities, indexing="ij"
        )
        solubility_in_water_table = build_gas_solubility_in_water_grid(  # type: ignore[assignment]
            pressure_grid=pressures_3d,
            temperature_grid=temperatures_3d,
            salinity_grid=salinities_3d,
            gas=gas_name,
        )

    return PVTData(
        phase=FluidPhase.GAS,
        pressures=pressures,
        temperatures=temperatures,
        salinities=water_salinities,
        viscosity_table=viscosity_table,
        density_table=density_table,
        formation_volume_factor_table=formation_volume_factor_table,
        compressibility_table=compressibility_table,
        compressibility_factor_table=compressibility_factor_table,
        specific_gravity_table=specific_gravity_table,
        molecular_weight_table=molecular_weight_table,
        solubility_in_water_table=solubility_in_water_table,
    )


def build_water_pvt_data(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    salinities: typing.Optional[OneDimensionalGrid] = None,
    water_salinity: typing.Optional[float] = None,
    gas_gravity: typing.Optional[float] = None,
    gas: typing.Optional[typing.Union[str, typing.Any]] = None,
    # Pre-computed tables (all optional, 3D shape (n_p, n_t, n_s))
    viscosity_table: typing.Optional[ThreeDimensionalGrid] = None,
    density_table: typing.Optional[ThreeDimensionalGrid] = None,
    formation_volume_factor_table: typing.Optional[ThreeDimensionalGrid] = None,
    compressibility_table: typing.Optional[ThreeDimensionalGrid] = None,
    specific_gravity_table: typing.Optional[ThreeDimensionalGrid] = None,
    molecular_weight_table: typing.Optional[ThreeDimensionalGrid] = None,
    bubble_point_pressure_table: typing.Optional[ThreeDimensionalGrid] = None,
    gas_free_fvf_table: typing.Optional[TwoDimensionalGrid] = None,
    **kwargs: typing.Any,
) -> PVTData:
    """
    Build a `PVTData` for the water phase.

    All property tables are 3D with shape `(n_pressures, n_temperatures, n_salinities)`.

    :param pressures: 1D array of pressures (psi), strictly increasing.
    :param temperatures: 1D array of temperatures (°F), strictly increasing.
    :param salinities: 1D array of salinities (ppm NaCl), strictly increasing.
        If None, falls back to `[water_salinity]` or the library default.
    :param water_salinity: Single salinity scalar (ppm). Used only when
        `salinities` is None.
    :param gas_gravity: Gas specific gravity. Used for water compressibility liberation correction.
    :param gas: Gas identity. String name or `Fluid`. If a `Fluid` with
        a `pvt_table` is supplied, Bg is read from that table for water compressibility calculations.
    :param viscosity_table: Pre-computed water viscosity (n_p, n_t, n_s) in cP.
    :param density_table: Pre-computed water density (n_p, n_t, n_s) in lbm/ft³.
    :param formation_volume_factor_table: Pre-computed water formation volume factor (n_p, n_t, n_s) in bbl/STB.
    :param compressibility_table: Pre-computed water compressibility (n_p, n_t, n_s) in psi⁻¹.
    :param specific_gravity_table: Pre-computed water specific gravity (n_p, n_t, n_s).
    :param molecular_weight_table: Pre-computed water molecular weight (n_p, n_t, n_s).
    :param bubble_point_pressure_table: Pre-computed water bubble point pressure (n_p, n_t, n_s) in psi.
    :param gas_free_fvf_table: Pre-computed gas-free water formation volume factor (n_p, n_t) in bbl/STB.
    :return: `PVTData` with `phase=WATER`.
    """
    if pressures.ndim != 1 or not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be a strictly increasing 1D array.")
    if temperatures.ndim != 1 or not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be a strictly increasing 1D array.")

    dtype = get_dtype()
    n_p = len(pressures)
    n_t = len(temperatures)

    if salinities is None:
        salinity = (
            water_salinity
            if water_salinity is not None
            else c.DEFAULT_WATER_SALINITY_PPM
        )
        salinities = np.array([salinity], dtype=dtype)  # type: ignore[aassignment]

    _, gas_gravity, gas_pvt_table = _resolve_gas(gas, gas_gravity)
    assert gas_gravity is not None

    n_s = len(salinities)  # type: ignore[arg-type]

    pressure_grid, temperature_grid = np.meshgrid(
        pressures, temperatures, indexing="ij"
    )
    pressure_grid_3d, temperature_grid_3d, salinity_grid_3d = np.meshgrid(
        pressures,
        temperatures,
        salinities,  # type: ignore[arg-type]
        indexing="ij",
    )

    # Gas Bg (2D) for water compressibility liberation correction
    gas_fvf_grid: np.typing.NDArray
    if gas_pvt_table is not None:
        gas_tables = _tables_from_gas_pvt_table(
            gas_pvt_table, pressure_grid, temperature_grid
        )
        gas_fvf_grid_2d = gas_tables.get("formation_volume_factor_table")
        if gas_fvf_grid_2d is not None:
            gas_fvf_grid = np.broadcast_to(
                gas_fvf_grid_2d[:, :, np.newaxis], (n_p, n_t, n_s)
            ).copy()
        else:
            gas_fvf_grid = np.ones((n_p, n_t, n_s), dtype=dtype)
    else:
        z_factor_grid = build_gas_compressibility_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
        )
        gas_fvf_grid_2d = build_gas_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_compressibility_factor_grid=z_factor_grid,
        )
        gas_fvf_grid = np.broadcast_to(
            gas_fvf_grid_2d[:, :, np.newaxis], (n_p, n_t, n_s)
        ).copy()

    # Gas gravity 3D (for density calc)
    gas_gravity_3d = np.full((n_p, n_t, n_s), gas_gravity, dtype=dtype)

    # Gas-free water FVF (2D)
    if gas_free_fvf_table is None:
        gas_free_fvf_table = build_gas_free_water_formation_volume_factor_grid(  # type: ignore[assignment]
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
        )

    # Solubility & bubble point
    gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
        pressure_grid=pressure_grid_3d,
        temperature_grid=temperature_grid_3d,
        salinity_grid=salinity_grid_3d,
        gas=c.RESERVOIR_GAS,
    )

    if bubble_point_pressure_table is None:
        bubble_point_pressure_table = build_water_bubble_point_pressure_grid(  # type: ignore[assignment]
            temperature_grid=temperature_grid_3d,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            salinity_grid=salinity_grid_3d,
            gas=c.RESERVOIR_GAS,
        )

    # gas-free water formation volume factor broadcast to 3D for water compressibility/density calcs
    gas_free_fvf_3d = np.broadcast_to(
        gas_free_fvf_table[:, :, np.newaxis],  # type: ignore
        (n_p, n_t, n_s),
    ).copy()

    # Compressibility
    if compressibility_table is None:
        compressibility_table = build_water_compressibility_grid(  # type: ignore[assignment]
            pressure_grid=pressure_grid_3d,
            temperature_grid=temperature_grid_3d,
            bubble_point_pressure_grid=bubble_point_pressure_table,  # type: ignore[arg-type]
            gas_formation_volume_factor_grid=gas_fvf_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            gas_free_water_formation_volume_factor_grid=gas_free_fvf_3d,
            salinity=salinity_grid_3d,
        )

    if density_table is None:
        density_table = build_water_density_grid(  # type: ignore[assignment]
            pressure_grid=pressure_grid_3d,
            temperature_grid=temperature_grid_3d,
            gas_gravity_grid=gas_gravity_3d,
            salinity_grid=salinity_grid_3d,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            gas_free_water_formation_volume_factor_grid=gas_free_fvf_3d,
        )

    if formation_volume_factor_table is None:
        formation_volume_factor_table = build_water_formation_volume_factor_grid(  # type: ignore[assignment]
            water_density_grid=density_table,  # type: ignore[arg-type]
            salinity_grid=salinity_grid_3d,
        )

    if viscosity_table is None:
        viscosity_table = build_water_viscosity_grid(  # type: ignore[arg-type]
            temperature_grid=temperature_grid_3d,
            salinity_grid=salinity_grid_3d,
            pressure_grid=pressure_grid_3d,
        )

    # Specific gravity (water density relative to fresh water at SC: 62.4 lbm/ft³)
    if specific_gravity_table is None:
        specific_gravity_table = (
            density_table / c.STANDARD_WATER_DENSITY_IMPERIAL
        ).astype(dtype, copy=False)

    # Molecular weight - pure water ≈ 18.015, increases with salinity
    # Approximate: molecular weight ≈ 18.015 + salinity_ppm * 5.85e-5  (NaCl contribution)
    if molecular_weight_table is None:
        molecular_weight_table = (18.015 + salinity_grid_3d * 5.85e-5).astype(  # type: ignore[assignment]
            dtype, copy=False
        )

    return PVTData(
        phase=FluidPhase.WATER,
        pressures=pressures,
        temperatures=temperatures,
        salinities=salinities,
        viscosity_table=viscosity_table,
        density_table=density_table,
        formation_volume_factor_table=formation_volume_factor_table,
        compressibility_table=compressibility_table,
        specific_gravity_table=specific_gravity_table,
        molecular_weight_table=molecular_weight_table,
        bubble_point_pressure_table=bubble_point_pressure_table,
        gas_free_fvf_table=gas_free_fvf_table,
    )


def build_pvt_dataset(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    oil_specific_gravity: float = 0.85,
    gas_gravity: typing.Optional[float] = None,
    water_salinity: typing.Optional[float] = None,
    salinities: typing.Optional[OneDimensionalGrid] = None,
    estimated_solution_gor: typing.Optional[float] = None,
    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None,
    solution_gas_to_oil_ratios: typing.Optional[OneDimensionalGrid] = None,
    gas: typing.Optional[typing.Union[str, typing.Any]] = None,
    build_oil_data: bool = True,
    build_gas_data: bool = True,
    build_water_data: bool = True,
    **kwargs: typing.Any,
) -> PVTDataSet:
    """
    Constructs water, oil, and gas `PVTData` and returns `PVTDataSet`.

    Returns a 3-tuple `(water_data, oil_data, gas_data)`. Any element is
    `None` when the corresponding `build_*` flag is `False`.

    Keyword arguments prefixed with `oil_`, `gas_`, or `water_` are
    forwarded to the appropriate phase builder after stripping the prefix.
    All other `kwargs` are forwarded to all three builders.

    :param pressures: 1D array of pressures (psi).
    :param temperatures: 1D array of temperatures (°F).
    :param oil_specific_gravity: Oil specific gravity (water=1). Forwarded to oil builder.
    :param gas_gravity: Gas specific gravity (air=1). Forwarded to all builders.
    :param water_salinity: Single salinity scalar (ppm). Forwarded to water builder.
    :param salinities: 1D salinity array (ppm). Forwarded to water builder.
    :param estimated_solution_gor: Estimated solution gas-oil ratio (SCF/STB). Forwarded to oil builder.
    :param bubble_point_pressures: Pre-computed bubble point pressures. Forwarded to oil builder.
    :param solution_gas_to_oil_ratios: Solution gas-to-oil ratio axis for 2D bubble point pressures. Forwarded to oil builder.
    :param gas: Gas identity (str or `Fluid`). Forwarded to all builders.
    :param build_oil_data: If False, skip oil builder and return `None` for oil.
    :param build_gas_data: If False, skip gas builder and return `None` for gas.
    :param build_water_data: If False, skip water builder and return `None` for water.
    :param kwargs: Additional kwargs forwarded by prefix to phase builders.
    :return: A `PVTDataSet`.
    """
    # Partition kwargs by prefix
    oil_kwargs: typing.Dict[str, typing.Any] = {}
    gas_kwargs: typing.Dict[str, typing.Any] = {}
    water_kwargs: typing.Dict[str, typing.Any] = {}
    shared_kwargs: typing.Dict[str, typing.Any] = {}

    for key, val in kwargs.items():
        if key.startswith("oil_"):
            oil_kwargs[key[4:]] = val
        elif key.startswith("gas_"):
            gas_kwargs[key[4:]] = val
        elif key.startswith("water_"):
            water_kwargs[key[6:]] = val
        else:
            shared_kwargs[key] = val

    oil_data = None
    gas_data = None
    water_data = None

    if build_oil_data:
        oil_data = build_oil_pvt_data(
            pressures=pressures,
            temperatures=temperatures,
            oil_specific_gravity=oil_specific_gravity,
            gas_gravity=gas_gravity,
            estimated_solution_gor=estimated_solution_gor,
            bubble_point_pressures=bubble_point_pressures,
            solution_gas_to_oil_ratios=solution_gas_to_oil_ratios,
            gas=gas,
            **{**shared_kwargs, **oil_kwargs},
        )

    if build_gas_data:
        gas_data = build_gas_pvt_data(
            pressures=pressures,
            temperatures=temperatures,
            gas_gravity=gas_gravity,
            gas=gas,
            water_salinities=salinities,
            **{**shared_kwargs, **gas_kwargs},
        )

    if build_water_data:
        water_data = build_water_pvt_data(
            pressures=pressures,
            temperatures=temperatures,
            salinities=salinities,
            water_salinity=water_salinity,
            gas_gravity=gas_gravity,
            gas=gas,
            **{**shared_kwargs, **water_kwargs},
        )

    return PVTDataSet(oil=oil_data, water=water_data, gas=gas_data)
