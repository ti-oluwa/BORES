import logging
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.stores import StoreSerializable
from bores.types import (
    FluidPhase,
    NumberArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)
from bores.utils import scale

logger = logging.getLogger(__name__)

__all__ = ["PVTData", "PVTDataSet"]


@attrs.frozen(slots=True)
class PVTData(StoreSerializable):
    """
    Raw PVT table data for a single fluid phase.

    Phase-tagged container for tabulated fluid properties. All table arrays
    are optional. Only `phase`, `pressures`, and `temperatures` are
    required. `PVTTable` validates which fields are meaningful for the
    phase at initialisation time.

    **Array shapes**

    - Oil / Gas: 2-D arrays with shape `(n_pressures, n_temperatures)`.
    - Water: 3-D arrays with shape `(n_pressures, n_temperatures, n_salinities)`.
    - For wet-gas (PVTG) tables: `(n_pressures, n_rv)` where the Rv axis
      replaces the temperature axis (`temperatures` carries the Rv values in
      that case).

    **Primary (interpolated) properties**

    These are the only quantities read directly from Eclipse deck keywords:

    - Oil: `formation_volume_factor_table` (Bo), `viscosity_table` (μo),
      `solution_gor_table` (Rs).
    - Gas: `formation_volume_factor_table` (Bg), `viscosity_table` (μg),
      `compressibility_factor_table` (z), `vaporized_oil_ratio_table` (Rv,
      wet-gas / condensate only).
    - Water: scalars on `PVT`. No table needed.

    **Derived (pre-built) properties**

    These are built from primary quantities plus stock-tank reference densities
    at `PVTTable` construction time. They are stored here as optional arrays
    so they can be serialised and reloaded without rebuilding:

    - `density_table` - ρ(P, T).
    - `compressibility_table` - c(P, T).

    Oil-specific fields: `bubble_point_pressures`, `solution_gas_to_oil_ratios`,
    `solution_gor_table`.

    Gas-specific fields: `compressibility_factor_table`,
    `solubility_in_water_table`, `vaporized_oil_ratio_table`,
    `dew_point_pressures`.

    Water-specific fields: `salinities`, `bubble_point_pressure_table`,
    `gas_free_water_fvf_table`.
    """

    phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """Fluid phase this data describes."""

    # Coordinate grids
    pressures: NumberArray[OneDimension]
    """1-D array of pressures, strictly increasing. Units depend on `unit_system`."""

    temperatures: NumberArray[OneDimension]
    """
    1-D array of temperatures, strictly increasing. Units depend on `unit_system`.

    For wet-gas (PVTG) tables this axis carries Rv values instead of
    temperatures; the `pvtg` flag on `PVTTable` signals this.
    """

    # Water-only coordinate
    salinities: NumberArray[OneDimension] | None = None
    """1-D array of salinities (ppm NaCl), strictly increasing. Water phase only. Unit-system independent."""

    # Oil-only coordinates / meta
    bubble_point_pressures: NumberArray[OneDimension] | NumberArray[TwoDimensions] | None = None
    """
    Bubble-point pressures. Oil phase only. Units depend on `unit_system`.

    - 1-D shape `(n_t,)`      -> Pb(T).
    - 2-D shape `(n_rs, n_t)` -> Pb(Rs, T); requires `solution_gas_to_oil_ratios`.
    """

    solution_gas_to_oil_ratios: NumberArray[OneDimension] | None = None
    """
    1-D array of Rs values for the first axis of a 2-D
    `bubble_point_pressures` table. Required when `bubble_point_pressures`
    is 2-D. Oil phase only. Units depend on `unit_system`.
    """

    # Gas-only: dew point and Rv
    dew_point_pressures: NumberArray[OneDimension] | None = None
    """
    Dew-point pressures Pdew(T). Gas / condensate phase only.
    Shape `(n_t,)`. Units depend on `unit_system`.
    """

    vaporized_oil_ratio_table: NumberArray[TwoDimensions] | None = None
    """
    Vaporised oil ratio Rv(P, T). Gas / condensate phase only. Shape `(n_p, n_t)`.
    Units: STB/scf (FIELD), Sm³/Sm³ (METRIC/SI), scc/scc (LAB). Dimensionless ratios are unit-system independent.
    Rv is capped at Rv_sat above dew point (analogous to Rs being capped at Rsb above bubble point for oil).
    """

    # Shared primary tables (2-D for oil/gas; 3-D for water)
    viscosity_table: NumberArray[TwoDimensions] | NumberArray[ThreeDimensions] | None = None
    """Viscosity μ(P, T). Units depend on `unit_system` (cP in FIELD/METRIC/LAB, Pa·s in SI). 2-D for oil/gas, 3-D for water."""

    formation_volume_factor_table: (
        NumberArray[TwoDimensions] | NumberArray[ThreeDimensions] | None
    ) = None
    """
    Formation volume factor B(P, T). 2-D for oil/gas, 3-D for water.

    Units depend on `unit_system` and phase:
    - Oil/water: bbl/STB (FIELD), m³/Sm³ (METRIC/SI), cc/scc (LAB)
    - Gas: ft³/SCF (FIELD), m³/Sm³ (METRIC/SI), cc/scc (LAB)
    """

    # Shared derived tables (optional; built at `PVTTable` construction when absent)
    density_table: NumberArray[TwoDimensions] | NumberArray[ThreeDimensions] | None = None
    """
    Density ρ(P, T). 2-D for oil/gas, 3-D for water.

    Units depend on `unit_system` (lbm/ft³ in FIELD, kg/m³ in METRIC/SI, g/cm³ in LAB).

    Derived from FVF and stock-tank reference densities:

    - Oil:  ρo = (ρo,SC + Rs·ρg,SC) / Bo
    - Gas:  ρg = (ρg,SC + Rv·ρo,SC) / Bg   [wet gas]
            ρg = ρg,SC / Bg                  [dry gas]
    - Water: ρw = ρw,SC / Bw

    Set automatically by `PVTTable` if absent and reference densities are
    provided.
    """

    compressibility_table: NumberArray[TwoDimensions] | NumberArray[ThreeDimensions] | None = None
    """
    Compressibility c(P, T). 2-D for oil/gas, 3-D for water.

    Units depend on `unit_system` (1/psi in FIELD, 1/bar in METRIC, 1/atm in LAB, 1/Pa in SI).

    Derived from the pressure-derivative of FVF:

    - Oil / water: c = -(1/B) · (∂B/∂P)
    - Gas:         cg = 1/P - (1/z) · (∂z/∂P)

    Set automatically by `PVTTable` if absent and the FVF table is present.
    """

    # Oil-only primary
    solution_gor_table: NumberArray[TwoDimensions] | None = None
    """Solution GOR Rs(P, T). Oil phase only. Units: SCF/STB (FIELD), Sm³/Sm³ (METRIC/SI), scc/scc (LAB). Dimensionless ratios are unit-system independent."""

    # Gas-only primary
    compressibility_factor_table: NumberArray[TwoDimensions] | None = None
    """Z-factor z(P, T), dimensionless. Gas phase only."""

    solubility_in_water_table: NumberArray[ThreeDimensions] | None = None
    """
    Gas solubility in water Rsw(P, T, S). Units depend on `unit_system`.
    Gas phase only. 3-D shape `(n_p, n_t, n_s)`; requires `salinities`.
    """

    # Water-only primary
    bubble_point_pressure_table: NumberArray[ThreeDimensions] | None = None
    """Water bubble-point pressure Pbw(P, T, S). Water phase only. Units depend on `unit_system`."""

    gas_free_water_fvf_table: NumberArray[TwoDimensions] | None = None
    """
    Gas-free water FVF Bw_gf(P, T). Water phase only.

    Units depend on `unit_system` (bbl/STB in FIELD, m³/Sm³ in METRIC/SI, cc/scc in LAB).

    Used internally to compute `density_table` and `compressibility_table`
    for the water phase; not exposed as a direct query method on `PVTTable`.
    """
    dtype: npt.DTypeLike = None

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Unit system in which all dimensional quantities in this data are expressed.

    Determines units for all dimensional fields (pressure, temperature, density, viscosity, etc.):
    - FIELD: psi, °F, lbm/ft³, cP, etc.
    - METRIC: bar, °C, kg/m³, cP, etc.
    - LAB: atm, °C, g/cm³, cP, etc.
    - SI: Pa, K, kg/m³, Pa·s, etc.
    """

    def __attrs_post_init__(self) -> None:
        self._warn_phase_mismatches()
        self.ensure_dtype(self.dtype, force=True)

    def _warn_phase_mismatches(self) -> None:
        phase = typing.cast(FluidPhase, self.phase)
        if phase == FluidPhase.GAS and self.solution_gor_table is not None:
            warnings.warn(
                f"{type(self).__name__}: `solution_gor_table` is oil-only and will "
                "be ignored for GAS phase.",
                UserWarning,
                stacklevel=3,
            )
        if phase == FluidPhase.OIL and self.compressibility_factor_table is not None:
            warnings.warn(
                f"{type(self).__name__}: `compressibility_factor_table` is gas-only "
                "and will be ignored for OIL phase.",
                UserWarning,
                stacklevel=3,
            )
        if phase == FluidPhase.WATER and self.bubble_point_pressures is not None:
            warnings.warn(
                f"{type(self).__name__}: `bubble_point_pressures` is oil-only. For "
                "water bubble point use `bubble_point_pressure_table` (3-D).",
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
                f"{type(self).__name__}: 2-D `bubble_point_pressures` requires "
                "`solution_gas_to_oil_ratios` to be provided."
            )

    def ensure_dtype(self, dtype: npt.DTypeLike = None, force: bool = True) -> None:
        if not force and self.dtype is not None and self.dtype == np.dtype(dtype):
            return

        dtype = np.dtype(dtype if dtype is not None else get_dtype())
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)
            if value is not None and isinstance(value, np.ndarray) and value.dtype != dtype:
                object.__setattr__(self, field.name, value.astype(dtype, copy=False))

        if self.dtype != dtype:
            object.__setattr__(self, "dtype", dtype if dtype is not None else None)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVTData` with all dimensional quantities rescaled to *target*.

        Dimensionless properties (specific gravity, compressibility factor,
        vaporized oil ratio, solution GOR ratio) and multiplier-type quantities
        are copied unchanged. Pressure axes, densities, FVFs, and viscosities
        are rescaled using appropriate factors.

        :param target: Target `UnitSystem`.
        :returns `PVTData`: New `PVTData` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        density_factor = factors["density"]
        viscosity_factor = factors["viscosity"]
        liquid_fvf_factor = factors["liquid_fvf"]
        gas_fvf_factor = factors["gas_fvf"]
        fvf_factor = gas_fvf_factor if self.phase == FluidPhase.GAS else liquid_fvf_factor
        # Compressibility is 1/pressure
        compressibility_factor = 1.0 / pressure_factor
        return attrs.evolve(
            self,
            pressures=scale(self.pressures, pressure_factor),
            bubble_point_pressures=scale(self.bubble_point_pressures, pressure_factor),
            dew_point_pressures=scale(self.dew_point_pressures, pressure_factor),
            formation_volume_factor_table=scale(self.formation_volume_factor_table, fvf_factor),
            viscosity_table=scale(self.viscosity_table, viscosity_factor),
            density_table=scale(self.density_table, density_factor),
            compressibility_table=scale(self.compressibility_table, compressibility_factor),
            gas_free_water_fvf_table=scale(self.gas_free_water_fvf_table, liquid_fvf_factor),
            # Others are dimensionless, so no scaling.
            unit_system=target,
        )


@attrs.frozen(slots=True)
class PVTDataSet(StoreSerializable):
    """
    Serialisable bundle of raw `PVTData` for all three fluid phases.

    Stores the raw tabulated data for oil, gas, and water independently of
    any interpolation settings. Use it to persist PVT data and rebuild
    `PVTTables` later with different interpolation options.

    Typical workflow:

    ```python
    # Build and persist raw data
    dataset = PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)
    dataset.save("run/pvt.h5")

    # Reload and build tables with a specific interpolation config
    dataset = PVTDataSet.load("run/pvt.h5")
    tables = PVTTables.from_dataset(dataset, interpolation_method="cubic")
    ```
    """

    oil: PVTData | None = None
    """Raw PVT data for the oil phase."""

    gas: PVTData | None = None
    """Raw PVT data for the gas phase."""

    water: PVTData | None = None
    """Raw PVT data for the water phase."""

    unit_system: UnitSystem = attrs.field(init=False, repr=False)
    """Unit system in which all PVTData are expressed."""

    def __attrs_post_init__(self) -> None:
        # Check that not all phases are None
        if self.oil is None and self.gas is None and self.water is None:
            raise ValidationError(
                f"{type(self).__name__}: At least one of oil, gas, or water must be provided."
            )

        # Check that the phase field of each `PVTData` matches the attribute name
        for phase, data in (
            (FluidPhase.OIL, self.oil),
            (FluidPhase.GAS, self.gas),
            (FluidPhase.WATER, self.water),
        ):
            if data is not None and data.phase != phase:
                raise ValidationError(
                    f"{type(self).__name__}: {phase.value!r} `PVTData` has phase={data.phase}"
                )

        # Check that all present `PVTData` have the same unit system
        unit_systems = {
            data.unit_system for data in (self.oil, self.gas, self.water) if data is not None
        }
        if len(unit_systems) > 1:
            raise ValidationError(
                f"{type(self).__name__}: All `PVTData` must have the same unit system. Found: {unit_systems}"
            )

        object.__setattr__(self, "unit_system", unit_systems.pop())

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVTDataset` with all pvt data converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTDataset` in *target* units.
        """
        return self.__class__(
            oil=self.oil.convert(target, table=table) if self.oil is not None else None,
            gas=self.gas.convert(target, table=table) if self.gas is not None else None,
            water=self.water.convert(target, table=table) if self.water is not None else None,
        )
