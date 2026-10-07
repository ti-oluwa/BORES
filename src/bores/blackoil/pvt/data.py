import logging
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, c, get_conversion_factors
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.stores import StoreSerializable
from bores.types import (
    FluidPhase,
    Number,
    NumberArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)
from bores.utils import scale, scale_and_offset

logger = logging.getLogger(__name__)

__all__ = ["PVTData", "PVTDataSet"]


def get_stb_to_volume_factor(unit_system: UnitSystem) -> Number:
    """
    Return the factor that reconciles gas-oil ratio units with density units.

    In FIELD units `Rs` and `Rsw` (SCF/STB) and `Rv` (STB/SCF) are per barrel of
    stock-tank liquid, while densities are per cubic foot. Mass balances such as
    `ρo = (ρo,SC + Rs·ρg,SC) / Bo` are therefore only dimensionally consistent once
    `1 STB = 5.614583 ft³` is applied to the dissolved-gas / vaporized-oil term. In
    METRIC, SI and LAB the ratios are volume/volume (Sm³/Sm³, scc/scc) and need none.

    - Oil / water: `ρ = (ρ,SC + Rs·ρg,SC / f) / B`
    - Wet gas: `ρg = (ρg,SC + Rv·ρo,SC · f) / Bg`

    :param unit_system: Unit system the ratios and densities are expressed in.
    :returns: `f` in ft³/STB for FIELD, `1.0` otherwise.
    """
    if unit_system == UnitSystem.FIELD:
        return c.STB_TO_CUBIC_FEET
    return 1.0


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
    - Wet gas (PVTG): the saturated tables are 2-D like any gas (values along the dew
      curve); the `undersaturated_*` tables are 3-D with shape
      `(n_pressures, n_temperatures, n_rv)` over `vaporized_oil_ratios`.

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
    `vaporized_oil_ratios`, `dew_point_pressures`,
    `undersaturated_formation_volume_factor_table`, `undersaturated_viscosity_table`.

    Water-specific fields: `salinities`, `bubble_point_pressure_table`,
    `gas_free_water_fvf_table`.
    """

    phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """Fluid phase this data describes."""

    # Coordinate grids
    pressures: NumberArray[OneDimension]
    """1-D array of pressures, strictly increasing. Units depend on `unit_system`."""

    temperatures: NumberArray[OneDimension]
    """1-D array of temperatures, strictly increasing. Units depend on `unit_system`."""

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

    # Gas-only: Rv axis, dew point and saturated Rv
    vaporized_oil_ratios: NumberArray[OneDimension] | None = None
    """
    1-D array of Rv values, strictly increasing: the third axis of the undersaturated wet-gas
    tables and the first axis of a 2-D `dew_point_pressures` table. Gas / condensate phase
    only. Units: STB/scf (FIELD), Sm³/Sm³ (METRIC/SI), scc/scc (LAB).
    """

    dew_point_pressures: NumberArray[OneDimension] | NumberArray[TwoDimensions] | None = None
    """
    Dew-point pressures. Gas / condensate phase only. Units depend on `unit_system`.

    - 1-D shape `(n_t,)`      -> Pdew(T).
    - 2-D shape `(n_rv, n_t)` -> Pdew(Rv, T); requires `vaporized_oil_ratios`.
    """

    vaporized_oil_ratio_table: NumberArray[TwoDimensions] | None = None
    """
    Saturated vaporised oil ratio Rv_sat(P, T): the Rv of gas on the dew curve at `(P, T)`.
    Gas / condensate phase only. Shape `(n_p, n_t)`.
    Units: STB/scf (FIELD), Sm³/Sm³ (METRIC/SI), scc/scc (LAB).
    Gas whose Rv is below Rv_sat at the same pressure is undersaturated: its Rv is a state
    variable held by the caller and looked up in the `undersaturated_*` tables.
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

    In FIELD units the `Rs` term is divided, and the `Rv` term multiplied, by
    5.614583 ft³/STB (see `get_stb_to_volume_factor`) so that every term is in
    lbm per cubic foot.

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
    """Solution GOR Rs(P, T). Oil phase only. Units: SCF/STB (FIELD), Sm³/Sm³ (METRIC/SI), scc/scc (LAB)."""

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

    # Undersaturated wet-gas tables (3-D over the Rv axis)
    undersaturated_formation_volume_factor_table: NumberArray[ThreeDimensions] | None = None
    """
    FVF of undersaturated gas B(P, T, Rv), shape `(n_p, n_t, n_rv)`; requires
    `vaporized_oil_ratios`. Gas / condensate phase only. Units as `formation_volume_factor_table`
    (ft³/SCF in FIELD). The 2-D `formation_volume_factor_table` is the FVF of saturated gas
    (on the dew curve).
    """

    undersaturated_viscosity_table: NumberArray[ThreeDimensions] | None = None
    """
    Viscosity of undersaturated gas μ(P, T, Rv), shape `(n_p, n_t, n_rv)`; requires
    `vaporized_oil_ratios`. Gas / condensate phase only. Units as `viscosity_table`.
    The 2-D `viscosity_table` is the viscosity of saturated gas (on the dew curve).
    """

    dtype: npt.DTypeLike = None
    """Floating-point dtype of all arrays. Defaults to the active `BORES` precision."""

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
        self._check_phase_fields()
        self.ensure_dtype(self.dtype, force=True)

    def _check_phase_fields(self) -> None:
        """Warn about fields that do not apply to the phase, and require the Rs axis of a 2-D Pb table."""
        phase = typing.cast(FluidPhase, self.phase)
        # `stacklevel=4`: this method -> `__attrs_post_init__` -> attrs' `__init__` -> the caller
        if phase == FluidPhase.GAS and self.solution_gor_table is not None:
            warnings.warn(
                f"{type(self).__name__}: `solution_gor_table` is oil-only and will "
                "be ignored for GAS phase.",
                UserWarning,
                stacklevel=4,
            )
        if phase == FluidPhase.OIL and self.compressibility_factor_table is not None:
            warnings.warn(
                f"{type(self).__name__}: `compressibility_factor_table` is gas-only "
                "and will be ignored for OIL phase.",
                UserWarning,
                stacklevel=4,
            )
        if phase == FluidPhase.WATER and self.bubble_point_pressures is not None:
            warnings.warn(
                f"{type(self).__name__}: `bubble_point_pressures` is oil-only. For "
                "water bubble point use `bubble_point_pressure_table` (3-D).",
                UserWarning,
                stacklevel=4,
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
        if (
            self.dew_point_pressures is not None
            and isinstance(self.dew_point_pressures, np.ndarray)
            and self.dew_point_pressures.ndim == 2
            and self.vaporized_oil_ratios is None
        ):
            raise ValidationError(
                f"{type(self).__name__}: 2-D `dew_point_pressures` requires "
                "`vaporized_oil_ratios` to be provided."
            )

    def ensure_dtype(self, dtype: npt.DTypeLike = None, force: bool = True) -> None:
        """
        Cast every array field to *dtype*, in place (the instance is frozen, so this
        bypasses attrs).

        :param dtype: Target dtype. Defaults to the active `BORES` precision.
        :param force: If `False`, do nothing when the data is already stored as *dtype*.
        """
        if not force and self.dtype is not None and self.dtype == np.dtype(dtype):
            return

        dtype = np.dtype(dtype if dtype is not None else get_dtype())
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)
            if value is not None and isinstance(value, np.ndarray) and value.dtype != dtype:
                object.__setattr__(self, field.name, value.astype(dtype, copy=False))

        if self.dtype != dtype:
            object.__setattr__(self, "dtype", dtype)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVTData` with all dimensional quantities rescaled to *target*.

        Pressure and temperature axes, bubble/dew point pressures, densities, FVFs,
        viscosities, compressibilities, and the gas-oil / oil-gas ratios (Rs, Rv, Rsw)
        are rescaled using `get_conversion_factors`. Temperatures use the affine map
        (`T * scale + offset`). Dimensionless quantities (compressibility factor) and
        salinities (ppm) are copied unchanged.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
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
        gas_oil_ratio_factor = factors["gas_oil_ratio"]
        oil_gas_ratio_factor = factors["oil_gas_ratio"]
        fvf_factor = gas_fvf_factor if self.phase == FluidPhase.GAS else liquid_fvf_factor
        # Compressibility is 1/pressure
        compressibility_factor = 1.0 / pressure_factor
        return attrs.evolve(
            self,
            pressures=scale(self.pressures, pressure_factor),
            temperatures=scale_and_offset(
                self.temperatures, factors["temperature"], factors["temperature_offset"]
            ),
            solution_gas_to_oil_ratios=scale(
                self.solution_gas_to_oil_ratios, gas_oil_ratio_factor
            ),
            vaporized_oil_ratios=scale(self.vaporized_oil_ratios, oil_gas_ratio_factor),
            bubble_point_pressures=scale(self.bubble_point_pressures, pressure_factor),
            dew_point_pressures=scale(self.dew_point_pressures, pressure_factor),
            vaporized_oil_ratio_table=scale(self.vaporized_oil_ratio_table, oil_gas_ratio_factor),
            formation_volume_factor_table=scale(self.formation_volume_factor_table, fvf_factor),
            viscosity_table=scale(self.viscosity_table, viscosity_factor),
            undersaturated_formation_volume_factor_table=scale(
                self.undersaturated_formation_volume_factor_table, fvf_factor
            ),
            undersaturated_viscosity_table=scale(
                self.undersaturated_viscosity_table, viscosity_factor
            ),
            density_table=scale(self.density_table, density_factor),
            compressibility_table=scale(self.compressibility_table, compressibility_factor),
            solution_gor_table=scale(self.solution_gor_table, gas_oil_ratio_factor),
            solubility_in_water_table=scale(self.solubility_in_water_table, gas_oil_ratio_factor),
            bubble_point_pressure_table=scale(self.bubble_point_pressure_table, pressure_factor),
            gas_free_water_fvf_table=scale(self.gas_free_water_fvf_table, liquid_fvf_factor),
            unit_system=target,
        )


@attrs.frozen(slots=True)
class PVTDataSet(StoreSerializable):
    """
    Bundle of raw `PVTData` for all three fluid phases.

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
        Return a new `PVTDataSet` with all pvt data converted to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `PVTDataSet` in *target* units.
        """
        return self.__class__(
            oil=self.oil.convert(target, table=table) if self.oil is not None else None,
            gas=self.gas.convert(target, table=table) if self.gas is not None else None,
            water=self.water.convert(target, table=table) if self.water is not None else None,
        )
