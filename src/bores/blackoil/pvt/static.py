import attrs
from typing_extensions import Self

from bores.constants import UnitConversionTable, c, get_conversion_factors
from bores.correlations import scalars
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.types import Number, UnitSystem
from bores.utils import scale

__all__ = ["StaticPVT"]


@attrs.frozen(slots=True, kw_only=True)
class StaticPVT(Serializable):
    """
    Static/immutable fluid characterization. Pressure and temperature dependent
    properties are evaluated from the PVT tables or correlations during the simulation
    and may be cached separately for performance.

    Use `convert(target)` to rescale dimensional quantities to another unit system.
    """

    # Oil
    stock_tank_oil_density: Number | None = None
    """
    Stock-tank oil density at standard conditions.

    Units: lbm/ft³ (FIELD), kg/m³ (METRIC), g/cm³ (LAB).
    Read from the DENSITY keyword (column 1).
    Used in: ρo,res = (stock_tank_oil_density + Rs · stock_tank_gas_density / f) / Bo,
    where `f` is ft³/STB in FIELD units and 1 otherwise (see `get_stb_to_volume_factor`).
    """

    # Water
    water_reference_pressure: Number | None = None
    """
    Reference pressure at which `water_reference_fvf` and
    `water_reference_compressibility` are defined.

    This is the `PVTW` reference pressure (item 1).

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    """

    water_reference_fvf: Number | None = None
    """
    Water formation volume factor at `water_reference_pressure` (Bw_ref).

    Units: bbl/STB (FIELD), m³/sm³ (METRIC / SI), cc/scc (LAB).
    Approximately 1.00-1.08 depending on salinity and temperature.
    """

    water_reference_viscosity: Number | None = None
    """
    Water viscosity at reference conditions (μw_ref).

    Units: cP (FIELD / METRIC / LAB), Pa·s (SI).
    Approximately 0.3-1.0 cP at reservoir temperature.
    """

    water_reference_compressibility: Number | None = None
    """
    Water compressibility at `water_reference_pressure` (cw_ref).

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Typically 3-5 x 10⁻⁶ psi⁻¹.
    """

    stock_tank_water_density: Number | None = None
    """
    Stock-tank water density at standard conditions.

    Units: same as `stock_tank_oil_density`.
    Read from the DENSITY keyword (column 2).
    Used in: ρw,res = stock_tank_water_density / Bw
    """

    # Gas
    stock_tank_gas_density: Number | None = None
    """
    Stock-tank gas density at standard conditions.

    Units: same as `stock_tank_oil_density`.
    Read from the DENSITY keyword (column 3).
    Used in: ρg,res = (stock_tank_gas_density + Rv · stock_tank_oil_density · f) / Bg  [wet gas]
            ρg,res = stock_tank_gas_density / Bg                                   [dry gas]
    where `f` is ft³/STB in FIELD units and 1 otherwise (see `get_stb_to_volume_factor`).
    """

    water_viscosibility: Number | None = None
    """
    Water viscosibility - rate of change of water viscosity with pressure
    (d ln μw / dP), item 5 of the `PVTW` record.

    Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
    Zero for incompressible-viscosity water (the common default).
    """

    water_salinity: Number | None = None
    """
    Formation water salinity (ppm NaCl).

    Assumed spatially and temporally constant. Used in brine density and
    viscosity correlations (e.g. Batzle-Wang). Typical seawater: 35 000 ppm.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Unit-independent fields (e.g. `water_salinity`, in ppm) are unaffected by unit
    conversion.
    """

    @property
    def oil_specific_gravity(self) -> Number | None:
        """
        Oil specific gravity relative to fresh water at standard conditions (dimensionless).
        """
        if self.stock_tank_oil_density is None:
            return None

        # Derive oil specific gravity from stock-tank oil density relative to
        # fresh water at standard conditions in unit_system.
        if self.unit_system == UnitSystem.FIELD:
            reference_water_density = c.STANDARD_WATER_DENSITY_IMPERIAL
        elif self.unit_system == UnitSystem.METRIC:
            reference_water_density = c.STANDARD_WATER_DENSITY_METRIC
        elif self.unit_system == UnitSystem.SI:
            reference_water_density = c.STANDARD_WATER_DENSITY_SI
        else:  # UnitSystem.LAB
            reference_water_density = c.STANDARD_WATER_DENSITY_LAB
        return self.stock_tank_oil_density / reference_water_density

    @property
    def gas_gravity(self) -> Number | None:
        """
        Gas specific gravity relative to air (dimensionless).
        """
        if self.stock_tank_gas_density is None:
            return None
        # Derive gas gravity from stock-tank gas density relative to air.
        if self.unit_system == UnitSystem.FIELD:
            reference_air_density = c.STANDARD_AIR_DENSITY_IMPERIAL
        elif self.unit_system == UnitSystem.METRIC:
            reference_air_density = c.STANDARD_AIR_DENSITY_METRIC
        elif self.unit_system == UnitSystem.SI:
            reference_air_density = c.STANDARD_AIR_DENSITY_SI
        else:  # UnitSystem.LAB
            reference_air_density = c.STANDARD_AIR_DENSITY_LAB
        return self.stock_tank_gas_density / reference_air_density

    @property
    def oil_api_gravity(self) -> Number | None:
        """
        Oil API gravity (°API), computed as 141.5 / SG - 131.5.

        Provided for convenience; redundant with `oil_specific_gravity`.
        """
        oil_specific_gravity = self.oil_specific_gravity
        if oil_specific_gravity is None:
            return None
        return scalars.compute_oil_api_gravity(oil_specific_gravity)

    @property
    def gas_molecular_weight(self) -> Number | None:
        """
        Gas molecular weight (g/mol) computed from the gas gravity.
        """
        gas_gravity = self.gas_gravity
        if gas_gravity is None:
            return None
        return scalars.compute_gas_molecular_weight(gas_gravity)

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        pvtnum: int = 0,
        salinity: Number = 0.0,
    ) -> Self:
        """
        Load static PVT properties from a parsed `DeckFile` for a given region.

        Extracts `DENSITY` and `PVTW` records for the specified PVT region and
        constructs a `StaticPVT` instance with the stock-tank densities and
        water reference properties.

        :param deck_file: Parsed `DeckFile` containing `PROPS`-section keywords.
        :param pvtnum: 1-based PVT region index (matches Eclipse `PVTNUM`). Values below 1
            (including the default `0`) select the first region.
        :param salinity: Water salinity in ppm NaCl (default 0).
        :returns: New `StaticPVT` instance populated from deck data.
        :raises ValidationError: If required `PVTW` record is missing for the region.
        """
        # Convert 1-based pvtnum to 0-based index
        region_idx = max(pvtnum - 1, 0)

        # Extract `DENSITY` record for this region
        density_record: dict[str, Number] | None = None
        density_all = deck_file.get("DENSITY")
        if density_all is not None and region_idx < len(density_all):
            region_density_rows = density_all[region_idx]
            if region_density_rows:
                density_record = region_density_rows[0]

        # Extract PVTW record for this region
        pvtw_record: dict[str, Number] | None = None
        pvtw_all = deck_file.get("PVTW")
        if pvtw_all is not None and region_idx < len(pvtw_all):
            region_pvtw_rows = pvtw_all[region_idx]
            if region_pvtw_rows:
                pvtw_record = region_pvtw_rows[0]

        if pvtw_record is None:
            raise ValidationError(
                f"PVTW record not found for region {pvtnum}. "
                "PVTW is required to specify water reference properties."
            )

        # Extract stock-tank densities from DENSITY record
        stock_tank_oil_density: Number | None = None
        stock_tank_water_density: Number | None = None
        stock_tank_gas_density: Number | None = None
        if density_record is not None:
            stock_tank_oil_density = density_record.get("oil")
            stock_tank_water_density = density_record.get("water")
            stock_tank_gas_density = density_record.get("gas")

        # Extract water reference properties from PVTW record
        water_reference_pressure = pvtw_record.get("reference_pressure")
        water_reference_fvf = pvtw_record.get("fvf")
        water_reference_viscosity = pvtw_record.get("viscosity")
        water_reference_compressibility = pvtw_record.get("compressibility")
        water_viscosibility = pvtw_record.get("viscosibility", 0.0)
        return cls(
            stock_tank_oil_density=stock_tank_oil_density,
            water_reference_pressure=water_reference_pressure,
            water_reference_fvf=water_reference_fvf,
            water_reference_viscosity=water_reference_viscosity,
            water_reference_compressibility=water_reference_compressibility,
            stock_tank_water_density=stock_tank_water_density,
            stock_tank_gas_density=stock_tank_gas_density,
            water_viscosibility=water_viscosibility,
            water_salinity=salinity,
            unit_system=deck_file.unit_system,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `StaticPVT` with dimensional quantities rescaled to
        *target*.

        Unit-independent fields (e.g. `water_salinity`) are copied unchanged.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `StaticPVT` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return self.__class__(
            water_salinity=self.water_salinity,
            water_reference_pressure=scale(self.water_reference_pressure, factors["pressure"]),
            water_reference_fvf=scale(self.water_reference_fvf, factors["liquid_fvf"]),
            water_reference_viscosity=scale(self.water_reference_viscosity, factors["viscosity"]),
            water_reference_compressibility=scale(
                self.water_reference_compressibility, factors["compressibility"]
            ),
            water_viscosibility=scale(self.water_viscosibility, factors["compressibility"]),
            stock_tank_oil_density=scale(self.stock_tank_oil_density, factors["density"]),
            stock_tank_water_density=scale(self.stock_tank_water_density, factors["density"]),
            stock_tank_gas_density=scale(self.stock_tank_gas_density, factors["density"]),
            unit_system=target,
        )
