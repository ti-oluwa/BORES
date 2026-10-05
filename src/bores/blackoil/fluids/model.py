"""Black-oil fluid physics model."""

from typing_extensions import Self

from bores.blackoil.pvt.regions import PVT
from bores.blackoil.satfunc.regions import SatFunc
from bores.constants import UnitConversionTable, build_unit_conversion_table
from bores.errors import ValidationError
from bores.types import UnitSystem

__all__ = ["BlackOil"]


class BlackOil:
    """
    Black-oil fluid physics model/system.

    Holds the multi-region table objects that define the
    fluid physics for a black-oil simulation.

    All region tables are normalised to `unit_system` at construction.
    Use `convert(target)` to produce a fully rescaled copy.
    """

    __slots__ = ("pvt", "satfunc", "unit_system")

    def __init__(
        self,
        pvt: PVT,
        satfunc: SatFunc,
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        :param pvt: PVT region tables keyed by 1-based `PVTNUM` index.
        :param satfunc: Saturation function region tables keyed by
            1-based `SATNUM` index.
        :param unit_system: Target unit system for all region tables. When
            `None`, all supplied tables must share the same unit system -
            if they do not, a `ValidationError` is raised. When provided,
            each table group is converted to `unit_system` as needed.
        :raises ValidationError: If `unit_system` is `None` and the supplied
            tables do not all share the same unit system.
        """
        unit_conversion_table = build_unit_conversion_table()

        # Resolve unit systems of each supplied group
        pvt_unit_system = pvt.unit_system
        satfunc_unit_system = satfunc.unit_system

        if unit_system is None:
            # All present groups must agree
            systems = {pvt_unit_system, satfunc_unit_system}
            if len(systems) > 1:
                raise ValidationError(
                    "All region tables must share the same unit system when "
                    "`unit_system` is not explicitly provided. "
                    f"Found: {sorted(s.value for s in systems)}. "
                    "Pass `unit_system` explicitly to convert all tables to a "
                    "common system."
                )
            unit_system = pvt_unit_system

        # Convert each group to the target unit system
        if pvt_unit_system != unit_system:
            pvt = pvt.convert(unit_system, table=unit_conversion_table)
        if satfunc_unit_system != unit_system:
            satfunc = satfunc.convert(unit_system, table=unit_conversion_table)

        self.pvt = pvt
        """PVT region tables - one `PVTRegion` per `PVTNUM` region."""

        self.satfunc = satfunc
        """Saturation function region tables - one per `SATNUM` region."""

        self.unit_system = unit_system
        """Unit system in which all region tables are expressed."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `BlackOil` with all region tables rescaled to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `BlackOil` in *target* units.
        """
        if target == self.unit_system:
            return self

        return self.__class__(
            pvt=self.pvt.convert(target, table=table),
            satfunc=self.satfunc.convert(target, table=table),
            unit_system=target,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_pvt_regions={self.pvt.n_regions}, "
            f"n_saturation_regions={self.satfunc.n_regions}, "
            f"unit_system={self.unit_system.value!r}"  # type: ignore[union-attr]
            f")"
        )
