import contextvars
import typing
import warnings
from uuid import uuid4

from bores.constants import Constants, c
from bores.errors import ValidationError
from bores.types import UnitConversionFactors, UnitConversionTable, UnitSystem

__all__ = [
    "UNIT_SYSTEM",
    "UnitSystemContext",
    "build_unit_conversion_table",
    "get_unit_system",
    "set_unit_system",
    "u",
]


DEFAULT_UNIT_SYSTEM = UnitSystem.FIELD
DEFAULT_CONTEXT_ID = uuid4().hex
_unit_system_context: contextvars.ContextVar[tuple[UnitSystem, str]] = contextvars.ContextVar(
    "unit_system_context", default=(DEFAULT_UNIT_SYSTEM, DEFAULT_CONTEXT_ID)
)


class UnitSystemContext:
    """
    Context manager for temporary process-local unit-system overrides.

    The previous unit system is restored when the context exits, including
    when the context body raises an exception. Contexts can be nested.

    Example:

    ```python
    from bores.types import UnitSystem
    from bores.units import UnitSystemContext, get_unit_system

    with UnitSystemContext(UnitSystem.SI):
        assert get_unit_system() is UnitSystem.SI

    assert get_unit_system() is UnitSystem.FIELD
    ```
    """

    __slots__ = ("_entry_depth", "_id", "_token", "_unit_system")

    def __init__(self, unit_system: UnitSystem) -> None:
        """
        Initialize a temporary unit-system override.

        :param unit_system: Unit system to use within the context.
        """
        self._unit_system = UnitSystem(unit_system)
        self._entry_depth = 0
        self._id = uuid4().hex
        self._token: contextvars.Token[tuple[UnitSystem, str]] | None = None

    @property
    def id(self) -> str:
        """The context ID."""
        return self._id

    @property
    def unit_system(self) -> UnitSystem:
        """The unit system configured for this context."""
        return self._unit_system

    def __enter__(self) -> UnitSystem:
        """Enter the context and activate its unit system."""
        current_context_id = _unit_system_context.get()[1]
        if current_context_id == self._id:
            warnings.warn(
                f"Unit system context {current_context_id!r} is already active; re-entering it is unnecessary.",
                UserWarning,
                stacklevel=2,
            )
            self._entry_depth += 1
            return self._unit_system

        if self._entry_depth:
            raise RuntimeError(
                f"Unit system context {current_context_id!r} is already active in another context."
            )

        self._token = _unit_system_context.set((self._unit_system, self._id))
        self._entry_depth = 1
        return self._unit_system

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Exit the context and restore the previous unit system."""
        if self._entry_depth == 0 or _unit_system_context.get()[1] != self._id:
            return

        self._entry_depth -= 1
        if self._entry_depth == 0 and self._token is not None:
            _unit_system_context.reset(self._token)
            self._token = None


@typing.final
class __UnitSystemProxy:
    """Proxy exposing the unit system active in the current context."""

    @property
    def value(self) -> UnitSystem:
        """
        Get the current context's `UnitSystem` instance.

        :return: Current `UnitSystem` instance
        """
        return get_unit_system()

    @property
    def context_id(self) -> str:
        """
        Get the current context's ID.

        :return: `UnitSystemContext` ID.
        """
        return _unit_system_context.get()[1]

    def in_default_context(self) -> bool:
        """Returns `True` if we are the default (process local) `UnitSystem` context"""
        return self.context_id == DEFAULT_CONTEXT_ID

    def __str__(self) -> str:
        return self.value.value

    def __repr__(self) -> str:
        return repr(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self.value, name)

    def __dir__(self):
        default = super().__dir__()
        return sorted({*default, *self.value.__dir__()})


UNIT_SYSTEM = u = __UnitSystemProxy()
"""Global proxy to access the active unit system."""


def get_unit_system() -> UnitSystem:
    """Return the unit system active in the current context."""
    return _unit_system_context.get()[0]


def set_unit_system(unit_system: UnitSystem, /) -> None:
    """
    Set the process-local default unit system.

    This function must only be called outside a `UnitSystemContext`. Use the
    context manager for temporary unit-system overrides.

    :raises ValidationError: If called inside a `UnitSystemContext`.
    """
    if u.context_id != DEFAULT_CONTEXT_ID:
        raise ValidationError(
            "Cannot set unit system. Are you in a `UnitSystemContext`? "
            "Only call in default context"
        )
    _unit_system_context.set((UnitSystem(unit_system), uuid4().hex))


def build_unit_conversion_table(constants: Constants | None = None) -> UnitConversionTable:
    """
    Build a complete unit conversion table from the provided or default
    constants registry.

    All numeric values are read from `constants` (or the global `c`
    proxy) so they stay in sync with any application-level overrides made
    via `ConstantsContext`.

    Each entry converts every dimensional quantity from the source
    `UnitSystem` to the target `UnitSystem`. The twelve source -> target
    pairs cover all ordered combinations of FIELD, METRIC, LAB, and SI
    (same-system pairs are handled by `IDENTITY_FACTORS` in
    `get_conversion_factors`).

    Intermediate factors are derived algebraically from the primitives
    stored in the constants registry so no magic numbers are hard-coded
    here.
    """
    con = constants if constants is not None else c

    # Primitive conversion factors from the constants registry
    psi_to_pa: float = con.PSI_TO_PASCAL  # 6894.757 Pa/psi
    psi_to_bar: float = con.PSI_TO_BAR  # 0.0689476 bar/psi
    atm_to_pa: float = con.ATM_TO_PASCAL  # 101 325.0 Pa/atm
    ft_to_m: float = con.FEET_TO_METERS  # 0.3048 m/ft
    m_to_ft: float = con.METERS_TO_FEET  # 3.28084 ft/m
    lbm_ft3_to_kg_m3: float = con.POUNDS_PER_CUBIC_FEET_TO_KILOGRAM_PER_CUBIC_METER
    lbm_ft3_to_g_cm3: float = con.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER
    cp_to_pas: float = con.CENTIPOISE_TO_PASCAL_SECONDS  # 0.001
    md_to_m2: float = con.MILLIDARCY_TO_SQUARE_METER  # 9.869233e-16
    scf_stb_to_sm3_sm3: float = con.SCF_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER
    stb_to_m3: float = con.STB_TO_CUBIC_METER  # 0.158987
    scf_to_m3: float = con.SCF_TO_SCM  # 0.0283168
    seconds_per_day: float = con.SECONDS_PER_DAY  # 86400.0
    hours_per_day: float = con.HOURS_PER_DAY  # 24.0

    # Derived intermediates (no magic numbers beyond what is above)
    cm_to_m: float = 0.01
    m_to_cm: float = 100.0
    ft_to_cm: float = ft_to_m * m_to_cm
    cm_to_ft: float = cm_to_m * m_to_ft
    kg_m3_to_g_cm3: float = cm_to_m**3  # 1e-6 / 1e-3 = 1e-3

    bar_to_pa: float = psi_to_pa / (psi_to_pa / atm_to_pa * (1.0 / psi_to_bar) * psi_to_bar)
    # Simpler: bar_to_pa = 1e5; but derive from constants to stay consistent
    # 1 bar = 14.5038 psi; bar_to_pa = 14.5038 * psi_to_pa / 14.5038... just:
    bar_to_pa = 1.0 / psi_to_bar * psi_to_pa  # 100 000 Pa/bar
    psi_to_atm: float = psi_to_pa / atm_to_pa
    bar_to_atm: float = bar_to_pa / atm_to_pa

    # Volume (reservoir)
    ft3_to_m3: float = ft_to_m**3
    m3_to_ft3: float = m_to_ft**3
    ft3_to_cm3: float = ft_to_cm**3
    cm3_to_ft3: float = cm_to_ft**3
    m3_to_cm3: float = m_to_cm**3
    cm3_to_m3: float = cm_to_m**3

    # Time
    seconds_per_hour: float = seconds_per_day / hours_per_day  # 3600.0
    days_per_second: float = 1.0 / seconds_per_day

    # Surface volumes
    # STB -> m³: stb_to_m3
    # STB -> cm³:
    stb_to_cm3: float = stb_to_m3 * m3_to_cm3
    # SCF -> m³: scf_to_m3
    # SCF -> cm³:
    scf_to_cm3: float = scf_to_m3 * m3_to_cm3
    # Sm³ -> scc:
    sm3_to_scc: float = m3_to_cm3
    # scc -> Sm³:
    scc_to_sm3: float = cm3_to_m3

    # GOR: SCF/STB -> Sm³/Sm³
    # = (scf_to_m3) / (stb_to_m3)  -- same as scf_stb_to_sm3_sm3
    gor_field_to_metric: float = scf_stb_to_sm3_sm3
    # GOR: Sm³/Sm³ -> SCF/STB
    gor_metric_to_field: float = 1.0 / scf_stb_to_sm3_sm3
    # GOR: SCF/STB -> scc/scc  (scf->scc / stb->scc)
    gor_field_to_lab: float = scf_to_cm3 / stb_to_cm3
    # GOR: scc/scc -> SCF/STB
    gor_lab_to_field: float = 1.0 / gor_field_to_lab
    # GOR: Sm³/Sm³ -> scc/scc  (both dimensionless, same ratio - 1.0)
    # Sm³/Sm³ and scc/scc are both volume/volume in their respective systems;
    # the numerical value of the ratio is unchanged.
    gor_metric_to_lab: float = 1.0
    gor_lab_to_metric: float = 1.0

    # OGR (Rv): STB/SCF -> Sm³/Sm³
    ogr_field_to_metric: float = stb_to_m3 / scf_to_m3
    ogr_metric_to_field: float = 1.0 / ogr_field_to_metric
    ogr_field_to_lab: float = stb_to_cm3 / scf_to_cm3
    ogr_lab_to_field: float = 1.0 / ogr_field_to_lab
    ogr_metric_to_lab: float = 1.0
    ogr_lab_to_metric: float = 1.0

    # FVF
    # liquid FVF: rb/STB -> rm³/Sm³
    # rb = reservoir barrel = ft³/5.614583 ... but FVF is dimensionless ratio
    # rb/STB and rm³/Sm³ are both (reservoir vol)/(surface vol); what changes
    # is the unit of each. rb/STB = 5.614583 ft³ / (5.614583 ft³) = 1 numerically
    # if reservoir and surface are same fluid. The actual conversion factor
    # between rb/STB and rm³/Sm³ is:
    #   (rb -> rm³) / (STB -> Sm³) = (stb_to_m3) / (stb_to_m3) = 1.0
    # Similarly rcf/SCF -> rm³/Sm³ = (ft3_to_m3) / (scf_to_m3)
    liquid_fvf_field_to_metric: float = 1.0  # rb/STB -> rm³/Sm³
    liquid_fvf_field_to_lab: float = 1.0  # rb/STB -> rcc/scc
    liquid_fvf_field_to_si: float = 1.0  # rb/STB -> rm³/Sm³
    gas_fvf_field_to_metric: float = ft3_to_m3 / scf_to_m3  # rcf/SCF -> rm³/Sm³
    gas_fvf_field_to_lab: float = ft3_to_cm3 / scf_to_cm3  # rcf/SCF -> rcc/scc
    gas_fvf_metric_to_field: float = 1.0 / gas_fvf_field_to_metric
    gas_fvf_lab_to_field: float = 1.0 / gas_fvf_field_to_lab

    # Surface liquid rates: STB/day -> Sm³/day, scc/hr, Sm³/s
    liquid_rate_field_to_metric: float = stb_to_m3  # STB/day -> Sm³/day
    liquid_rate_field_to_lab: float = stb_to_cm3 / hours_per_day  # STB/day -> scc/hr
    liquid_rate_field_to_si: float = stb_to_m3 / seconds_per_day  # STB/day -> Sm³/s
    liquid_rate_metric_to_field: float = 1.0 / liquid_rate_field_to_metric
    liquid_rate_metric_to_lab: float = m3_to_cm3 / hours_per_day  # Sm³/day -> scc/hr
    liquid_rate_metric_to_si: float = days_per_second  # Sm³/day -> Sm³/s
    liquid_rate_lab_to_field: float = 1.0 / liquid_rate_field_to_lab
    liquid_rate_lab_to_metric: float = 1.0 / liquid_rate_metric_to_lab
    liquid_rate_lab_to_si: float = cm3_to_m3 * seconds_per_hour  # scc/hr -> Sm³/s
    liquid_rate_si_to_field: float = 1.0 / liquid_rate_field_to_si
    liquid_rate_si_to_metric: float = 1.0 / liquid_rate_metric_to_si
    liquid_rate_si_to_lab: float = 1.0 / liquid_rate_lab_to_si

    # Surface gas rates: SCF/day -> Sm³/day, scc/hr, Sm³/s
    gas_rate_field_to_metric: float = scf_to_m3  # SCF/day -> Sm³/day
    gas_rate_field_to_lab: float = scf_to_cm3 / hours_per_day  # SCF/day -> scc/hr
    gas_rate_field_to_si: float = scf_to_m3 / seconds_per_day  # SCF/day -> Sm³/s
    gas_rate_metric_to_field: float = 1.0 / gas_rate_field_to_metric
    gas_rate_metric_to_lab: float = m3_to_cm3 / hours_per_day  # Sm³/day -> scc/hr
    gas_rate_metric_to_si: float = days_per_second  # Sm³/day -> Sm³/s
    gas_rate_lab_to_field: float = 1.0 / gas_rate_field_to_lab
    gas_rate_lab_to_metric: float = 1.0 / gas_rate_metric_to_lab
    gas_rate_lab_to_si: float = cm3_to_m3 * seconds_per_hour  # scc/hr -> Sm³/s
    gas_rate_si_to_field: float = 1.0 / gas_rate_field_to_si
    gas_rate_si_to_metric: float = 1.0 / gas_rate_metric_to_si
    gas_rate_si_to_lab: float = 1.0 / gas_rate_lab_to_si

    # Reservoir rates: ft³/day -> m³/day, cm³/hr, m³/s
    res_rate_field_to_metric: float = ft3_to_m3  # ft³/day -> m³/day
    res_rate_field_to_lab: float = ft3_to_cm3 / hours_per_day  # ft³/day -> cm³/hr
    res_rate_field_to_si: float = ft3_to_m3 / seconds_per_day  # ft³/day -> m³/s
    res_rate_metric_to_field: float = 1.0 / res_rate_field_to_metric
    res_rate_metric_to_lab: float = m3_to_cm3 / hours_per_day  # m³/day -> cm³/hr
    res_rate_metric_to_si: float = days_per_second  # m³/day -> m³/s
    res_rate_lab_to_field: float = 1.0 / res_rate_field_to_lab
    res_rate_lab_to_metric: float = 1.0 / res_rate_metric_to_lab
    res_rate_lab_to_si: float = cm3_to_m3 * seconds_per_hour  # cm³/hr -> m³/s
    res_rate_si_to_field: float = 1.0 / res_rate_field_to_si
    res_rate_si_to_metric: float = 1.0 / res_rate_metric_to_si
    res_rate_si_to_lab: float = 1.0 / res_rate_lab_to_si

    # Mass
    mass_field_to_metric: float = lbm_ft3_to_kg_m3 * ft3_to_m3  # lbm -> kg
    mass_field_to_lab: float = lbm_ft3_to_g_cm3 * ft3_to_cm3  # lbm -> g
    mass_metric_to_field: float = 1.0 / mass_field_to_metric
    mass_metric_to_lab: float = 1.0 / kg_m3_to_g_cm3  # kg -> g (1000)
    mass_lab_to_field: float = 1.0 / mass_field_to_lab
    mass_lab_to_metric: float = kg_m3_to_g_cm3  # g -> kg

    def _inverse(x: float) -> float:
        return 1.0 / x

    table: UnitConversionTable = {
        ##############
        # FIELD -> *
        ##############
        (UnitSystem.FIELD, UnitSystem.METRIC): UnitConversionFactors(
            pressure=psi_to_bar,
            length=ft_to_m,
            area=ft_to_m**2,
            volume=ft3_to_m3,
            time=1.0,  # day -> day
            mass=mass_field_to_metric,
            temperature=5.0 / 9.0,
            temperature_offset=(-32.0) * (5.0 / 9.0),  # °F -> °C
            density=lbm_ft3_to_kg_m3,
            viscosity=1.0,  # cP -> cP
            permeability=1.0,  # mD -> mD
            compressibility=_inverse(psi_to_bar),
            liquid_surface_volume=stb_to_m3,
            gas_surface_volume=scf_to_m3,
            liquid_fvf=liquid_fvf_field_to_metric,
            gas_fvf=gas_fvf_field_to_metric,
            gas_oil_ratio=gor_field_to_metric,
            oil_gas_ratio=ogr_field_to_metric,
            liquid_surface_rate=liquid_rate_field_to_metric,
            gas_surface_rate=gas_rate_field_to_metric,
            reservoir_rate=res_rate_field_to_metric,
        ),
        (UnitSystem.FIELD, UnitSystem.SI): UnitConversionFactors(
            pressure=psi_to_pa,
            length=ft_to_m,
            area=ft_to_m**2,
            volume=ft3_to_m3,
            time=days_per_second,  # day -> s
            mass=mass_field_to_metric,  # lbm -> kg (SI mass = kg)
            temperature=5.0 / 9.0,
            temperature_offset=(-32.0 * 5.0 / 9.0) + 273.15,  # °F -> K
            density=lbm_ft3_to_kg_m3,
            viscosity=cp_to_pas,
            permeability=md_to_m2,
            compressibility=_inverse(psi_to_pa),
            liquid_surface_volume=stb_to_m3,
            gas_surface_volume=scf_to_m3,
            liquid_fvf=liquid_fvf_field_to_si,
            gas_fvf=gas_fvf_field_to_metric,  # rcf/SCF -> rm³/Sm³ same as metric
            gas_oil_ratio=gor_field_to_metric,
            oil_gas_ratio=ogr_field_to_metric,
            liquid_surface_rate=liquid_rate_field_to_si,
            gas_surface_rate=gas_rate_field_to_si,
            reservoir_rate=res_rate_field_to_si,
        ),
        (UnitSystem.FIELD, UnitSystem.LAB): UnitConversionFactors(
            pressure=psi_to_atm,
            length=ft_to_cm,
            area=ft_to_cm**2,
            volume=ft3_to_cm3,
            time=_inverse(hours_per_day),  # day -> hr
            mass=mass_field_to_lab,
            temperature=5.0 / 9.0,
            temperature_offset=(-32.0) * (5.0 / 9.0),  # °F -> °C
            density=lbm_ft3_to_g_cm3,
            viscosity=1.0,  # cP -> cP
            permeability=1.0,  # mD -> mD
            compressibility=_inverse(psi_to_atm),
            liquid_surface_volume=stb_to_cm3,
            gas_surface_volume=scf_to_cm3,
            liquid_fvf=liquid_fvf_field_to_lab,
            gas_fvf=gas_fvf_field_to_lab,
            gas_oil_ratio=gor_field_to_lab,
            oil_gas_ratio=ogr_field_to_lab,
            liquid_surface_rate=liquid_rate_field_to_lab,
            gas_surface_rate=gas_rate_field_to_lab,
            reservoir_rate=res_rate_field_to_lab,
        ),
        ##############
        # METRIC -> *
        ##############
        (UnitSystem.METRIC, UnitSystem.FIELD): UnitConversionFactors(
            pressure=_inverse(psi_to_bar),
            length=m_to_ft,
            area=m_to_ft**2,
            volume=m3_to_ft3,
            time=1.0,  # day -> day
            mass=mass_metric_to_field,
            temperature=9.0 / 5.0,
            temperature_offset=32.0,  # °C -> °F
            density=_inverse(lbm_ft3_to_kg_m3),
            viscosity=1.0,
            permeability=1.0,
            compressibility=psi_to_bar,
            liquid_surface_volume=_inverse(stb_to_m3),
            gas_surface_volume=_inverse(scf_to_m3),
            liquid_fvf=_inverse(liquid_fvf_field_to_metric),
            gas_fvf=gas_fvf_metric_to_field,
            gas_oil_ratio=gor_metric_to_field,
            oil_gas_ratio=ogr_metric_to_field,
            liquid_surface_rate=liquid_rate_metric_to_field,
            gas_surface_rate=gas_rate_metric_to_field,
            reservoir_rate=res_rate_metric_to_field,
        ),
        (UnitSystem.METRIC, UnitSystem.SI): UnitConversionFactors(
            pressure=bar_to_pa,
            length=1.0,
            area=1.0,
            volume=1.0,
            time=days_per_second,  # day -> s
            mass=1.0,  # kg -> kg
            temperature=1.0,
            temperature_offset=273.15,  # °C -> K
            density=1.0,
            viscosity=cp_to_pas,
            permeability=md_to_m2,
            compressibility=_inverse(bar_to_pa),
            liquid_surface_volume=1.0,  # Sm³ -> Sm³
            gas_surface_volume=1.0,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=1.0,
            oil_gas_ratio=1.0,
            liquid_surface_rate=liquid_rate_metric_to_si,
            gas_surface_rate=gas_rate_metric_to_si,
            reservoir_rate=res_rate_metric_to_si,
        ),
        (UnitSystem.METRIC, UnitSystem.LAB): UnitConversionFactors(
            pressure=bar_to_atm,
            length=m_to_cm,
            area=m_to_cm**2,
            volume=m3_to_cm3,
            time=_inverse(hours_per_day),  # day -> hr
            mass=mass_metric_to_lab,
            temperature=1.0,
            temperature_offset=0.0,  # °C -> °C
            density=kg_m3_to_g_cm3,
            viscosity=1.0,
            permeability=1.0,
            compressibility=_inverse(bar_to_atm),
            liquid_surface_volume=sm3_to_scc,  # Sm³ -> scc
            gas_surface_volume=sm3_to_scc,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_metric_to_lab,
            oil_gas_ratio=ogr_metric_to_lab,
            liquid_surface_rate=liquid_rate_metric_to_lab,
            gas_surface_rate=gas_rate_metric_to_lab,
            reservoir_rate=res_rate_metric_to_lab,
        ),
        ##############
        # SI -> *
        ##############
        (UnitSystem.SI, UnitSystem.FIELD): UnitConversionFactors(
            pressure=_inverse(psi_to_pa),
            length=m_to_ft,
            area=m_to_ft**2,
            volume=m3_to_ft3,
            time=seconds_per_day,  # s -> day
            mass=_inverse(mass_field_to_metric),
            temperature=9.0 / 5.0,
            temperature_offset=(-273.15 * 9.0 / 5.0) + 32.0,  # K -> °F
            density=_inverse(lbm_ft3_to_kg_m3),
            viscosity=_inverse(cp_to_pas),
            permeability=_inverse(md_to_m2),
            compressibility=psi_to_pa,
            liquid_surface_volume=_inverse(stb_to_m3),
            gas_surface_volume=_inverse(scf_to_m3),
            liquid_fvf=_inverse(liquid_fvf_field_to_si),
            gas_fvf=_inverse(gas_fvf_field_to_metric),
            gas_oil_ratio=gor_metric_to_field,
            oil_gas_ratio=ogr_metric_to_field,
            liquid_surface_rate=liquid_rate_si_to_field,
            gas_surface_rate=gas_rate_si_to_field,
            reservoir_rate=res_rate_si_to_field,
        ),
        (UnitSystem.SI, UnitSystem.METRIC): UnitConversionFactors(
            pressure=_inverse(bar_to_pa),
            length=1.0,
            area=1.0,
            volume=1.0,
            time=seconds_per_day,  # s -> day
            mass=1.0,
            temperature=1.0,
            temperature_offset=-273.15,  # K -> °C
            density=1.0,
            viscosity=_inverse(cp_to_pas),
            permeability=_inverse(md_to_m2),
            compressibility=bar_to_pa,
            liquid_surface_volume=1.0,
            gas_surface_volume=1.0,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=1.0,
            oil_gas_ratio=1.0,
            liquid_surface_rate=liquid_rate_si_to_metric,
            gas_surface_rate=gas_rate_si_to_metric,
            reservoir_rate=res_rate_si_to_metric,
        ),
        (UnitSystem.SI, UnitSystem.LAB): UnitConversionFactors(
            pressure=_inverse(atm_to_pa),
            length=m_to_cm,
            area=m_to_cm**2,
            volume=m3_to_cm3,
            time=seconds_per_hour,  # s -> hr
            mass=mass_metric_to_lab,  # kg -> g
            temperature=1.0,
            temperature_offset=-273.15,  # K -> °C
            density=kg_m3_to_g_cm3,
            viscosity=_inverse(cp_to_pas),
            permeability=_inverse(md_to_m2),
            compressibility=atm_to_pa,
            liquid_surface_volume=sm3_to_scc,
            gas_surface_volume=sm3_to_scc,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_metric_to_lab,
            oil_gas_ratio=ogr_metric_to_lab,
            liquid_surface_rate=liquid_rate_si_to_lab,
            gas_surface_rate=gas_rate_si_to_lab,
            reservoir_rate=res_rate_si_to_lab,
        ),
        ##############
        # LAB -> *
        ##############
        (UnitSystem.LAB, UnitSystem.FIELD): UnitConversionFactors(
            pressure=_inverse(psi_to_atm),
            length=cm_to_ft,
            area=cm_to_ft**2,
            volume=cm3_to_ft3,
            time=hours_per_day,  # hr -> day
            mass=mass_lab_to_field,
            temperature=9.0 / 5.0,
            temperature_offset=32.0,  # °C -> °F
            density=_inverse(lbm_ft3_to_g_cm3),
            viscosity=1.0,
            permeability=1.0,
            compressibility=psi_to_atm,
            liquid_surface_volume=_inverse(stb_to_cm3),
            gas_surface_volume=_inverse(scf_to_cm3),
            liquid_fvf=_inverse(liquid_fvf_field_to_lab),
            gas_fvf=gas_fvf_lab_to_field,
            gas_oil_ratio=gor_lab_to_field,
            oil_gas_ratio=ogr_lab_to_field,
            liquid_surface_rate=liquid_rate_lab_to_field,
            gas_surface_rate=gas_rate_lab_to_field,
            reservoir_rate=res_rate_lab_to_field,
        ),
        (UnitSystem.LAB, UnitSystem.METRIC): UnitConversionFactors(
            pressure=_inverse(bar_to_atm),
            length=cm_to_m,
            area=cm_to_m**2,
            volume=cm3_to_m3,
            time=hours_per_day,  # hr -> day
            mass=mass_lab_to_metric,
            temperature=1.0,
            temperature_offset=0.0,  # °C -> °C
            density=_inverse(kg_m3_to_g_cm3),
            viscosity=1.0,
            permeability=1.0,
            compressibility=bar_to_atm,
            liquid_surface_volume=scc_to_sm3,
            gas_surface_volume=scc_to_sm3,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_lab_to_metric,
            oil_gas_ratio=ogr_lab_to_metric,
            liquid_surface_rate=liquid_rate_lab_to_metric,
            gas_surface_rate=gas_rate_lab_to_metric,
            reservoir_rate=res_rate_lab_to_metric,
        ),
        (UnitSystem.LAB, UnitSystem.SI): UnitConversionFactors(
            pressure=atm_to_pa,
            length=cm_to_m,
            area=cm_to_m**2,
            volume=cm3_to_m3,
            time=seconds_per_hour,  # hr -> s
            mass=mass_lab_to_metric,  # g -> kg
            temperature=1.0,
            temperature_offset=273.15,  # °C -> K
            density=_inverse(kg_m3_to_g_cm3),
            viscosity=cp_to_pas,
            permeability=md_to_m2,
            compressibility=_inverse(atm_to_pa),
            liquid_surface_volume=scc_to_sm3,
            gas_surface_volume=scc_to_sm3,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_lab_to_metric,
            oil_gas_ratio=ogr_lab_to_metric,
            liquid_surface_rate=liquid_rate_lab_to_si,
            gas_surface_rate=gas_rate_lab_to_si,
            reservoir_rate=res_rate_lab_to_si,
        ),
    }
    return table


IDENTITY_FACTORS = UnitConversionFactors(
    pressure=1.0,
    length=1.0,
    area=1.0,
    volume=1.0,
    time=1.0,
    mass=1.0,
    temperature=1.0,
    temperature_offset=0.0,
    density=1.0,
    viscosity=1.0,
    permeability=1.0,
    compressibility=1.0,
    liquid_surface_volume=1.0,
    gas_surface_volume=1.0,
    liquid_fvf=1.0,
    gas_fvf=1.0,
    gas_oil_ratio=1.0,
    oil_gas_ratio=1.0,
    liquid_surface_rate=1.0,
    gas_surface_rate=1.0,
    reservoir_rate=1.0,
)
"""Identity unit conversion factors. Has all multiplicative factors as 1.0, and offset 0.0."""

UNIT_CONVERSION_TABLE = build_unit_conversion_table()
"""Default unit conversion table"""


def get_conversion_factors(
    from_system: UnitSystem,
    to_system: UnitSystem,
    /,
    *,
    table: UnitConversionTable | None = None,
) -> UnitConversionFactors:
    """
    Return a dictionary of scalar conversion factors for every physical
    dimension used by the classes in this library.

    Each value converts a quantity expressed in `from_system` to
    `to_system` by **multiplication**, except temperature which uses an
    affine map stored as two keys:

    - "temperature"  - multiplicative factor.
    - "temperature_offset" - additive delta (in target units) applied
    *after* scaling: T_to = T_from * scale + offset.

    :param from_system: Source `UnitSystem`.
    :param to_system: Target `UnitSystem`.
    :returns: Conversion-factor dictionary.
    :raises KeyError: If the (from_system, to_system) pair is not defined.
    """
    if from_system == to_system:
        return IDENTITY_FACTORS

    table = table or build_unit_conversion_table()
    key = (from_system, to_system)
    if key not in table:
        pairs = [f"{a.value} -> {b.value}" for a, b in table]
        raise KeyError(
            f"No unit conversion defined from {from_system.value!r} "
            f"to {to_system.value!r}. Supported pairs: {pairs}."
        )
    return table[key]
