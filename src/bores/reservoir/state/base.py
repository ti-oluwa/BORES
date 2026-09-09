import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.precision import get_dtype
from bores.serde.stores import StoreSerializable
from bores.types import (
    BooleanCellArray,
    CellArray,
    UnitSystem,
)
from bores.utils import scale, scale_and_offset

__all__ = ["Hysteresis", "ReservoirState"]


def advance_phase_hysteresis(
    new_saturation: npt.NDArray,
    previous_saturation: npt.NDArray,
    max_saturation: npt.NDArray,
    imbibition_flag: npt.NDArray,
    reversal_saturation: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Updates one phase's tracked saturation history for one new saturation
    value, per cell.

    A phase is on its imbibition path when its saturation is moving from
    `reversal_saturation` toward `max_saturation` (see
    `Hysteresis.water_imbibition_flag`/`gas_imbibition_flag`). Comparing
    the new saturation only against the historical maximum isn't enough to
    catch every reversal: a phase can turn around well before it climbs
    back past its own historical peak, so the previous saturation is
    needed to tell which way it is actually moving right now.

    :param new_saturation: This timestep's saturation, per cell.
    :param previous_saturation: The saturation this phase had before this update.
    :param max_saturation: The historical maximum before this update.
    :param imbibition_flag: Whether the phase was increasing before this update.
    :param reversal_saturation: The saturation at the last known reversal,
        before this update.
    :returns: `(new_max_saturation, new_imbibition_flag, new_reversal_saturation)`.
    """
    increasing_now = new_saturation > previous_saturation
    decreasing_now = new_saturation < previous_saturation
    turned_to_drainage = imbibition_flag & decreasing_now
    turned_to_imbibition = ~imbibition_flag & increasing_now

    new_max_saturation = np.maximum(max_saturation, new_saturation)
    new_imbibition_flag = np.where(
        increasing_now, True, np.where(decreasing_now, False, imbibition_flag)
    )
    new_reversal_saturation = np.where(
        turned_to_drainage,
        max_saturation,
        np.where(turned_to_imbibition, previous_saturation, reversal_saturation),
    )
    return new_max_saturation, new_imbibition_flag, new_reversal_saturation


@attrs.frozen(slots=True)
class Hysteresis(StoreSerializable):
    """
    Drainage / imbibition hysteresis tracking for Killough scanning curves.

    Maintains historical saturation extrema and displacement-regime flags
    required to compute effective residual saturations on the scanning curves.
    These are consumed inside relative-permeability and capillary-pressure
    evaluation routines; the flow solver does not interpret them directly.

    All arrays are dimensionless (saturations, flags) and therefore require no unit conversion.
    """

    max_water_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum water saturation reached in each
    cell (fraction).

    Initialised to the initial water saturation. Updated whenever the
    current water saturation exceeds the stored maximum. Determines the
    imbibition end-point on the scanning curve when drainage reverses.
    """

    max_gas_saturation: CellArray
    """
    Shape (n_cells,) - historical maximum gas saturation reached in each
    cell (fraction).

    Analogous to `max_water_saturation` for the gas phase.
    """

    water_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - `True` if the current water-phase displacement is
    imbibition (water saturation increasing toward `max_water_saturation`).

    `False` indicates drainage (water saturation decreasing).
    """

    gas_imbibition_flag: BooleanCellArray
    """
    Shape (n_cells,) - `True` if gas saturation is currently increasing
    toward `max_gas_saturation`, `False` if decreasing (drainage).

    Same convention as `water_imbibition_flag`: the flag tracks that
    phase's own saturation, not a wetting/non-wetting distinction.
    """

    water_reversal_saturation: CellArray
    """
    Shape (n_cells,) - water saturation at the most recent
    drainage-to-imbibition (or reverse) reversal point (fraction).

    Starting saturation of the Killough scanning curve when the displacement
    regime changes.
    """

    gas_reversal_saturation: CellArray
    """
    Shape (n_cells,) - gas saturation at the most recent reversal point
    (fraction).

    Analogous to `water_reversal_saturation` for the gas phase.
    """

    @classmethod
    def from_initial_saturation(
        cls,
        water_saturation: npt.ArrayLike,
        gas_saturation: npt.ArrayLike,
    ) -> Self:
        """
        Construct a `Hysteresis` from initial saturation arrays.

        Sets maximum saturations to the initial values, marks all cells as
        drainage (not yet reversing), and places reversal points at the
        initial saturation values.

        :param water_saturation: Array-like (n_cells,) - initial water
            saturation per cell (fraction).
        :param gas_saturation: Array-like (n_cells,) - initial gas saturation
            per cell (fraction).
        :returns: Initialised `Hysteresis`.
        """
        sw = np.asarray(water_saturation, dtype=get_dtype())
        sg = np.asarray(gas_saturation, dtype=get_dtype())
        return cls(
            max_water_saturation=typing.cast(CellArray, sw.copy()),
            max_gas_saturation=typing.cast(CellArray, sg.copy()),
            water_imbibition_flag=typing.cast(
                BooleanCellArray, np.zeros(sw.shape, dtype=np.bool_)
            ),
            gas_imbibition_flag=typing.cast(BooleanCellArray, np.zeros(sg.shape, dtype=np.bool_)),
            water_reversal_saturation=typing.cast(CellArray, sw.copy()),
            gas_reversal_saturation=typing.cast(CellArray, sg.copy()),
        )

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `Hysteresis` with selected fields replaced.

        :param kwargs: Field names and their replacement values.
        :returns: New immutable `Hysteresis`.
        """
        return attrs.evolve(self, **kwargs)

    def advance(
        self,
        water_saturation: npt.ArrayLike,
        gas_saturation: npt.ArrayLike,
        previous_water_saturation: npt.ArrayLike,
        previous_gas_saturation: npt.ArrayLike,
    ) -> Self:
        """
        Return a new `Hysteresis` updated for the current timestep's saturations.

        For each phase: if that phase's saturation is still moving the
        same way it was last time, its historical maximum extends (if
        increasing) and its reversal point stays put. If it has turned
        around since the last update, the reversal point moves to where
        the turn happened, and the imbibition flag flips.

        Call this once per timestep, after the new saturations for that
        step are known, to keep the tracked history current for the next
        saturation-function evaluation.

        :param water_saturation: Array-like (n_cells,) - this timestep's
            water saturation per cell (fraction).
        :param gas_saturation: Array-like (n_cells,) - this timestep's gas
            saturation per cell (fraction).
        :param previous_water_saturation: Array-like (n_cells,) - the water
            saturation this cell had before this timestep, needed to tell
            which way it is currently moving.
        :param previous_gas_saturation: Array-like (n_cells,) - the gas
            saturation this cell had before this timestep.
        :returns: New `Hysteresis` reflecting this timestep.
        """
        sw = np.asarray(water_saturation, dtype=get_dtype())
        sg = np.asarray(gas_saturation, dtype=get_dtype())
        previous_sw = np.asarray(previous_water_saturation, dtype=get_dtype())
        previous_sg = np.asarray(previous_gas_saturation, dtype=get_dtype())

        new_max_water, new_water_flag, new_water_reversal = advance_phase_hysteresis(
            new_saturation=sw,
            previous_saturation=previous_sw,
            max_saturation=np.asarray(self.max_water_saturation),
            imbibition_flag=np.asarray(self.water_imbibition_flag),
            reversal_saturation=np.asarray(self.water_reversal_saturation),
        )
        new_max_gas, new_gas_flag, new_gas_reversal = advance_phase_hysteresis(
            new_saturation=sg,
            previous_saturation=previous_sg,
            max_saturation=np.asarray(self.max_gas_saturation),
            imbibition_flag=np.asarray(self.gas_imbibition_flag),
            reversal_saturation=np.asarray(self.gas_reversal_saturation),
        )
        return self.evolve(
            max_water_saturation=new_max_water,
            max_gas_saturation=new_max_gas,
            water_imbibition_flag=new_water_flag,
            gas_imbibition_flag=new_gas_flag,
            water_reversal_saturation=new_water_reversal,
            gas_reversal_saturation=new_gas_reversal,
        )


@attrs.frozen(slots=True)
class ReservoirState(StoreSerializable):
    """
    Reservoir (dynamic) per-cell state, updated at every time step.

    This is the canonical state that the solver integrates forward in time.
    It holds exactly the quantities that cannot be recomputed from static
    data alone:

    **Primary unknowns** (solved implicitly):

    - `pressure` - oil-phase reference pressure.
    - `oil_saturation`, `water_saturation`, `gas_saturation` - phase
      saturations (must sum to 1 in every cell).
    - `solution_gor` (Rs) - gas dissolved in oil per unit stock-tank oil
      volume.  *Primary variable in saturated cells* (Sg > 0); capped at
      bubble-point Rs in undersaturated cells.
    - `oil_bubble_point_pressure` - *primary variable in undersaturated oil
      cells* (Sg = 0, Po < Pbub is not possible, so here Pbub tracks the
      evolving bubble-point as the reservoir depletes below saturation
      pressure).
    - `vaporized_oil_to_gas_ratio` (Rv) - stock-tank oil vaporized in gas per unit
      standard gas volume. Primary variable for volatile-oil / gas-condensate
      models when So = 0.
    - `water_bubble_point_pressure` - bubble-point pressure of the water
      phase with respect to dissolved gas (Rsw).  Relevant for CO₂ or sour-gas
      reservoirs where gas solubility in water is non-negligible.
    - `gas_dew_point_pressure` - dew-point pressure of the gas phase; primary
      variable in condensate models when So = 0.
    - `gas_solubility_in_water` (Rsw) - gas dissolved in water per unit
      stock-tank water volume.

    **Conserved component masses** (updated explicitly):

    - `oil_mass`, `water_mass`, `free_gas_mass`
    - `dissolved_gas_mass_in_oil`, `dissolved_gas_mass_in_water`
    - `vaporized_oil_mass_in_gas`

    Use `convert(target)` to rescale to another unit system, or
    `evolve(**kwargs)` to produce a new state with selected fields replaced.
    """

    pressure: CellArray
    """
    Shape (n_cells,) - oil-phase (reference) pressure.

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).
    Primary implicit unknown in the pressure equation.
    Phase pressures for water and gas are recovered via capillary pressure:
    Pw = Po - Pcow,  Pg = Po + Pcgo.
    """

    temperature: CellArray
    """
    Shape (n_cells,) - reservoir temperature, one value per cell.

    Units: °F (FIELD), °C (METRIC / LAB), K (SI).

    For standard isothermal black-oil the array is uniform (a single
    reservoir temperature broadcast across all cells) or a per-region temperature distribution.
    """

    oil_saturation: CellArray
    """
    Shape (n_cells,) - oil-phase saturation (fraction, [0, 1]).

    Derived as So = 1 - Sw - Sg after updating Sw and Sg.
    Must satisfy So + Sw + Sg = 1 in every cell.
    """

    water_saturation: CellArray
    """Shape (n_cells,) - water-phase saturation (fraction, [0, 1])."""

    gas_saturation: CellArray
    """
    Shape (n_cells,) - free gas-phase saturation (fraction, [0, 1]).

    Includes only free (non-dissolved, non-vaporized) gas.
    """

    solution_gor: CellArray
    """
    Shape (n_cells,) - solution gas-to-oil ratio (Rs).

    Units: SCF/STB (FIELD), sm³/sm³ (METRIC / SI), scc/scc (LAB).

    Gas dissolved in oil per unit stock-tank oil volume at current pressure.

    *Primary variable in saturated cells* (Sg > 0): updated by the solver.
    *In undersaturated cells* (Sg = 0): fixed at the bubble-point value
    corresponding to `oil_bubble_point_pressure`; not independently solved.
    """

    oil_bubble_point_pressure: CellArray
    """
    Shape (n_cells,) - bubble-point pressure of the oil phase (Pbub).

    Units: psi (FIELD), bar (METRIC), atm (LAB), Pa (SI).

    *In saturated cells* (Sg > 0): computed from Rs via the PVT table - not
    a stored primary variable; could be derived from `solution_gor`.

    *In undersaturated cells* (Sg = 0): this IS a primary variable, tracking
    the evolving bubble-point as the reservoir depletes while remaining single-
    phase.  Must be stored and checkpointed in this regime.

    The two-regime handling is the same saturated/undersaturated switching
    used by Eclipse E100 (PBPD switching logic).
    """

    vaporized_oil_to_gas_ratio: CellArray
    """
    Shape (n_cells,) - vaporized oil ratio (Rv).

    Units: STB/Mscf (FIELD), sm³/sm³ (METRIC / SI), scc/scc (LAB).

    Stock-tank oil vaporized in gas per unit standard gas volume.

    Non-zero only for volatile-oil and gas-condensate reservoirs.
    Set to all-zeros for standard dry-gas black-oil.
    """

    gas_dew_point_pressure: CellArray
    """
    Shape (n_cells,) - dew-point pressure of the gas phase (Pdew).

    Units: same as `oil_bubble_point_pressure`.

    Analogous to `oil_bubble_point_pressure` but for the gas phase in
    volatile-oil / gas-condensate models:

    *In two-phase cells* (So > 0): derived from Rv via the PVT table.
    *In single-phase gas cells* (So = 0): primary variable, tracking the
    evolving dew-point as the condensate reservoir depletes.

    Set to all-zeros for standard black-oil.
    """

    gas_solubility_in_water: CellArray
    """
    Shape (n_cells,) - gas solubility in water (Rsw).

    Units: same as `solution_gor`.

    Non-negligible for CO₂ or sour-gas injection scenarios; zero for
    standard black-oil with no dissolved gas in the water phase.
    """

    water_bubble_point_pressure: CellArray
    """
    Shape (n_cells,) - bubble-point pressure of the water phase with respect
    to dissolved gas (Pbub,w).

    Units: same as `oil_bubble_point_pressure`.

    Tracks the pressure at which dissolved gas (Rsw) begins to exsolve from
    water.  Relevant for CO₂ or sour-gas reservoirs.  In the undersaturated
    water regime (no free gas exsolving from water), this is a primary
    variable analogous to `oil_bubble_point_pressure`.

    Set to all-zeros for standard black-oil with Rsw = 0.
    """

    oil_mass: CellArray
    """
    Shape (n_cells,) - oil-component mass in each cell.

    Units: lbm (FIELD), kg (METRIC / SI), g (LAB).

    Conserved quantity in the oil-component material balance.
    Tracks only the liquid oil phase; vaporized oil in gas is accumulated
    separately in `vaporized_oil_mass_in_gas`.
    """

    water_mass: CellArray
    """
    Shape (n_cells,) - water-component mass in each cell.

    Units: same as `oil_mass`.
    """

    free_gas_mass: CellArray
    """
    Shape (n_cells,) - free gas-component mass in each cell.

    Units: same as `oil_mass`.

    The total gas material balance is:
    free_gas_mass + dissolved_gas_mass_in_oil + dissolved_gas_mass_in_water.
    """

    dissolved_gas_mass_in_oil: CellArray
    """
    Shape (n_cells,) - gas dissolved in the oil phase.

    Units: same as `oil_mass`.

    Equal to Rs x oil_mass x (ρg_STC / ρo_STC) at standard conditions.
    Tracked separately so the gas material balance can be assembled without
    recomputing Rs at every cell.
    """

    dissolved_gas_mass_in_water: CellArray
    """
    Shape (n_cells,) - gas dissolved in the water phase.

    Units: same as `oil_mass`.

    Non-negligible for CO₂ injection or sour-gas reservoirs (Rsw > 0).
    Zero for standard black-oil.
    """

    vaporized_oil_mass_in_gas: CellArray
    """
    Shape (n_cells,) - oil (condensate) vaporized in the gas phase.

    Units: same as `oil_mass`.

    Equal to Rv x free_gas_mass x (ρo_STC / ρg_STC) at standard conditions.
    Part of the *oil-component* material balance, not the gas balance.
    Zero for standard dry-gas black-oil; required for volatile-oil and
    gas-condensate simulations.
    """

    solvent_concentration: CellArray = attrs.field(factory=lambda: np.zeros(0, dtype=get_dtype()))
    """
    Shape (n_cells,) - solvent volume fraction in the oil-phase mixture
    (dimensionless, [0, 1]).

    0 = pure oil; 1 = pure solvent.  Populated only for Todd-Longstaff or
    similar EOR miscibility models.  Defaults to an empty array for standard
    black-oil (zero memory cost).
    """

    hysteresis: Hysteresis | None = None
    """
    Optional `HysteresisState` for Killough scanning curves.
    `None` (default) for simulations without hysteresis.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities are expressed.

    Use `convert(target)` to produce a rescaled copy.
    """

    @property
    def total_gas_mass(self) -> CellArray:
        """
        Shape (n_cells,) - total gas-component mass per cell.

        m_g,total = free_gas_mass + dissolved_gas_mass_in_oil
                    + dissolved_gas_mass_in_water

        This is the conserved quantity in the gas-component material balance.
        Note that `vaporized_oil_mass_in_gas` belongs to the *oil* component
        balance, not the gas balance.
        """
        return typing.cast(
            CellArray,
            self.free_gas_mass + self.dissolved_gas_mass_in_oil + self.dissolved_gas_mass_in_water,
        )

    @property
    def total_oil_mass(self) -> CellArray:
        """
        Shape (n_cells,) - total oil-component mass per cell.

        m_o,total = oil_mass + vaporized_oil_mass_in_gas

        This is the conserved quantity in the oil-component material balance
        for volatile-oil / gas-condensate models.
        """
        return typing.cast(CellArray, self.oil_mass + self.vaporized_oil_mass_in_gas)

    def evolve(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `ReservoirState` with selected fields replaced.

        All fields not present in *kwargs* are carried forward unchanged.
        Preferred solver pattern:

        ```python
        new_state = state.evolve(
            pressure=new_pressure,
            oil_saturation=new_oil_saturation,
            water_saturation=new_water_saturation,
            gas_saturation=new_gas_Saturation,
            oil_mass=new_oil_mass,
            free_gas_mass=new_gas_mass,
        )
        ```

        :param kwargs: Field names and their replacement values.
        :returns: New immutable `ReservoirState`.
        :raises TypeError: If an unknown field name is passed.
        """
        return attrs.evolve(self, **kwargs)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `ReservoirState` with all dimensional quantities rescaled
        to *target*.

        Dimensionless fields (saturations, `solvent_concentration`) are
        copied unchanged.  Rs, Rv, and Rsw are scaled by the GOR factor.
        Pressures use the pressure factor.  Masses use the combined density x
        length³ factor.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `ReservoirState` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        mass_factor = factors["mass"]
        pressure_factor = factors["pressure"]
        gor_factor = factors["gas_oil_ratio"]
        return attrs.evolve(
            self,
            pressure=scale(self.pressure, pressure_factor),
            temperature=scale_and_offset(
                self.temperature,
                factor=factors["temperature"],
                offset=factors["temperature_offset"],
            ),
            oil_saturation=self.oil_saturation,
            water_saturation=self.water_saturation,
            gas_saturation=self.gas_saturation,
            solution_gor=scale(self.solution_gor, gor_factor),
            oil_bubble_point_pressure=scale(self.oil_bubble_point_pressure, pressure_factor),
            vaporized_oil_to_gas_ratio=scale(self.vaporized_oil_to_gas_ratio, gor_factor),
            gas_dew_point_pressure=scale(self.gas_dew_point_pressure, pressure_factor),
            gas_solubility_in_water=scale(self.gas_solubility_in_water, gor_factor),
            water_bubble_point_pressure=scale(self.water_bubble_point_pressure, pressure_factor),
            oil_mass=scale(self.oil_mass, mass_factor),
            water_mass=scale(self.water_mass, mass_factor),
            free_gas_mass=scale(self.free_gas_mass, mass_factor),
            dissolved_gas_mass_in_oil=scale(self.dissolved_gas_mass_in_oil, mass_factor),
            dissolved_gas_mass_in_water=scale(self.dissolved_gas_mass_in_water, mass_factor),
            vaporized_oil_mass_in_gas=scale(self.vaporized_oil_mass_in_gas, mass_factor),
            solvent_concentration=self.solvent_concentration,
            hysteresis=self.hysteresis,
            unit_system=target,
        )
