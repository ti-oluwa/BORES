import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.capillary_pressures import (
    CapillaryPressureTable,
    ThreePhaseCapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
)
from bores.grids.utils import Spacing, make_saturation_grid
from bores.relperm import (
    MixingRule,
    RelativePermeabilityTable,
    ThreePhaseRelPermTable,
    TwoPhaseRelPermTable,
)
from bores.serialization import Serializable
from bores.types import (
    CapillaryPressures,
    FluidPhase,
    RelativePermeabilities,
)

__all__ = ["RockFluidTables", "as_capillary_pressure_table", "as_relperm_table"]


@typing.final
@attrs.frozen
class RockFluidTables(Serializable):
    """
    Tables defining rock-fluid interactions in the reservoir.

    Made up of a relative permeability table and an optional capillary pressure table. The relative
    permeability table is required, while the capillary pressure table is optional
    (but required if `config.disable_capillary_effects=False`).
    """

    relative_permeability_table: RelativePermeabilityTable
    """Callable that evaluates the relative permeability curves based on fluid saturations."""
    capillary_pressure_table: typing.Optional[CapillaryPressureTable] = None
    """Optional callable that evaluates the capillary pressure curves based on fluid saturations. This is required if `config.disable_capillary_effects=False`"""

    def get_relative_permeabilities(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Evaluates the relative permeability curves based on the provided fluid saturations and
        any additional parameters required by the specific relative permeability model.

        :param water_saturation: The saturation of water in the reservoir (between 0 and 1).
        :param oil_saturation: The saturation of oil in the reservoir (between 0 and 1).
        :param gas_saturation: The saturation of gas in the reservoir (between 0 and 1).
        :param kwargs: Additional parameters required by the specific relative permeability model
            (e.g., irreducible saturations, residual saturations, etc.).
        :return: A `RelativePermeabilities` object containing the relative permeabilities for water, oil, and gas
        """
        return self.relative_permeability_table.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def get_capillary_pressures(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Evaluates the capillary pressure curves based on the provided fluid saturations and
        any additional parameters required by the specific capillary pressure model.

        :param water_saturation: The saturation of water in the reservoir (between 0 and 1).
        :param oil_saturation: The saturation of oil in the reservoir (between 0 and 1).
        :param gas_saturation: The saturation of gas in the reservoir (between 0 and 1).
        :param kwargs: Additional parameters required by the specific capillary pressure model
            (e.g., entry pressure, pore size distribution index, etc.).
        :return: A `CapillaryPressures` object containing the capillary pressures for water-oil and gas-oil interfaces
        """
        if self.capillary_pressure_table is None:
            raise ValueError("Capillary pressure table is not defined.")
        return self.capillary_pressure_table.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


def _resolve_arg(
    arg: typing.Optional[float],
    model: object,
    attr: str,
    fn_name: str,
) -> float:
    """
    Return *arg* if given, otherwise read *attr* from *model*.

    :param arg: Explicit override value (`None` to fall back to the model).
    :param model: The analytical model instance.
    :param attr: Attribute name to read from the model.
    :param fn_name: Name of the calling factory function (used in the error message).
    :return: Resolved saturation value as a Python float.
    :raises ValueError: If neither *arg* nor the model attribute is set.
    """
    if arg is not None:
        return float(arg)

    v = getattr(model, attr, None)
    if v is None:
        raise ValueError(
            f"'{attr}' must be supplied either as an argument to "
            f"{fn_name}() or stored in the model."
        )
    return float(v)


def as_relperm_table(
    model: RelativePermeabilityTable,
    *,
    irreducible_water_saturation: typing.Optional[float] = None,
    residual_oil_saturation_water: typing.Optional[float] = None,
    residual_oil_saturation_gas: typing.Optional[float] = None,
    residual_gas_saturation: typing.Optional[float] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    oil_water_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    gas_oil_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] = "wetting",
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] = "non_wetting",
    n_points: int = 200,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    gas_oil_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    mixing_rule: typing.Optional[typing.Union[MixingRule, str]] = None,
) -> ThreePhaseRelPermTable:
    """
    Convert any `RelativePermeabilityTable` to a `ThreePhaseRelPermTable` backed by fast piecewise-linear
    `TwoPhaseRelPermTable` instances.

    If *model* is already a `ThreePhaseRelPermTable` it is returned unchanged (no-op).

    **How sampling works**

    Two two-phase sub-tables are built by sweeping one saturation while
    holding the others at their limiting values. Which saturation is swept
    and which is held fixed is determined jointly by the wetting phase and
    the reference phase axis for each sub-table:

    *Oil-water sub-table* (`Sg = 0` throughout):

    - `wetting_phase=WATER, reference_phase="wetting"` (default): sweep
      `Sw ∈ [Swc, 1 - Sorw]`, set `So = 1 - Sw`.
    - `wetting_phase=WATER, reference_phase="non_wetting"`: sweep
      `So ∈ [Sorw, 1 - Swc]`, set `Sw = Swc`.
    - `wetting_phase=OIL,   reference_phase="wetting"`: sweep
      `So ∈ [Sorw, 1 - Swc]`, set `Sw = Swc`.
    - `wetting_phase=OIL,   reference_phase="non_wetting"`: sweep
      `Sw ∈ [Swc, 1 - Sorw]`, set `So = 1 - Sw`.

    *Gas-oil sub-table* (`Sw = Swc` throughout):

    - `wetting_phase=OIL, reference_phase="non_wetting"` (default): sweep
      `Sg ∈ [Sgr, 1 - Swc - Sorg]`, set `So = 1 - Swc - Sg`.
    - `wetting_phase=OIL, reference_phase="wetting"`: sweep
      `So ∈ [Sorg, 1 - Swc - Sgr]`, set `Sg = 1 - Swc - So`.
    - `wetting_phase=GAS, reference_phase="wetting"`: sweep
      `Sg ∈ [Sgr, 1 - Swc - Sorg]`, set `So = 1 - Swc - Sg`.
    - `wetting_phase=GAS, reference_phase="non_wetting"`: sweep
      `So ∈ [Sorg, 1 - Swc - Sgr]`, set `Sg = 1 - Swc - So`.

    :param model: Source analytical (or tabular) relative permeability model.
    :param irreducible_water_saturation: Irreducible (connate) water saturation
        `Swc`. Used to bound the mobile saturation window and forwarded to
        the model. May be omitted if the model already carries a default.
    :param residual_oil_saturation_water: Residual oil saturation after water
        flood `Sorw`. May be omitted if stored in the model.
    :param residual_oil_saturation_gas: Residual oil saturation after gas flood
        `Sorg`. May be omitted if stored in the model.
    :param residual_gas_saturation: Residual gas saturation `Sgr`. May be
        omitted if stored in the model.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
        Accepts `FluidPhase.WATER` or `FluidPhase.OIL` (or their
        string equivalents). When `None` (default), the value is inferred
        from `model.oil_water_table.wetting_phase` if that attribute exists,
        otherwise `FluidPhase.WATER` is used.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
        Accepts `FluidPhase.OIL` or `FluidPhase.GAS`. When
        `None` (default), inferred from `model.gas_oil_table.wetting_phase`
        if available, otherwise `FluidPhase.OIL`.
    :param oil_water_reference_phase: Saturation axis the oil-water sub-table
        is indexed by. `"wetting"` (default) indexes by the wetting-phase
        saturation; `"non_wetting"` indexes by the non-wetting-phase
        saturation.
    :param gas_oil_reference_phase: Saturation axis the gas-oil sub-table is
        indexed by. `"non_wetting"` (default) indexes by the non-wetting
        saturation (typically `Sg`); `"wetting"` indexes by the wetting
        saturation (typically `So`).
    :param n_points: Number of saturation grid points per sub-table. Must be
        ≥ 2. Default is 200.
    :param spacing: Grid spacing mode. `"cosine"` (default) uses
        Chebyshev-cosine spacing, which is denser at the endpoints and gives
        better accuracy for curved relperm profiles. `"linspace"` uses
        uniform spacing.
    :param oil_water_reference_saturation: Override the auto-generated
        oil-water saturation axis with a custom monotonically increasing array.
        When supplied, *n_points* and *spacing* are ignored for this sub-table.
    :param gas_oil_reference_saturation: Override the auto-generated gas-oil
        saturation axis with a custom monotonically increasing array.
    :param mixing_rule: Three-phase oil relative permeability mixing rule for
        the output `ThreePhaseRelPermTable`. When `None` (default),
        the source model's `mixing_rule` attribute is used if present,
        otherwise :data:`eclipse_rule`.
    :return: A `ThreePhaseRelPermTable` whose two-phase sub-tables are
        backed by piecewise-linear interpolation over the sampled grid.
    :raises ValueError: If a required residual saturation cannot be resolved
        from the arguments or the model.
    """
    if isinstance(model, ThreePhaseRelPermTable):
        return model

    # Residual saturations
    fn = "as_relperm_table"
    Swc = _resolve_arg(
        irreducible_water_saturation, model, "irreducible_water_saturation", fn
    )
    Sorw = _resolve_arg(
        residual_oil_saturation_water, model, "residual_oil_saturation_water", fn
    )
    Sorg = _resolve_arg(
        residual_oil_saturation_gas, model, "residual_oil_saturation_gas", fn
    )
    Sgr = _resolve_arg(residual_gas_saturation, model, "residual_gas_saturation", fn)

    residual_kwargs = dict(  # noqa
        irreducible_water_saturation=Swc,
        residual_oil_saturation_water=Sorw,
        residual_oil_saturation_gas=Sorg,
        residual_gas_saturation=Sgr,
    )
    call_kwargs = {
        **residual_kwargs,
        **(model_kwargs or {}),
    }

    # Wetting phases
    if oil_water_wetting_phase is not None:
        ow_wetting = FluidPhase(oil_water_wetting_phase)
    else:
        ow_wetting = model.get_oil_water_wetting_phase()

    if gas_oil_wetting_phase is not None:
        go_wetting = FluidPhase(gas_oil_wetting_phase)
    else:
        go_wetting = model.get_gas_oil_wetting_phase()

    ow_non_wetting = (
        FluidPhase.OIL if ow_wetting == FluidPhase.WATER else FluidPhase.WATER
    )
    go_non_wetting = FluidPhase.GAS if go_wetting == FluidPhase.OIL else FluidPhase.OIL

    # Oil-water sub-table
    # The sweep saturation is determined by which phase is wetting and which
    # axis is declared as the reference.  The non-swept saturation is fixed at
    # its connate/residual limit; Sg is always 0 for this sub-table.
    #
    # ow_sweep_is_sw = True  → sweep Sw, hold So = 1 - Sw, Sg = 0
    # ow_sweep_is_sw = False → sweep So, hold Sw = Swc,    Sg = 0

    ow_sweep_is_sw = (
        ow_wetting == FluidPhase.WATER and oil_water_reference_phase == "wetting"
    ) or (ow_wetting == FluidPhase.OIL and oil_water_reference_phase == "non_wetting")

    if oil_water_reference_saturation is not None:
        ow_ref = np.asarray(oil_water_reference_saturation, dtype=np.float64)
    else:
        if ow_sweep_is_sw:
            s_min_ow, s_max_ow = Swc, max(Swc + 1e-6, 1.0 - Sorw)
        else:
            s_min_ow, s_max_ow = Sorw, max(Sorw + 1e-6, 1.0 - Swc)
        ow_ref = make_saturation_grid(n_points, s_min_ow, s_max_ow, spacing)

    krw_ow = np.empty_like(ow_ref)
    kro_ow = np.empty_like(ow_ref)

    for idx, s in enumerate(ow_ref):
        if ow_sweep_is_sw:
            sw_val, so_val, sg_val = s, max(0.0, 1.0 - s), 0.0
        else:
            sw_val, so_val, sg_val = Swc, s, max(0.0, 1.0 - Swc - s)

        kr = model.get_relative_permeabilities(
            water_saturation=sw_val,
            oil_saturation=so_val,
            gas_saturation=sg_val,
            **call_kwargs,
        )
        krw_ow[idx] = float(kr["water"])
        kro_ow[idx] = float(kr["oil"])

    # Slot krw/kro into the correct wetting/non-wetting positions.
    if ow_wetting == FluidPhase.WATER:
        ow_wetting_kr, ow_non_wetting_kr = krw_ow, kro_ow
    else:
        # Oil-wet oil-water table: oil is wetting, water is non-wetting.
        ow_wetting_kr, ow_non_wetting_kr = kro_ow, krw_ow

    ow_table = TwoPhaseRelPermTable(
        wetting_phase=ow_wetting,
        non_wetting_phase=ow_non_wetting,
        reference_saturation=ow_ref,
        wetting_phase_relative_permeability=ow_wetting_kr,
        non_wetting_phase_relative_permeability=ow_non_wetting_kr,
        reference_phase=oil_water_reference_phase,
    )

    # Gas-oil sub-table
    # Sw is always fixed at Swc (connate water only) for this sub-table.
    #
    # go_sweep_is_sg = True  → sweep Sg, hold So = 1 - Swc - Sg
    # go_sweep_is_sg = False → sweep So, hold Sg = 1 - Swc - So

    go_sweep_is_sg = (
        go_wetting == FluidPhase.OIL and gas_oil_reference_phase == "non_wetting"
    ) or (go_wetting == FluidPhase.GAS and gas_oil_reference_phase == "wetting")

    if gas_oil_reference_saturation is not None:
        gas_oil_reference_saturation = np.asarray(
            gas_oil_reference_saturation, dtype=np.float64
        )
    else:
        if go_sweep_is_sg:
            s_min_go, s_max_go = Sgr, max(Sgr + 1e-6, 1.0 - Swc - Sorg)
        else:
            s_min_go, s_max_go = Sorg, max(Sorg + 1e-6, 1.0 - Swc - Sgr)
        gas_oil_reference_saturation = make_saturation_grid(
            n_points, s_min_go, s_max_go, spacing
        )

    krg_go = np.empty_like(gas_oil_reference_saturation)
    kro_go = np.empty_like(gas_oil_reference_saturation)

    for idx, s in enumerate(gas_oil_reference_saturation):
        if go_sweep_is_sg:
            # s is Sg; So fills the remaining mobile pore space above Swc.
            sw_val = Swc
            sg_val = s
            so_val = max(0.0, 1.0 - Swc - s)
        else:
            # s is So; Sg fills the remaining mobile pore space above Swc.
            sw_val = Swc
            so_val = s
            sg_val = max(0.0, 1.0 - Swc - s)

        kr = model.get_relative_permeabilities(
            water_saturation=sw_val,
            oil_saturation=so_val,
            gas_saturation=sg_val,
            **call_kwargs,
        )
        krg_go[idx] = float(kr["gas"])
        kro_go[idx] = float(kr["oil"])

    if go_wetting == FluidPhase.OIL:
        go_wetting_kr, go_non_wetting_kr = kro_go, krg_go
    else:
        # Gas-wet gas-oil table (uncommon): gas is wetting, oil is non-wetting.
        go_wetting_kr, go_non_wetting_kr = krg_go, kro_go

    go_table = TwoPhaseRelPermTable(
        wetting_phase=go_wetting,
        non_wetting_phase=go_non_wetting,
        reference_saturation=gas_oil_reference_saturation,
        wetting_phase_relative_permeability=go_wetting_kr,
        non_wetting_phase_relative_permeability=go_non_wetting_kr,
        reference_phase=gas_oil_reference_phase,
    )
    if mixing_rule is None:
        mixing_rule = getattr(model, "mixing_rule", "eclipse_rule")

    return ThreePhaseRelPermTable(
        oil_water_table=ow_table,
        gas_oil_table=go_table,
        mixing_rule=mixing_rule,
    )


def as_capillary_pressure_table(
    model: CapillaryPressureTable,
    *,
    irreducible_water_saturation: typing.Optional[float] = None,
    residual_oil_saturation_water: typing.Optional[float] = None,
    residual_oil_saturation_gas: typing.Optional[float] = None,
    residual_gas_saturation: typing.Optional[float] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    oil_water_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    gas_oil_wetting_phase: typing.Optional[typing.Union[FluidPhase, str]] = None,
    oil_water_reference_phase: typing.Literal["wetting", "non_wetting"] = "wetting",
    gas_oil_reference_phase: typing.Literal["wetting", "non_wetting"] = "non_wetting",
    n_points: int = 200,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    gas_oil_reference_saturation: typing.Optional[npt.ArrayLike] = None,
) -> ThreePhaseCapillaryPressureTable:
    """
    Convert any `CapillaryPressureTable` to a
    `ThreePhaseCapillaryPressureTable` backed by fast piecewise-linear
    `TwoPhaseCapillaryPressureTable` instances.

    If *model* is already a `ThreePhaseCapillaryPressureTable` it is
    returned unchanged (no-op).

    **How sampling works**

    Two two-phase sub-tables are built following the same sweep logic as
    :func:`as_relperm_table`.  Only the Pc component relevant to each
    sub-table is recorded:

    *Oil-water sub-table* — `Pcow = Po - Pw` is sampled with `Sg = 0`
    throughout.  The swept saturation axis and the fixed saturation are
    determined by *oil_water_wetting_phase* and *oil_water_reference_phase*
    exactly as described in :func:`as_relperm_table`.

    *Gas-oil sub-table* — `Pcgo = Pg - Po` is sampled with `Sw = Swc`
    throughout.  The swept saturation axis is determined by
    *gas_oil_wetting_phase* and *gas_oil_reference_phase*.

    :param model: Source analytical (or tabular) capillary pressure model.
    :param irreducible_water_saturation: Irreducible (connate) water saturation
        `Swc`. May be omitted if stored in the model.
    :param residual_oil_saturation_water: Residual oil saturation after water
        flood `Sorw`. May be omitted if stored in the model.
    :param residual_oil_saturation_gas: Residual oil saturation after gas flood
        `Sorg`. May be omitted if stored in the model.
    :param residual_gas_saturation: Residual gas saturation `Sgr`. May be
        omitted if stored in the model.
    :param model_kwargs: Extra keyword arguments forwarded verbatim to each
        :meth:`get_capillary_pressures` call. Useful for models like
        `LeverettJCapillaryPressureModel` that additionally require
        `permeability` and `porosity`.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
        Accepts `FluidPhase.WATER` or `FluidPhase.OIL` (or their
        string equivalents). When `None` (default), inferred from
        `model.oil_water_table.wetting_phase` if available, otherwise
        `FluidPhase.WATER`.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
        Accepts `FluidPhase.OIL` or `FluidPhase.GAS`. When
        `None` (default), inferred from `model.gas_oil_table.wetting_phase`
        if available, otherwise `FluidPhase.OIL`.
    :param oil_water_reference_phase: Saturation axis for the oil-water
        sub-table. `"wetting"` (default) indexes by the wetting-phase
        saturation; `"non_wetting"` indexes by the non-wetting-phase
        saturation.
    :param gas_oil_reference_phase: Saturation axis for the gas-oil sub-table.
        `"non_wetting"` (default) indexes by the non-wetting saturation
        (typically `Sg`); `"wetting"` indexes by the wetting saturation
        (typically `So`).
    :param n_points: Number of saturation grid points per sub-table. Must be
        ≥ 2. Default is 200.
    :param spacing: Grid spacing mode. `"cosine"` (default) uses
        Chebyshev-cosine spacing, denser at the endpoints. `"linspace"`
        uses uniform spacing.
    :param oil_water_reference_saturation: Override the auto-generated
        oil-water saturation axis with a custom monotonically increasing array.
        When supplied, *n_points* and *spacing* are ignored for this sub-table.
    :param gas_oil_reference_saturation: Override the auto-generated gas-oil
        saturation axis with a custom monotonically increasing array.
    :return: A `ThreePhaseCapillaryPressureTable` backed by
        piecewise-linear interpolation over the sampled grid.
    :raises ValueError: If a required residual saturation cannot be resolved
        from the arguments or the model.
    """
    if isinstance(model, ThreePhaseCapillaryPressureTable):
        return model

    fn = "as_capillary_pressure_table"
    Swc = _resolve_arg(
        irreducible_water_saturation, model, "irreducible_water_saturation", fn
    )
    Sorw = _resolve_arg(
        residual_oil_saturation_water, model, "residual_oil_saturation_water", fn
    )
    Sorg = _resolve_arg(
        residual_oil_saturation_gas, model, "residual_oil_saturation_gas", fn
    )
    Sgr = _resolve_arg(residual_gas_saturation, model, "residual_gas_saturation", fn)

    residual_kwargs = dict(  # noqa
        irreducible_water_saturation=Swc,
        residual_oil_saturation_water=Sorw,
        residual_oil_saturation_gas=Sorg,
        residual_gas_saturation=Sgr,
    )
    call_kwargs = {
        **residual_kwargs,
        **(model_kwargs or {}),
    }

    # Wetting phases
    if oil_water_wetting_phase is not None:
        ow_wetting = FluidPhase(oil_water_wetting_phase)
    else:
        ow_wetting = model.get_oil_water_wetting_phase()

    if gas_oil_wetting_phase is not None:
        go_wetting = FluidPhase(gas_oil_wetting_phase)
    else:
        go_wetting = model.get_gas_oil_wetting_phase()

    ow_non_wetting = (
        FluidPhase.OIL if ow_wetting == FluidPhase.WATER else FluidPhase.WATER
    )
    go_non_wetting = FluidPhase.GAS if go_wetting == FluidPhase.OIL else FluidPhase.OIL

    # Oil-water sub-table  (same sweep logic as as_relperm_table)
    ow_sweep_is_sw = (
        ow_wetting == FluidPhase.WATER and oil_water_reference_phase == "wetting"
    ) or (ow_wetting == FluidPhase.OIL and oil_water_reference_phase == "non_wetting")

    if oil_water_reference_saturation is not None:
        ow_ref = np.asarray(oil_water_reference_saturation, dtype=np.float64)
    else:
        if ow_sweep_is_sw:
            s_min_ow, s_max_ow = Swc, max(Swc + 1e-6, 1.0 - Sorw)
        else:
            s_min_ow, s_max_ow = Sorw, max(Sorw + 1e-6, 1.0 - Swc)
        ow_ref = make_saturation_grid(n_points, s_min_ow, s_max_ow, spacing)

    pcow_vals = np.empty_like(ow_ref)

    for idx, s in enumerate(ow_ref):
        if ow_sweep_is_sw:
            sw_val, so_val, sg_val = s, max(0.0, 1.0 - s), 0.0
        else:
            sw_val, so_val, sg_val = Swc, s, max(0.0, 1.0 - Swc - s)

        pc = model.get_capillary_pressures(
            water_saturation=sw_val,
            oil_saturation=so_val,
            gas_saturation=sg_val,
            **call_kwargs,
        )
        pcow_vals[idx] = float(pc["oil_water"])

    oil_water_capillary_pressure_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=ow_wetting,
        non_wetting_phase=ow_non_wetting,
        reference_saturation=ow_ref,
        capillary_pressure=pcow_vals,
        reference_phase=oil_water_reference_phase,
    )

    # Gas-oil sub-table
    go_sweep_is_sg = (
        go_wetting == FluidPhase.OIL and gas_oil_reference_phase == "non_wetting"
    ) or (go_wetting == FluidPhase.GAS and gas_oil_reference_phase == "wetting")

    if gas_oil_reference_saturation is not None:
        gas_oil_reference_saturation = np.asarray(
            gas_oil_reference_saturation, dtype=np.float64
        )
    else:
        if go_sweep_is_sg:
            s_min_go, s_max_go = Sgr, max(Sgr + 1e-6, 1.0 - Swc - Sorg)
        else:
            s_min_go, s_max_go = Sorg, max(Sorg + 1e-6, 1.0 - Swc - Sgr)
        gas_oil_reference_saturation = make_saturation_grid(
            n_points, s_min_go, s_max_go, spacing
        )

    pcgo_values = np.empty_like(gas_oil_reference_saturation)

    for idx, s in enumerate(gas_oil_reference_saturation):
        if go_sweep_is_sg:
            # s is Sg; So fills the remaining mobile pore space above Swc.
            sw_val = Swc
            sg_val = s
            so_val = max(0.0, 1.0 - Swc - s)
        else:
            # s is So; Sg fills the remaining mobile pore space above Swc.
            sw_val = Swc
            so_val = s
            sg_val = max(0.0, 1.0 - Swc - s)

        capillary_pressures = model.get_capillary_pressures(
            water_saturation=sw_val,
            oil_saturation=so_val,
            gas_saturation=sg_val,
            **call_kwargs,
        )
        pcgo_values[idx] = float(capillary_pressures["gas_oil"])

    gas_oil_capillary_pressure_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=go_wetting,
        non_wetting_phase=go_non_wetting,
        reference_saturation=gas_oil_reference_saturation,
        capillary_pressure=pcgo_values,
        reference_phase=gas_oil_reference_phase,
    )
    return ThreePhaseCapillaryPressureTable(
        oil_water_table=oil_water_capillary_pressure_table,
        gas_oil_table=gas_oil_capillary_pressure_table,
    )
