import typing

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

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

    Made up of a relative permeability table and an optional capillary pressure
    table.
    """

    relative_permeability_table: RelativePermeabilityTable
    capillary_pressure_table: typing.Optional[CapillaryPressureTable] = None

    def get_relative_permeabilities(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
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
    if arg is not None:
        return float(arg)
    v = getattr(model, attr, None)
    if v is None:
        raise ValueError(
            f"'{attr}' must be supplied either as an argument to "
            f"{fn_name}() or stored in the model."
        )
    return float(v)


def _safe_grid(
    n_points: int,
    s_min: float,
    s_max: float,
    spacing: Spacing,
    *,
    min_span: float = 1e-6,
) -> npt.NDArray:
    if s_max - s_min < min_span:
        return np.array([s_min, max(s_min + min_span, s_max)], dtype=np.float64)
    return make_saturation_grid(n_points, s_min, s_max, spacing)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _pchip_resample(
    x: npt.NDArray,
    y: npt.NDArray,
    n_out: int,
    spacing: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Fit a PCHIP interpolant through (x, y), resample at n_out points, and
    return (x_new, y_new, dy_new) where dy_new is the derivative of the
    interpolant at each new point.

    Returning the derivative alongside the values means the caller can
    populate both the value and derivative arrays of the two-phase table in
    one shot, giving the Newton solver a C¹-consistent Jacobian.
    """
    interp = PchipInterpolator(x, y)
    x_new = make_saturation_grid(n_out, float(x[0]), float(x[-1]), spacing)
    return x_new, interp(x_new), interp(x_new, 1)


def _build_ref_grid(
    n_points: int,
    s_lo: float,
    s_hi: float,
    spacing: Spacing,
    *,
    n_endpoint_extra: int = 20,
) -> npt.NDArray:
    base = _safe_grid(n_points, s_lo, s_hi, spacing)
    span = s_hi - s_lo
    if span < 1e-6 or n_endpoint_extra <= 0:
        return base
    decade = 0.10 * span
    lo_extra = np.linspace(s_lo, s_lo + decade, n_endpoint_extra + 2)
    hi_extra = np.linspace(s_hi - decade, s_hi, n_endpoint_extra + 2)
    return np.unique(np.concatenate([base, lo_extra, hi_extra]))


def _ow_point_sweep_sw(
    s: float, Swc: float, Sorw: float
) -> typing.Tuple[float, float, float]:
    sw = _clamp(s, Swc, 1.0 - Sorw)
    so = _clamp(1.0 - sw, 0.0, 1.0 - Swc)
    return sw, so, 0.0


def _ow_point_sweep_so(
    s: float, Swc: float, Sorw: float
) -> typing.Tuple[float, float, float]:
    so = _clamp(s, Sorw, 1.0 - Swc)
    sw = _clamp(Swc, 0.0, 1.0 - so)
    return sw, so, 0.0


def _go_point_sweep_sg(
    s: float, Swc: float, Sorg: float, Sgr: float
) -> typing.Tuple[float, float, float]:
    sg = _clamp(s, Sgr, 1.0 - Swc - Sorg)
    so = _clamp(1.0 - Swc - sg, 0.0, 1.0 - Swc)
    return Swc, so, sg


def _go_point_sweep_so(
    s: float, Swc: float, Sorg: float, Sgr: float
) -> typing.Tuple[float, float, float]:
    so = _clamp(s, Sorg, 1.0 - Swc - Sgr)
    sg = _clamp(1.0 - Swc - so, 0.0, 1.0 - Swc)
    return Swc, so, sg


def _oil_water_sweep_is_sw(
    wetting: FluidPhase, reference_phase: typing.Literal["wetting", "non_wetting"]
) -> bool:
    return (wetting == FluidPhase.WATER) == (reference_phase == "wetting")


def _gas_oil_sweep_is_sg(
    wetting: FluidPhase, reference_phase: typing.Literal["wetting", "non_wetting"]
) -> bool:
    return (wetting == FluidPhase.GAS) == (reference_phase == "wetting")


def _has_relperm_derivatives(model: RelativePermeabilityTable) -> bool:
    return callable(getattr(model, "get_relative_permeability_derivatives", None))


def _has_capillary_pressure_derivatives(model: CapillaryPressureTable) -> bool:
    return callable(getattr(model, "get_capillary_pressure_derivatives", None))


def _sample_oil_water_relperm(
    model: RelativePermeabilityTable,
    ow_ref: npt.NDArray,
    sweep_sw: bool,
    Swc: float,
    Sorw: float,
    call_kwargs: typing.Dict[str, typing.Any],
    ow_wetting: FluidPhase,
    n_points: int,
    spacing: Spacing,
) -> typing.Tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    typing.Optional[npt.NDArray],
    typing.Optional[npt.NDArray],
]:
    """
    Sample oil-water kr values and, when the model exposes analytical
    derivatives, sample those too so the two-phase table can serve a
    smooth Jacobian to the Newton solver.

    For tabular source models (ThreePhaseRelPermTable) we PCHIP-resample
    the existing knots directly, recovering both values and derivatives
    from the cubic interpolant in one pass.
    """
    if isinstance(model, ThreePhaseRelPermTable):
        ow_table = model.oil_water_table
        x_new, wetting_kr, d_wetting_kr = _pchip_resample(
            ow_table.reference_saturation,
            ow_table.wetting_phase_relative_permeability,
            n_points,
            spacing,
        )
        _, non_wetting_kr, d_non_wetting_kr = _pchip_resample(
            ow_table.reference_saturation,
            ow_table.non_wetting_phase_relative_permeability,
            n_points,
            spacing,
        )
        return x_new, wetting_kr, non_wetting_kr, d_wetting_kr, d_non_wetting_kr

    krw = np.empty(len(ow_ref))
    kro = np.empty(len(ow_ref))
    d_krw: typing.Optional[npt.NDArray] = (
        np.empty(len(ow_ref)) if _has_relperm_derivatives(model) else None
    )
    d_kro: typing.Optional[npt.NDArray] = (
        np.empty(len(ow_ref)) if _has_relperm_derivatives(model) else None
    )

    for idx, s in enumerate(ow_ref):
        sw_v, so_v, sg_v = (
            _ow_point_sweep_sw(float(s), Swc, Sorw)
            if sweep_sw
            else _ow_point_sweep_so(float(s), Swc, Sorw)
        )
        kr = model.get_relative_permeabilities(
            water_saturation=sw_v,
            oil_saturation=so_v,
            gas_saturation=sg_v,
            **call_kwargs,
        )
        krw[idx] = float(kr["water"])
        kro[idx] = float(kr["oil"])

        if d_krw is not None:
            dkr = model.get_relative_permeability_derivatives(  # type: ignore[union-attr]
                water_saturation=sw_v,
                oil_saturation=so_v,
                gas_saturation=sg_v,
                **call_kwargs,
            )
            # derivative w.r.t. the reference axis
            d_krw[idx] = float(dkr["dKrw_dSw"] if sweep_sw else dkr["dKrw_dSo"])
            d_kro[idx] = float(dkr["dKro_dSw"] if sweep_sw else dkr["dKro_dSo"])  # type: ignore[index]

    if ow_wetting == FluidPhase.WATER:
        return ow_ref, krw, kro, d_krw, d_kro
    return ow_ref, kro, krw, d_kro, d_krw  # type: ignore[return-value]


def _sample_gas_oil_relperm(
    model: RelativePermeabilityTable,
    go_ref: npt.NDArray,
    sweep_sg: bool,
    Swc: float,
    Sorg: float,
    Sgr: float,
    call_kwargs: typing.Dict[str, typing.Any],
    go_wetting: FluidPhase,
    n_points: int,
    spacing: Spacing,
) -> typing.Tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    typing.Optional[npt.NDArray],
    typing.Optional[npt.NDArray],
]:
    """
    Sample gas-oil kr values and analytical derivatives when available.
    PCHIP fast-path for tabular source models.
    """
    if isinstance(model, ThreePhaseRelPermTable):
        go_table = model.gas_oil_table
        x_new, wetting_kr, d_wetting_kr = _pchip_resample(
            go_table.reference_saturation,
            go_table.wetting_phase_relative_permeability,
            n_points,
            spacing,
        )
        _, non_wetting_kr, d_non_wetting_kr = _pchip_resample(
            go_table.reference_saturation,
            go_table.non_wetting_phase_relative_permeability,
            n_points,
            spacing,
        )
        return x_new, wetting_kr, non_wetting_kr, d_wetting_kr, d_non_wetting_kr

    krg = np.empty(len(go_ref))
    kro = np.empty(len(go_ref))
    d_krg: typing.Optional[npt.NDArray] = (
        np.empty(len(go_ref)) if _has_relperm_derivatives(model) else None
    )
    d_kro: typing.Optional[npt.NDArray] = (
        np.empty(len(go_ref)) if _has_relperm_derivatives(model) else None
    )

    for idx, s in enumerate(go_ref):
        sw_v, so_v, sg_v = (
            _go_point_sweep_sg(float(s), Swc, Sorg, Sgr)
            if sweep_sg
            else _go_point_sweep_so(float(s), Swc, Sorg, Sgr)
        )
        kr = model.get_relative_permeabilities(
            water_saturation=sw_v,
            oil_saturation=so_v,
            gas_saturation=sg_v,
            **call_kwargs,
        )
        krg[idx] = float(kr["gas"])
        kro[idx] = float(kr["oil"])

        if d_krg is not None:
            dkr = model.get_relative_permeability_derivatives(  # type: ignore[union-attr]
                water_saturation=sw_v,
                oil_saturation=so_v,
                gas_saturation=sg_v,
                **call_kwargs,
            )
            d_krg[idx] = float(dkr["dKrg_dSg"] if sweep_sg else dkr["dKrg_dSo"])
            d_kro[idx] = float(dkr["dKro_dSg"] if sweep_sg else dkr["dKro_dSo"])  # type: ignore[index]

    if go_wetting == FluidPhase.OIL:
        return go_ref, kro, krg, d_kro, d_krg
    return go_ref, krg, kro, d_krg, d_kro  # type: ignore[return-value]


def _sample_oil_water_capillary_pressure(
    model: CapillaryPressureTable,
    ow_ref: npt.NDArray,
    sweep_sw: bool,
    Swc: float,
    Sorw: float,
    call_kwargs: typing.Dict[str, typing.Any],
    n_points: int,
    spacing: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray, typing.Optional[npt.NDArray]]:
    """
    Sample Pcow and its analytical derivative when available.
    PCHIP fast-path for tabular source models.
    """
    if isinstance(model, ThreePhaseCapillaryPressureTable):
        ow_table = model.oil_water_table
        x_new, pcow, d_pcow = _pchip_resample(
            ow_table.reference_saturation,
            ow_table.capillary_pressure,
            n_points,
            spacing,
        )
        return x_new, pcow, d_pcow

    pcow = np.empty(len(ow_ref))
    d_pcow: typing.Optional[npt.NDArray] = (
        np.empty(len(ow_ref)) if _has_capillary_pressure_derivatives(model) else None
    )

    for idx, s in enumerate(ow_ref):
        sw_v, so_v, sg_v = (
            _ow_point_sweep_sw(float(s), Swc, Sorw)
            if sweep_sw
            else _ow_point_sweep_so(float(s), Swc, Sorw)
        )
        pc = model.get_capillary_pressures(
            water_saturation=sw_v,
            oil_saturation=so_v,
            gas_saturation=sg_v,
            **call_kwargs,
        )
        pcow[idx] = float(pc["oil_water"])

        if d_pcow is not None:
            dpc = model.get_capillary_pressure_derivatives(  # type: ignore[union-attr]
                water_saturation=sw_v,
                oil_saturation=so_v,
                gas_saturation=sg_v,
                **call_kwargs,
            )
            d_pcow[idx] = float(dpc["dPcow_dSw"] if sweep_sw else dpc["dPcow_dSo"])

    return ow_ref, pcow, d_pcow


def _sample_gas_oil_capillary_pressure(
    model: CapillaryPressureTable,
    go_ref: npt.NDArray,
    sweep_sg: bool,
    Swc: float,
    Sorg: float,
    Sgr: float,
    call_kwargs: typing.Dict[str, typing.Any],
    n_points: int,
    spacing: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray, typing.Optional[npt.NDArray]]:
    """
    Sample Pcgo and its analytical derivative when available.
    PCHIP fast-path for tabular source models.
    """
    if isinstance(model, ThreePhaseCapillaryPressureTable):
        go_table = model.gas_oil_table
        x_new, pcgo, d_pcgo = _pchip_resample(
            go_table.reference_saturation,
            go_table.capillary_pressure,
            n_points,
            spacing,
        )
        return x_new, pcgo, d_pcgo

    pcgo = np.empty(len(go_ref))
    d_pcgo: typing.Optional[npt.NDArray] = (
        np.empty(len(go_ref)) if _has_capillary_pressure_derivatives(model) else None
    )

    for idx, s in enumerate(go_ref):
        sw_v, so_v, sg_v = (
            _go_point_sweep_sg(float(s), Swc, Sorg, Sgr)
            if sweep_sg
            else _go_point_sweep_so(float(s), Swc, Sorg, Sgr)
        )
        pc = model.get_capillary_pressures(
            water_saturation=sw_v,
            oil_saturation=so_v,
            gas_saturation=sg_v,
            **call_kwargs,
        )
        pcgo[idx] = float(pc["gas_oil"])

        if d_pcgo is not None:
            dpc = model.get_capillary_pressure_derivatives(  # type: ignore[union-attr]
                water_saturation=sw_v,
                oil_saturation=so_v,
                gas_saturation=sg_v,
                **call_kwargs,
            )
            d_pcgo[idx] = float(dpc["dPcgo_dSg"] if sweep_sg else dpc["dPcgo_dSo"])

    return go_ref, pcgo, d_pcgo


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
    n_endpoint_extra: int = 20,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    gas_oil_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    mixing_rule: typing.Optional[typing.Union[MixingRule, str]] = None,
) -> ThreePhaseRelPermTable:
    """
    Convert any `RelativePermeabilityTable` to a `ThreePhaseRelPermTable`
    backed by piecewise-linear `TwoPhaseRelPermTable` instances.

    Returns the model unchanged if it is already a `ThreePhaseRelPermTable`.

    When the source model exposes `get_relative_permeability_derivatives`,
    analytical derivatives are sampled at every knot and stored in the
    two-phase sub-tables so that `get_*_derivative` returns smooth,
    consistent values instead of piecewise-linear slopes.

    For tabular source models (`ThreePhaseRelPermTable`) the existing knots
    are PCHIP-resampled to the denser output grid, recovering C¹-continuous
    values and derivatives without any additional model calls.

    :param model: Source analytical or tabular relative permeability model.
    :param irreducible_water_saturation: Irreducible water saturation *Swc*.
    :param residual_oil_saturation_water: Residual oil saturation *Sorw*.
    :param residual_oil_saturation_gas: Residual oil saturation *Sorg*.
    :param residual_gas_saturation: Residual gas saturation *Sgr*.
    :param model_kwargs: Extra kwargs forwarded to every model evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
    :param oil_water_reference_phase: Reference saturation axis for the oil-water table.
    :param gas_oil_reference_phase: Reference saturation axis for the gas-oil table.
    :param n_points: Base number of sample points per sub-table.
    :param n_endpoint_extra: Extra knots in each boundary decade (set 0 to disable).
    :param spacing: Grid spacing mode.
    :param oil_water_reference_saturation: Custom saturation axis for the oil-water sub-table.
    :param gas_oil_reference_saturation: Custom saturation axis for the gas-oil sub-table.
    :param mixing_rule: Three-phase oil mixing rule.
    :return: `ThreePhaseRelPermTable` with piecewise-linear sub-tables.
    """
    if isinstance(model, ThreePhaseRelPermTable):
        return model

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

    call_kwargs = {
        "irreducible_water_saturation": Swc,
        "residual_oil_saturation_water": Sorw,
        "residual_oil_saturation_gas": Sorg,
        "residual_gas_saturation": Sgr,
        **(model_kwargs or {}),
    }

    ow_wetting: FluidPhase = (
        FluidPhase(oil_water_wetting_phase)
        if oil_water_wetting_phase is not None
        else model.get_oil_water_wetting_phase()
    )
    go_wetting: FluidPhase = (
        FluidPhase(gas_oil_wetting_phase)
        if gas_oil_wetting_phase is not None
        else model.get_gas_oil_wetting_phase()
    )
    ow_non_wetting = (
        FluidPhase.OIL if ow_wetting == FluidPhase.WATER else FluidPhase.WATER
    )
    go_non_wetting = FluidPhase.GAS if go_wetting == FluidPhase.OIL else FluidPhase.OIL

    sweep_sw = _oil_water_sweep_is_sw(ow_wetting, oil_water_reference_phase)
    sweep_sg = _gas_oil_sweep_is_sg(go_wetting, gas_oil_reference_phase)

    if oil_water_reference_saturation is not None:
        ow_ref = np.asarray(oil_water_reference_saturation, dtype=np.float64)
    else:
        ow_lo, ow_hi = (Swc, 1.0 - Sorw) if sweep_sw else (Sorw, 1.0 - Swc)
        ow_ref = _build_ref_grid(
            n_points, ow_lo, ow_hi, spacing, n_endpoint_extra=n_endpoint_extra
        )

    if gas_oil_reference_saturation is not None:
        go_ref = np.asarray(gas_oil_reference_saturation, dtype=np.float64)
    else:
        go_lo, go_hi = (Sgr, 1.0 - Swc - Sorg) if sweep_sg else (Sorg, 1.0 - Swc - Sgr)
        go_ref = _build_ref_grid(
            n_points, go_lo, go_hi, spacing, n_endpoint_extra=n_endpoint_extra
        )

    ow_ref, ow_wetting_kr, ow_non_wetting_kr, d_ow_wetting, d_ow_non_wetting = (
        _sample_oil_water_relperm(
            model,
            ow_ref,
            sweep_sw,
            Swc,
            Sorw,
            call_kwargs,
            ow_wetting,
            n_points,
            spacing,
        )
    )
    go_ref, go_wetting_kr, go_non_wetting_kr, d_go_wetting, d_go_non_wetting = (
        _sample_gas_oil_relperm(
            model,
            go_ref,
            sweep_sg,
            Swc,
            Sorg,
            Sgr,
            call_kwargs,
            go_wetting,
            n_points,
            spacing,
        )
    )

    ow_table = TwoPhaseRelPermTable(
        wetting_phase=ow_wetting,
        non_wetting_phase=ow_non_wetting,
        reference_saturation=ow_ref,
        wetting_phase_relative_permeability=ow_wetting_kr,
        non_wetting_phase_relative_permeability=ow_non_wetting_kr,
        reference_phase=oil_water_reference_phase,
        wetting_phase_relative_permeability_derivative=d_ow_wetting,
        non_wetting_phase_relative_permeability_derivative=d_ow_non_wetting,
    )
    go_table = TwoPhaseRelPermTable(
        wetting_phase=go_wetting,
        non_wetting_phase=go_non_wetting,
        reference_saturation=go_ref,
        wetting_phase_relative_permeability=go_wetting_kr,
        non_wetting_phase_relative_permeability=go_non_wetting_kr,
        reference_phase=gas_oil_reference_phase,
        wetting_phase_relative_permeability_derivative=d_go_wetting,
        non_wetting_phase_relative_permeability_derivative=d_go_non_wetting,
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
    n_endpoint_extra: int = 30,
    spacing: Spacing = "cosine",
    oil_water_reference_saturation: typing.Optional[npt.ArrayLike] = None,
    gas_oil_reference_saturation: typing.Optional[npt.ArrayLike] = None,
) -> ThreePhaseCapillaryPressureTable:
    """
    Convert any `CapillaryPressureTable` to a `ThreePhaseCapillaryPressureTable` backed by piecewise-linear
    `TwoPhaseCapillaryPressureTable` instances.

    Returns the model unchanged if it is already a `ThreePhaseCapillaryPressureTable`.

    When the source model exposes `get_capillary_pressure_derivatives`,
    analytical derivatives are sampled at every knot and stored in the
    two-phase sub-tables. The default `n_endpoint_extra=30` (vs 20 for
    relperm) reflects that Pc curves are unbounded near residual saturation,
    making endpoint fidelity especially important for SI convergence.

    :param model: Source analytical or tabular capillary pressure model.
    :param irreducible_water_saturation: *Swc*.
    :param residual_oil_saturation_water: *Sorw*.
    :param residual_oil_saturation_gas: *Sorg*.
    :param residual_gas_saturation: *Sgr*.
    :param model_kwargs: Extra kwargs forwarded to every model evaluation call.
    :param oil_water_wetting_phase: Wetting phase for the oil-water sub-table.
    :param gas_oil_wetting_phase: Wetting phase for the gas-oil sub-table.
    :param oil_water_reference_phase: Reference saturation axis for the oil-water table.
    :param gas_oil_reference_phase: Reference saturation axis for the gas-oil table.
    :param n_points: Base number of sample points per sub-table.
    :param n_endpoint_extra: Extra knots in each boundary decade.
    :param spacing: Grid spacing mode.
    :param oil_water_reference_saturation: Custom saturation axis for the oil-water sub-table.
    :param gas_oil_reference_saturation: Custom saturation axis for the gas-oil sub-table.
    :return: `ThreePhaseCapillaryPressureTable` backed by piecewise-linear sub-tables.
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

    call_kwargs = {
        "irreducible_water_saturation": Swc,
        "residual_oil_saturation_water": Sorw,
        "residual_oil_saturation_gas": Sorg,
        "residual_gas_saturation": Sgr,
        **(model_kwargs or {}),
    }

    ow_wetting: FluidPhase = (
        FluidPhase(oil_water_wetting_phase)
        if oil_water_wetting_phase is not None
        else model.get_oil_water_wetting_phase()
    )
    go_wetting: FluidPhase = (
        FluidPhase(gas_oil_wetting_phase)
        if gas_oil_wetting_phase is not None
        else model.get_gas_oil_wetting_phase()
    )
    ow_non_wetting = (
        FluidPhase.OIL if ow_wetting == FluidPhase.WATER else FluidPhase.WATER
    )
    go_non_wetting = FluidPhase.GAS if go_wetting == FluidPhase.OIL else FluidPhase.OIL

    sweep_sw = _oil_water_sweep_is_sw(ow_wetting, oil_water_reference_phase)
    sweep_sg = _gas_oil_sweep_is_sg(go_wetting, gas_oil_reference_phase)

    if oil_water_reference_saturation is not None:
        ow_ref = np.asarray(oil_water_reference_saturation, dtype=np.float64)
    else:
        ow_lo, ow_hi = (Swc, 1.0 - Sorw) if sweep_sw else (Sorw, 1.0 - Swc)
        ow_ref = _build_ref_grid(
            n_points, ow_lo, ow_hi, spacing, n_endpoint_extra=n_endpoint_extra
        )

    if gas_oil_reference_saturation is not None:
        go_ref = np.asarray(gas_oil_reference_saturation, dtype=np.float64)
    else:
        go_lo, go_hi = (Sgr, 1.0 - Swc - Sorg) if sweep_sg else (Sorg, 1.0 - Swc - Sgr)
        go_ref = _build_ref_grid(
            n_points, go_lo, go_hi, spacing, n_endpoint_extra=n_endpoint_extra
        )

    ow_ref, pcow_vals, d_pcow = _sample_oil_water_capillary_pressure(
        model,
        ow_ref,
        sweep_sw,
        Swc,
        Sorw,
        call_kwargs,
        n_points,
        spacing,
    )
    go_ref, pcgo_vals, d_pcgo = _sample_gas_oil_capillary_pressure(
        model,
        go_ref,
        sweep_sg,
        Swc,
        Sorg,
        Sgr,
        call_kwargs,
        n_points,
        spacing,
    )

    ow_pc_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=ow_wetting,
        non_wetting_phase=ow_non_wetting,
        reference_saturation=ow_ref,
        capillary_pressure=pcow_vals,
        reference_phase=oil_water_reference_phase,
        capillary_pressure_derivative=d_pcow,
    )
    go_pc_table = TwoPhaseCapillaryPressureTable(
        wetting_phase=go_wetting,
        non_wetting_phase=go_non_wetting,
        reference_saturation=go_ref,
        capillary_pressure=pcgo_vals,
        reference_phase=gas_oil_reference_phase,
        capillary_pressure_derivative=d_pcgo,
    )
    return ThreePhaseCapillaryPressureTable(
        oil_water_table=ow_pc_table,
        gas_oil_table=go_pc_table,
    )
