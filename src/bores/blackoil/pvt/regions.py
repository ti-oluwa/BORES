import logging
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import (  # type: ignore[import-untyped]
    PchipInterpolator,
    interp1d,
)
from typing_extensions import Self

from bores.blackoil.pvt.data import PVTData, PVTDataSet, get_stb_to_volume_factor
from bores.blackoil.pvt.static import StaticPVT
from bores.blackoil.pvt.tables import PVTTables, clip_compressibility
from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.reservoir.temperature import (
    Temperature,
    TemperatureGradient,
    TemperatureSpec,
    TemperatureTable,
)
from bores.types import (
    FloatArray,
    FluidPhase,
    InterpolationMethod,
    Number,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitConversionTable,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = ["PVT", "load_pvt_regions"]


@attrs.frozen(slots=True)
class PVTRegion:
    """A collection of PVT tables and static properties for a single Eclipse PVT region."""

    static: StaticPVT
    """Static PVT properties for this region (e.g. DENSITY, VISCOSITY)."""

    tables: PVTTables
    """Dynamic PVT tables for this region (e.g. PVTO, PVTG, PVTW)."""

    unit_system: UnitSystem
    """Unit system of the tables and static properties."""

    def __attrs_post_init__(self) -> None:
        if self.static.unit_system != self.unit_system:
            raise ValidationError(
                f"Static PVT unit system {self.static.unit_system} does not match "
                f"region unit system {self.unit_system}."
            )
        if self.tables.unit_system != self.unit_system:
            raise ValidationError(
                f"Tables unit system {self.tables.unit_system} does not match "
                f"region unit system {self.unit_system}."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVTRegion` with all region tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTRegion` in *target* units.
        """
        return attrs.evolve(
            self,
            static=self.static.convert(target, table=table),
            tables=self.tables.convert(target, table=table),
            unit_system=target,
        )


class PVT:
    """
    Multi-region PVT tables keyed by 1-based `PVTNUM` region index.

    Eclipse supports multiple PVT regions via the `PVTNUM` keyword. Each
    cell is assigned a region index and its PVT properties are evaluated from
    the corresponding `PVTTables` instance.

    Use `region(pvtnum)` to retrieve the tables and static properties for a given region, and
    `from_deck` to construct from a deck.

    Example:

    ```python
    pvt = PVT.from_deck(deck_file, temperature=200.0)
    region = pvt.region(pvtnum_array[cell_idx])
    bo = region.tables.oil.formation_volume_factor(p, t)
    ```
    """

    __slots__ = ("regions", "unit_system")

    def __init__(
        self,
        regions: dict[int, PVTRegion],
        *,
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Build a `PVT` from a pre-built regions dict.

        :param regions: Mapping from 1-based PVTNUM index to `PVTRegion`.
        :param unit_system: Expected unit system for all regions. If omitted,
            it is inferred from the first region and every other region is
            required to match it.
        :raises ValidationError: If *regions* is empty, or if any region's
            unit system does not match *unit_system* (explicit or inferred).
        """
        if not regions:
            raise ValidationError("`regions` must contain at least one entry.")

        expected_unit_system = unit_system or next(iter(regions.values())).unit_system
        mismatched = {
            pvtnum: region.unit_system
            for pvtnum, region in regions.items()
            if region.unit_system != expected_unit_system
        }
        if mismatched:
            raise ValidationError(
                f"All PVT regions must share `{expected_unit_system.value!r}` as "
                f"`{self.__class__.__name__}.unit_system`; mismatches "
                f"(pvtnum -> unit_system): "
                f"{ {k: v.value for k, v in mismatched.items()} }."
            )

        self.regions = regions
        self.unit_system = expected_unit_system

    def region(self, pvtnum: int) -> PVTRegion:
        """
        Return the `PVTRegion` for a given 1-based region index.

        :param pvtnum: 1-based PVT region index.
        :returns: `PVTRegion` for that region.
        :raises KeyError: If the region index does not exist.
        """
        region = self.regions.get(pvtnum)
        if region is None:
            available = sorted(self.regions.keys())
            raise KeyError(f"PVT region {pvtnum} not found. Available regions: {available}.")
        return region

    @property
    def n_regions(self) -> int:
        """Number of PVT regions."""
        return len(self.regions)

    @classmethod
    def from_one(cls, region: PVTRegion) -> Self:
        """
        Wrap a single `PVTRegion` as region 1.

        Convenience factory for the common single-region case.

        :param region: `PVTRegion` instance.
        :returns: `PVT` with one entry at key 1.
        """
        return cls(regions={1: region})

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        temperature: Temperature | Number,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build all PVT regions' tables from a parsed `DeckFile`.

        Detects which Eclipse PVT keywords are present (`PVTO` > `PVCO` > `PVDO` > `PVCDO` for oil;
        `PVTG` > `PVDG` for gas; `PVTW` for water) and builds one `PVTRegion` per `PVTNUM` region.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param temperature: Reservoir temperature (°F) used for all regions,
            or a reservoir regional `Temperature` instance.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks.
        :param warn_on_extrapolation: Log warnings on extrapolation.
        :returns: `PVT` keyed by 1-based PVTNUM index.
        """
        regions = load_pvt_regions(
            deck_file=deck_file,
            temperature=temperature
            if isinstance(temperature, Temperature)
            else Temperature(temperature, unit_system=UnitSystem.FIELD),
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            dtype=dtype,
        )
        return cls(regions=regions)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVT` with all region tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVT` in *target* units.
        """
        return self.__class__(
            regions={
                pvtnum: region.convert(target, table=table)
                for pvtnum, region in self.regions.items()
            }
        )

    def __getitem__(self, key: int) -> PVTRegion:
        return self.region(key)

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __contains__(self, key: object) -> bool:
        return key in self.regions


def get_min_temperature_points(interpolation_method: InterpolationMethod) -> int:
    """Minimum temperature-axis length required by `PVTTable` for a given method."""
    return 4 if interpolation_method == "cubic" else 2


def ensure_strictly_increasing(
    values: npt.NDArray, min_points: int, dtype: npt.DTypeLike
) -> npt.NDArray:
    """
    Deduplicate, sort, and pad *values* so the result is strictly increasing
    and has at least *min_points* entries (both required by `PVTTable`).

    Padding interpolates additional points between the existing extremes
    rather than fabricating unrelated ones, so the physical range of the
    axis is preserved.
    """
    values = np.unique(values.astype(dtype, copy=False))
    if len(values) == 1:
        # Degenerate (isothermal) case - bracket it with a tiny symmetric span.
        min_value, max_value = values[0] - 1.0, values[0] + 1.0
        values = np.linspace(min_value, max_value, max(min_points, 2), dtype=dtype)
    elif len(values) < min_points:
        # Preserve the real knots exactly; fill in between them to satisfy
        # the interpolator's minimum-point requirement.
        filler = np.linspace(values[0], values[-1], min_points, dtype=dtype)
        values = np.unique(np.concatenate([values, filler])).astype(dtype, copy=False)
    return values


def generate_temperature_axis(
    temperature: TemperatureSpec,
    dtype: npt.DTypeLike,
    *,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    n_points: int | None = None,
    max_points: int = 25,
) -> npt.NDArray:
    """
    Build a temperature axis appropriate to *temperature*'s spec type.

    Eclipse PVT tables are 2-D `(n_p, n_t)`. `PVTTable` requires the `n_t`
    axis to be strictly increasing, with at least 2 points for `"linear"`
    interpolation or 4 for `"cubic"`. This builds the smallest axis that
    faithfully represents each spec type while satisfying that constraint:

    - `Number` (scalar / isothermal): a tight symmetric bracket around the
      value. All property columns are broadcast identically across it
      (see `_broadcast_to_2d`), so extra points cost nothing and change
      nothing physically - they only exist to satisfy `PVTTable`.
    - `TemperatureGradient`: samples `gradient.at_depth(...)` across
      `depth_range` (the actual depth extent of the region's cells, when
      known) so the axis truly brackets the temperatures the region will
      see. Without a known depth extent, falls back to a minimal bracket
      around the reference temperature and warns, since the gradient's
      true range can't be determined.
    - `TemperatureTable`: uses the table's own (sorted, unique) temperature
      knots directly. This is the most faithful axis possible. It
      reproduces the table's actual breakpoints with zero extra
      interpolation error, rather than resampling onto an arbitrary grid.
      Very dense tables are downsampled to *max_points* (endpoints
      preserved) to keep the resulting PVT table size bounded.

    :param temperature: Reservoir temperature spec for this region.
    :param dtype: Output dtype.
    :param interpolation_method: `"linear"` or `"cubic"` - determines the
        minimum number of axis points required downstream.
    :param depth_range: Optional `(min_depth, max_depth)` of the cells this
        region covers, in the same length units as `temperature`. Only
        used for `TemperatureGradient`; ignored otherwise.
    :param n_points: Override the number of points to sample (gradient) or
        the target density (ignored for scalar/table, which use their own
        natural sizing). Defaults to a method-appropriate value.
    :param max_points: Cap on axis length for `TemperatureTable` downsampling.
    :returns: 1-D strictly increasing array, dtype *dtype*.
    """
    dtype = np.dtype(dtype)
    min_points = get_min_temperature_points(interpolation_method)

    if isinstance(temperature, TemperatureGradient):
        count = n_points or max(min_points, 8)
        if depth_range is not None:
            min_value, max_value = sorted((depth_range[0], depth_range[1]))
            if max_value <= min_value:
                max_value = min_value + 1.0
            depths = np.linspace(min_value, max_value, count, dtype=dtype)
        else:
            warnings.warn(
                "`TemperatureGradient` without a `depth_range`. Falling back to a "
                "minimal bracket around the reference temperature. Pass the "
                "region's actual cell-depth extent for a physically accurate "
                "temperature axis.",
                UserWarning,
                stacklevel=3,
            )
            ref = temperature.reference_depth
            depths = np.linspace(ref - 1.0, ref + 1.0, max(min_points, 2), dtype=dtype)
        temperatures = temperature.at_depth(depths).astype(dtype, copy=False)  # type: ignore[arg-type]
        return ensure_strictly_increasing(temperatures, min_points, dtype)

    if isinstance(temperature, TemperatureTable):
        knots = np.unique(temperature.temperatures.astype(dtype, copy=False))
        if len(knots) > max_points:
            # Downsample onto an evenly spaced grid spanning the same range;
            # endpoints (and thus the full physical range) are preserved.
            knots = np.linspace(knots[0], knots[-1], max_points, dtype=dtype)
        return ensure_strictly_increasing(knots, min_points, dtype)

    # Scalar Number - unchanged behavior, generalized to respect min_points.
    count = n_points or min_points
    return np.linspace(temperature - 1.0, temperature + 1.0, max(count, 2), dtype=dtype)


def _broadcast_to_2d(values_1d: npt.NDArray, n_t: int = 2) -> npt.NDArray:
    """
    Broadcast a 1-D array of shape `(n_p,)` to 2-D shape `(n_p, n_t)`.

    Used to satisfy the `(n_p, n_t)` shape requirement of `PVTTable`
    when deck data contains only a pressure axis (isothermal tables).

    :param values_1d: 1-D array of shape `(n_p,)`.
    :param n_t: Number of temperature knots (usually 2 for degenerate axis).
    :returns: 2-D array of shape `(n_p, n_t)`.
    """
    return np.tile(values_1d[:, np.newaxis], (1, n_t)).astype(values_1d.dtype, copy=False)


def build_oil_data_from_pvto(
    pvto_records: list[dict[str, typing.Any]],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build oil `PVTData` from a parsed `PVTO` record set.

    `PVTO` format: Rs is the outer key; each Rs group contains rows of
    `(pressure, bo, viscosity)` ordered by ascending pressure. The first
    row in each Rs group is the saturated row at bubble-point pressure;
    subsequent rows at higher pressure are the undersaturated branch.

    The builder:

    1. Extracts the saturated branch `(Pb(Rs), Bo_sat(Rs), μo_sat(Rs))`.
    2. Builds a regular pressure grid from the union of all bubble-point
       pressures extended to cover the maximum undersaturated pressure seen.
    3. At each pressure, interpolates Bo and μo from the appropriate Rs
       group's undersaturated branch (or uses the saturated value when
       P ≤ Pb).
    4. Derives the density table using `ρo = (ρo,SC + Rs·ρg,SC) / Bo`
       when reference densities are available.
    5. Derives compressibility from `co = -(1/Bo)·(∂Bo/∂P)`.

    :param pvto_records: List of row dicts from the parsed `PVTO` keyword.
        Each dict has keys `"solution_gor"`, `"pressure"`, `"fvf"`, `"viscosity"`.
    :param density_record: `DENSITY` record dict with `"oil"` and `"gas"` keys
        (lbm/ft³ at standard conditions).
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    # Eclipse reports Rs in Mscf/STB under FIELD units (see PVTO's deck docs);
    # internally we standardize on SCF/STB, matching the `gas_oil_ratio`
    # convention `get_conversion_factors` assumes. METRIC/LAB decks already
    # report Rs in the internally-expected units (sm³/sm³ / scc/scc), so no
    # rescale is needed there.
    mscf_to_scf = c.MSCF_TO_SCF if unit_system == UnitSystem.FIELD else 1.0

    temperatures = generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)

    # Group records by Rs value. Rows are copied so the deck's own records are never
    # rescaled in place (a second load of the same deck would otherwise rescale twice).
    solution_gor_to_rows: dict[float, list[dict]] = {}
    for record in pvto_records:
        row = {**record, "solution_gor": record["solution_gor"] * mscf_to_scf}
        solution_gor_to_rows.setdefault(row["solution_gor"], []).append(row)

    if len(solution_gor_to_rows) < 2:
        raise ValidationError(
            f"`PVTO` table requires at least 2 Rs values; got {len(solution_gor_to_rows)}."
        )

    solution_gor_keys = sorted(solution_gor_to_rows.keys())
    solution_gor_values = np.array(solution_gor_keys, dtype=dtype)
    n_rs = len(solution_gor_values)

    # Saturated branch: lowest-pressure row in each Rs group = bubble point
    bubble_point_pressure_values = np.empty(n_rs, dtype=dtype)
    saturated_oil_fvf = np.empty(n_rs, dtype=dtype)
    saturated_oil_viscosity = np.empty(n_rs, dtype=dtype)

    for i, solution_gor_key in enumerate(solution_gor_keys):
        rows = sorted(solution_gor_to_rows[solution_gor_key], key=lambda row: row["pressure"])
        saturated_row = rows[0]
        bubble_point_pressure_values[i] = saturated_row["pressure"]
        saturated_oil_fvf[i] = saturated_row["fvf"]
        saturated_oil_viscosity[i] = saturated_row["viscosity"]

    # Pressure grid: bubble-point pressures + extension to max undersaturated pressure.
    # We do not merge all undersaturated rows into one flat grid - that would mix
    # physically distinct branches. Instead we extend monotonically from the
    # highest Pb to cover the undersaturated range.
    max_pressure = max(
        float(row["pressure"]) for rows in solution_gor_to_rows.values() for row in rows
    )
    extension = np.linspace(
        float(bubble_point_pressure_values[-1]),
        max_pressure,
        max(10, n_rs),
        dtype=dtype,
    )
    pressures = np.unique(np.concatenate([bubble_point_pressure_values, extension])).astype(dtype)
    n_p = len(pressures)

    # Reference undersaturated branch: most Eclipse decks only attach an
    # undersaturated (>1 row) branch to the highest Rs value - all other Rs
    # groups have just their single saturated point. A group with a single
    # pressure point has no slope to interpolate at all (feeding scipy a
    # length-1 x-array degenerates to a 0/0 divide -> NaN, which then blows
    # up the PCHIP derivative below with "y must contain only finite
    # values"). We borrow the ΔBo(ΔP) and viscosity-ratio(ΔP) behavior from
    # whichever Rs group actually has an undersaturated branch ("most rows"
    # is used as a robust proxy for "the one with undersaturated data" in
    # case a deck attaches it somewhere other than the top Rs) and apply it
    # relative to each single-row group's own saturated point.
    reference_key = max(solution_gor_keys, key=lambda k: len(solution_gor_to_rows[k]))
    reference_rows = sorted(solution_gor_to_rows[reference_key], key=lambda row: row["pressure"])
    if len(reference_rows) < 2:
        raise ValidationError(
            "`PVTO` table has no Rs group with an undersaturated (>1 row) "
            "branch; cannot extrapolate undersaturated properties for the "
            "single-row Rs groups."
        )

    reference_pressure_arr = np.array([row["pressure"] for row in reference_rows], dtype=dtype)
    reference_oil_fvf_arr = np.array([row["fvf"] for row in reference_rows], dtype=dtype)
    reference_viscosity_arr = np.array([row["viscosity"] for row in reference_rows], dtype=dtype)
    reference_delta_pressure = reference_pressure_arr - reference_pressure_arr[0]
    reference_delta_oil_fvf = reference_oil_fvf_arr - reference_oil_fvf_arr[0]
    reference_viscosity_ratio = reference_viscosity_arr / reference_viscosity_arr[0]

    # ΔBo(ΔP) and viscosity-ratio(ΔP) relative to the reference branch's own
    # saturated point, so they can be applied to any other Rs group's
    # saturated point regardless of its own pressure scale.
    delta_oil_fvf_of_dp = interp1d(
        reference_delta_pressure,
        reference_delta_oil_fvf,
        kind="linear",
        bounds_error=False,
        fill_value=(reference_delta_oil_fvf[0], reference_delta_oil_fvf[-1]),
    )
    viscosity_ratio_of_dp = interp1d(
        reference_delta_pressure,
        reference_viscosity_ratio,
        kind="linear",
        bounds_error=False,
        fill_value=(reference_viscosity_ratio[0], reference_viscosity_ratio[-1]),
    )

    # Per-Rs interpolators for the full (saturated + undersaturated) branch
    oil_fvf_interps: list[interp1d] = []
    oil_viscosity_interps: list[interp1d] = []

    for solution_gor_key in solution_gor_keys:
        rows = sorted(solution_gor_to_rows[solution_gor_key], key=lambda row: row["pressure"])
        pressure_arr = np.array([row["pressure"] for row in rows], dtype=dtype)
        oil_fvf_arr = np.array([row["fvf"] for row in rows], dtype=dtype)
        oil_viscosity_arr = np.array([row["viscosity"] for row in rows], dtype=dtype)

        if len(rows) < 2:
            # No explicit undersaturated branch for this Rs value - borrow
            # the reference branch's ΔBo(ΔP) / viscosity-ratio(ΔP) and apply
            # them relative to this group's own saturated point, so this
            # group also ends up with >= 2 monotonically increasing
            # pressure points for interp1d/PCHIP.
            extra_dp = reference_delta_pressure[reference_delta_pressure > 0]
            if len(extra_dp) == 0:
                extra_dp = np.array([1.0], dtype=dtype)
            extra_pressures = pressure_arr[0] + extra_dp
            extra_oil_fvf = oil_fvf_arr[0] + delta_oil_fvf_of_dp(extra_dp)
            extra_viscosity = oil_viscosity_arr[0] * viscosity_ratio_of_dp(extra_dp)

            pressure_arr = np.concatenate([pressure_arr, extra_pressures])
            oil_fvf_arr = np.concatenate([oil_fvf_arr, extra_oil_fvf])
            oil_viscosity_arr = np.concatenate([oil_viscosity_arr, extra_viscosity])
            order = np.argsort(pressure_arr)
            pressure_arr = pressure_arr[order]
            oil_fvf_arr = oil_fvf_arr[order]
            oil_viscosity_arr = oil_viscosity_arr[order]

        oil_fvf_interps.append(
            interp1d(
                pressure_arr,
                oil_fvf_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(oil_fvf_arr[0], oil_fvf_arr[-1]),
            )
        )
        oil_viscosity_interps.append(
            interp1d(
                pressure_arr,
                oil_viscosity_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(oil_viscosity_arr[0], oil_viscosity_arr[-1]),
            )
        )

    # Rs(P) on the saturated branch via inverse of Pb(Rs).
    # Pb must be monotonically increasing with Rs for a well-formed `PVTO` table.
    # If not strictly monotone, we clip safely.
    if not np.all(np.diff(bubble_point_pressure_values) > 0):
        warnings.warn(
            "`PVTO` bubble-point pressures are not strictly increasing with Rs. "
            "Rs(P) inversion may be inaccurate for some pressure values.",
            UserWarning,
            stacklevel=4,
        )

    solution_gor_of_pressure = interp1d(
        bubble_point_pressure_values,
        solution_gor_values,
        kind="linear",
        bounds_error=False,
        fill_value=(solution_gor_values[0], solution_gor_values[-1]),
    )

    oil_fvf_2d = np.empty((n_p, n_t), dtype=dtype)
    oil_viscosity_2d = np.empty((n_p, n_t), dtype=dtype)
    solution_gor_2d = np.empty((n_p, n_t), dtype=dtype)

    for i, pressure in enumerate(pressures):
        # Determine which Rs group governs at this pressure:
        # find the Rs group whose Pb is closest to (and ≤) this pressure.
        rs_idx = int(np.searchsorted(bubble_point_pressure_values, pressure, side="right")) - 1
        rs_idx = int(np.clip(rs_idx, 0, n_rs - 1))

        # Rs at this pressure on the saturated envelope
        rs_at_pressure = solution_gor_of_pressure(pressure)
        solution_gor_2d[i, :] = rs_at_pressure

        oil_fvf = oil_fvf_interps[rs_idx](pressure)
        oil_viscosity = oil_viscosity_interps[rs_idx](pressure)

        oil_fvf_2d[i, :] = oil_fvf
        oil_viscosity_2d[i, :] = oil_viscosity

    # Bubble-point table: Pb(Rs, T) - 2-D, one Pb per Rs per temperature
    bubble_point_pressure_2d = np.tile(
        bubble_point_pressure_values[:, np.newaxis], (1, n_t)
    ).astype(dtype, copy=False)

    # Resolve reference densities: pvt takes precedence over DENSITY record
    stock_tank_oil_density: Number | None = None
    stock_tank_gas_density: Number | None = None
    if density_record is not None:
        stock_tank_oil_density = density_record.get("oil")
        stock_tank_gas_density = density_record.get("gas")

    # Density: ρo = (ρo,SC + Rs·ρg,SC / f) / Bo, with f = ft³/STB in FIELD units (1 otherwise)
    oil_density_2d: npt.NDArray | None = None
    if stock_tank_oil_density is not None and stock_tank_gas_density is not None:
        stb_to_volume = get_stb_to_volume_factor(unit_system)
        oil_density_2d = (
            (stock_tank_oil_density + solution_gor_2d * stock_tank_gas_density / stb_to_volume)
            / oil_fvf_2d
        ).astype(dtype, copy=False)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P) via PCHIP derivative
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbo_dp = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)(pressures)
        oil_compressibility_2d[:, j] = -(1.0 / oil_fvf_2d[:, j]) * dbo_dp
    # Compressibility must be non-negative; clamp to physical range
    clip_compressibility(
        oil_compressibility_2d,
        dtype=dtype,
        unit_system=unit_system,
        context="`PVTO` oil compressibility",
    )
    return PVTData(
        phase=FluidPhase.OIL,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        bubble_point_pressures=typing.cast(FloatArray[TwoDimensions], bubble_point_pressure_2d),
        solution_gas_to_oil_ratios=typing.cast(FloatArray[OneDimension], solution_gor_values),
        formation_volume_factor_table=typing.cast(FloatArray[TwoDimensions], oil_fvf_2d),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], oil_viscosity_2d),
        solution_gor_table=typing.cast(FloatArray[TwoDimensions], solution_gor_2d),
        density_table=typing.cast(FloatArray[TwoDimensions], oil_density_2d)
        if oil_density_2d is not None
        else None,
        compressibility_table=typing.cast(FloatArray[TwoDimensions], oil_compressibility_2d),
        dtype=dtype,
        unit_system=unit_system,
    )


def build_oil_data_from_pvdo(
    pvdo_records: list[dict[str, typing.Any]],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build dead-oil `PVTData` from a parsed `PVDO` record set.

    `PVDO` format: single table of `(pressure, bo, viscosity)` rows for
    dead oil (Rs = 0 everywhere). No bubble-point switching is required
    since dead oil has no dissolved gas.

    :param pvdo_records: List of row dicts with keys `"pressure"`, `"fvf"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"oil"` key used for ρo,SC.
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    temperatures = generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)

    rows = sorted(pvdo_records, key=lambda row: row["pressure"])
    if len(rows) < 2:
        raise ValidationError(f"`PVDO` table requires at least 2 rows; got {len(rows)}.")

    pressures = np.array([row["pressure"] for row in rows], dtype=dtype)
    oil_fvf_1d = np.array([row["fvf"] for row in rows], dtype=dtype)
    oil_viscosity_1d = np.array([row["viscosity"] for row in rows], dtype=dtype)
    n_p = len(pressures)

    if not np.all(np.diff(pressures) > 0):
        raise ValidationError("`PVDO` pressures must be strictly increasing.")
    if np.any(oil_fvf_1d <= 0):
        raise ValidationError("`PVDO` Bo values must be positive.")
    if np.any(oil_viscosity_1d <= 0):
        raise ValidationError("`PVDO` viscosity values must be positive.")

    oil_fvf_2d = _broadcast_to_2d(oil_fvf_1d, n_t)
    oil_viscosity_2d = _broadcast_to_2d(oil_viscosity_1d, n_t)
    # Dead oil: Rs = 0 everywhere
    solution_gor_2d = np.zeros((n_p, n_t), dtype=dtype)

    stock_tank_oil_density: Number | None = None
    if density_record is not None:
        stock_tank_oil_density = density_record.get("oil")

    # Density: ρo = ρo,SC / Bo (dead oil - Rs = 0)
    oil_density_2d: npt.NDArray | None = None
    if stock_tank_oil_density is not None:
        oil_density_2d = (stock_tank_oil_density / oil_fvf_2d).astype(dtype, copy=False)

    # Compressibility: co = -(1/Bo)·(∂Bo/∂P)
    oil_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbo_dp = PchipInterpolator(pressures, oil_fvf_2d[:, j]).derivative(1)(pressures)
        oil_compressibility_2d[:, j] = -(1.0 / oil_fvf_2d[:, j]) * dbo_dp
    clip_compressibility(
        oil_compressibility_2d,
        dtype=dtype,
        unit_system=unit_system,
        context="`PVDO` oil compressibility",
    )

    return PVTData(
        phase=FluidPhase.OIL,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        formation_volume_factor_table=typing.cast(FloatArray[TwoDimensions], oil_fvf_2d),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], oil_viscosity_2d),
        solution_gor_table=typing.cast(FloatArray[TwoDimensions], solution_gor_2d),
        density_table=typing.cast(FloatArray[TwoDimensions], oil_density_2d)
        if oil_density_2d is not None
        else None,
        compressibility_table=typing.cast(FloatArray[TwoDimensions], oil_compressibility_2d),
        dtype=dtype,
        unit_system=unit_system,
    )


def build_oil_data_from_pvcdo(
    pvcdo_record: dict[str, Number],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    n_pressure_points: int = 40,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build dead-oil `PVTData` from a `PVCDO` analytical record.

    The record holds `(reference_pressure, fvf, compressibility, viscosity, viscosibility)`:
    oil without dissolved gas, with constant compressibility `co` and viscosibility `cv`.
    Like `PVTW` for water, `Bo(P)` and `μo(P)` are evaluated
    with Eclipse's second-order series on a synthetic pressure grid and handed to
    `build_oil_data_from_pvdo`. With `X = co · (P - P_ref)` and `Y = (co - cv) · (P - P_ref)`:

    - `Bo(P) = Bo_ref / (1 + X + X²/2)`
    - `μo(P) = μo_ref · (1 + X + X²/2) / (1 + Y + Y²/2)`

    The pressure grid spans `[P_ref/5, P_ref x 5]`.

    :param pvcdo_record: Dict with keys `"reference_pressure"`, `"fvf"`, `"compressibility"`,
        `"viscosity"`, and optionally `"viscosibility"` (default 0).
    :param density_record: `DENSITY` record; `"oil"` key used for ρo,SC.
    :param temperature: Reservoir temperature.
    :param n_pressure_points: Points in the synthetic pressure grid.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    reference_pressure = pvcdo_record["reference_pressure"]
    reference_oil_fvf = pvcdo_record["fvf"]
    oil_compressibility = pvcdo_record["compressibility"]
    reference_viscosity = pvcdo_record["viscosity"]
    oil_viscosibility = pvcdo_record.get("viscosibility", 0.0)

    if reference_oil_fvf <= 0:
        raise ValidationError("`PVCDO` Bo must be positive.")
    if oil_compressibility < 0:
        raise ValidationError("`PVCDO` co (compressibility) must be non-negative.")
    if reference_viscosity <= 0:
        raise ValidationError("`PVCDO` viscosity must be positive.")

    pressures = np.linspace(
        max(0.0, reference_pressure / 5.0), reference_pressure * 5.0, n_pressure_points
    )
    delta_p = pressures - reference_pressure
    x = oil_compressibility * delta_p
    y = (oil_compressibility - oil_viscosibility) * delta_p
    series_x = 1.0 + x + 0.5 * x * x
    oil_fvf = reference_oil_fvf / series_x
    oil_viscosity = reference_viscosity * series_x / (1.0 + y + 0.5 * y * y)
    synthetic_rows = [
        {"pressure": pressure, "fvf": fvf, "viscosity": viscosity}
        for pressure, fvf, viscosity in zip(pressures, oil_fvf, oil_viscosity, strict=False)
    ]
    return build_oil_data_from_pvdo(
        pvdo_records=synthetic_rows,
        density_record=density_record,
        temperature=temperature,
        unit_system=unit_system,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
        dtype=dtype,
    )


def build_oil_data_from_pvco(
    pvco_records: list[dict[str, typing.Any]],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    n_undersaturated_points: int = 10,
    pressure_span_factor: Number = 3.0,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build live-oil `PVTData` from a parsed `PVCO` record set.

    `PVCO` format: one row per bubble point, `(bubble_point_pressure, solution_gor, fvf,
    viscosity, compressibility, viscosibility)`. Each row is a saturated state. The undersaturated
    branch at that Rs follows from the row's own compressibility `co` and viscosibility `cv`, with
    Eclipse's second-order series (as for `PVTW` / `PVCDO`). With `X = co · (P - Pb)` and
    `Y = (co - cv) · (P - Pb)`:

    - `Bo(P) = Bob / (1 + X + X²/2)`
    - `μo(P) = μob · (1 + X + X²/2) / (1 + Y + Y²/2)`

    Every row is expanded into the equivalent `PVTO` group (the saturated point plus
    `n_undersaturated_points` points up to `pressure_span_factor x` the highest bubble point) and
    handed to `build_oil_data_from_pvto`, so the saturated envelope, bubble-point table and
    derived tables are built exactly as for `PVTO`.

    :param pvco_records: List of row dicts with keys `"bubble_point_pressure"`, `"solution_gor"`,
        `"fvf"`, `"viscosity"`, `"compressibility"` and optionally `"viscosibility"` (default 0).
        `"solution_gor"` is in deck units (Mscf/stb under FIELD), like `PVTO`.
    :param density_record: `DENSITY` record dict with `"oil"` and `"gas"` keys.
    :param temperature: Reservoir temperature.
    :param n_undersaturated_points: Undersaturated pressure points generated per bubble point.
    :param pressure_span_factor: The undersaturated branches extend to this multiple of the
        highest bubble-point pressure.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the oil phase.
    """
    if len(pvco_records) < 2:
        raise ValidationError(
            f"`PVCO` table requires at least 2 rows (bubble points); got {len(pvco_records)}."
        )

    rows = sorted(pvco_records, key=lambda row: row["bubble_point_pressure"])
    maximum_pressure = pressure_span_factor * rows[-1]["bubble_point_pressure"]
    synthetic_rows: list[dict[str, typing.Any]] = []
    for row in rows:
        bubble_point_pressure = row["bubble_point_pressure"]
        bubble_point_fvf = row["fvf"]
        bubble_point_viscosity = row["viscosity"]
        oil_compressibility = row["compressibility"]
        oil_viscosibility = row.get("viscosibility", 0.0)

        if bubble_point_pressure <= 0:
            raise ValidationError("`PVCO` bubble-point pressures must be positive.")
        if bubble_point_fvf <= 0:
            raise ValidationError("`PVCO` Bo values must be positive.")
        if oil_compressibility < 0:
            raise ValidationError("`PVCO` co (compressibility) values must be non-negative.")
        if bubble_point_viscosity <= 0:
            raise ValidationError("`PVCO` viscosity values must be positive.")

        pressures = np.linspace(bubble_point_pressure, maximum_pressure, n_undersaturated_points)
        delta_p = pressures - bubble_point_pressure
        x = oil_compressibility * delta_p
        y = (oil_compressibility - oil_viscosibility) * delta_p
        series_x = 1.0 + x + 0.5 * x * x
        oil_fvf = bubble_point_fvf / series_x
        oil_viscosity = bubble_point_viscosity * series_x / (1.0 + y + 0.5 * y * y)
        synthetic_rows.extend(
            {
                "solution_gor": row["solution_gor"],
                "pressure": pressure,
                "fvf": fvf,
                "viscosity": viscosity,
            }
            for pressure, fvf, viscosity in zip(pressures, oil_fvf, oil_viscosity, strict=False)
        )

    return build_oil_data_from_pvto(
        pvto_records=synthetic_rows,
        density_record=density_record,
        temperature=temperature,
        unit_system=unit_system,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
        dtype=dtype,
    )


def build_gas_data_from_pvdg(
    pvdg_records: list[dict[str, typing.Any]],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build dry-gas `PVTData` from a parsed `PVDG` record set.

    `PVDG` format: single table of `(pressure, bg, viscosity)` rows.
    Eclipse stores Bg in rb/Mscf; this builder converts to ft³/SCF:
    `Bg_ft3_scf = Bg_rb_Mscf x 5.615 / 1000`.

    :param pvdg_records: List of row dicts with keys `"pressure"`, `"fvf"`,
        `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` key used for ρg,SC.
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the gas phase.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    temperatures = generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)

    rows = sorted(pvdg_records, key=lambda row: row["pressure"])
    if len(rows) < 2:
        raise ValidationError(f"`PVDG` table requires at least 2 rows; got {len(rows)}.")

    pressures = np.array([row["pressure"] for row in rows], dtype=dtype)
    # Eclipse reports Bg in rb/Mscf under FIELD units only; METRIC/LAB decks
    # already report Bg in rm³/sm³ / rcc/scc, matching our internal `gas_fvf`
    # convention, so the rescale only applies to FIELD.
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET if unit_system == UnitSystem.FIELD else 1.0
    mscf_to_scf = c.MSCF_TO_SCF if unit_system == UnitSystem.FIELD else 1.0
    gas_fvf_1d = np.array([row["fvf"] * bbl_to_ft3 / mscf_to_scf for row in rows], dtype=dtype)
    gas_viscosity_1d = np.array([row["viscosity"] for row in rows], dtype=dtype)

    if not np.all(np.diff(pressures) > 0):
        raise ValidationError("`PVDG` pressures must be strictly increasing.")
    if np.any(gas_fvf_1d <= 0):
        raise ValidationError("`PVDG` Bg values must be positive.")
    if np.any(gas_viscosity_1d <= 0):
        raise ValidationError("`PVDG` viscosity values must be positive.")

    n_p = len(pressures)
    gas_fvf_2d = _broadcast_to_2d(gas_fvf_1d, n_t)
    gas_viscosity_2d = _broadcast_to_2d(gas_viscosity_1d, n_t)

    stock_tank_gas_density: Number | None = None
    if density_record is not None:
        stock_tank_gas_density = density_record.get("gas")

    # Density: ρg = ρg,SC / Bg
    gas_density_2d: npt.NDArray | None = None
    if stock_tank_gas_density is not None:
        gas_density_2d = (stock_tank_gas_density / gas_fvf_2d).astype(dtype, copy=False)

    # Compressibility: cg ≈ -(1/Bg)·(∂Bg/∂P)
    gas_compressibility_2d = np.empty((n_p, n_t), dtype=dtype)
    for j in range(n_t):
        dbg_dp = PchipInterpolator(pressures, gas_fvf_2d[:, j]).derivative(1)(pressures)
        gas_compressibility_2d[:, j] = -(1.0 / gas_fvf_2d[:, j]) * dbg_dp
    clip_compressibility(
        gas_compressibility_2d,
        dtype=dtype,
        unit_system=unit_system,
        pressure=pressures[:, np.newaxis],
        context="`PVDG` gas compressibility",
    )

    return PVTData(
        phase=FluidPhase.GAS,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        formation_volume_factor_table=typing.cast(FloatArray[TwoDimensions], gas_fvf_2d),
        viscosity_table=typing.cast(FloatArray[TwoDimensions], gas_viscosity_2d),
        density_table=typing.cast(FloatArray[TwoDimensions], gas_density_2d)
        if gas_density_2d is not None
        else None,
        compressibility_table=typing.cast(FloatArray[TwoDimensions], gas_compressibility_2d),
        dtype=dtype,
        unit_system=unit_system,
    )


def build_gas_data_from_pvtg(
    pvtg_records: list[dict[str, typing.Any]],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build wet-gas `PVTData` from a parsed `PVTG` record set.

    `PVTG` format: pressure is the outer key; each pressure group contains
    rows of `(rv, bg, viscosity)` ordered by ascending Rv. The row with the largest Rv
    in each group is the saturated (dew-point) state at that pressure.

    The returned data keeps the Rv axis in its own field, `vaporized_oil_ratios`, separate from
    `temperatures` (the tables are isothermal, so they are broadcast over the temperature axis):

    - Undersaturated gas: `undersaturated_formation_volume_factor_table` and
      `undersaturated_viscosity_table`, shape `(n_p, n_t, n_rv)`. All Rv values from all
      pressure groups are unioned into the common Rv grid; each group is linearly
      interpolated onto it from its own rows (flat beyond the group's own Rv range).
    - Saturated gas (on the dew curve): the 2-D `formation_volume_factor_table`,
      `viscosity_table`, `density_table` and `compressibility_table`, evaluated at each
      pressure's largest Rv, plus `vaporized_oil_ratio_table` (that Rv, Rv_sat(P)).
    - `dew_point_pressures`: Pdew(Rv, T), the inverse of Rv_sat(P), shape `(n_rv, n_t)`.

    :param pvtg_records: List of row dicts with keys `"pressure"`, `"vaporized_ogr"`,
        `"fvf"`, `"viscosity"`.
    :param density_record: `DENSITY` record; `"gas"` and `"oil"` keys used.
    :param temperature: Reservoir temperature.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the gas phase.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    temperatures = generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)
    # Eclipse reports Rv in STB/Mscf under FIELD units (see PVTG's deck docs);
    # internally we standardize on STB/SCF, matching the `vaporized_oil_gas_ratio`
    # convention `get_conversion_factors` assumes. METRIC/LAB decks already
    # report Rv in the internally-expected units (sm³/sm³ / scc/scc), so no
    # rescale is needed there.
    scf_to_mscf = c.SCF_TO_MSCF if unit_system == UnitSystem.FIELD else 1.0

    pressure_to_rows: dict[float, list[dict]] = {}
    for record in pvtg_records:
        # Copy the row so the deck's own record is never rescaled in place
        row = {**record, "vaporized_ogr": record["vaporized_ogr"] * scf_to_mscf}
        pressure_to_rows.setdefault(row["pressure"], []).append(row)

    if len(pressure_to_rows) < 2:
        raise ValidationError(
            f"`PVTG` table requires at least 2 pressure values; got {len(pressure_to_rows)}."
        )

    pressure_keys = sorted(pressure_to_rows.keys())
    pressure_values = np.array(pressure_keys, dtype=dtype)
    n_p = len(pressure_values)

    if not np.all(np.diff(pressure_values) > 0):
        raise ValidationError("`PVTG` pressures must be strictly increasing.")

    # Union of all Rv values across all pressure groups -> common Rv grid
    all_rv = sorted({row["vaporized_ogr"] for rows in pressure_to_rows.values() for row in rows})
    if len(all_rv) < 1:
        raise ValidationError("`PVTG` table contains no Rv values.")

    rv_values = np.array(all_rv, dtype=dtype)
    n_rv = len(rv_values)
    # Eclipse reports Bg in rb/Mscf under FIELD units only; METRIC/LAB decks
    # already report Bg in rm³/sm³ / rcc/scc, matching our internal `gas_fvf`
    # convention, so the rescale only applies to FIELD.
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET if unit_system == UnitSystem.FIELD else 1.0
    mscf_to_scf = c.MSCF_TO_SCF if unit_system == UnitSystem.FIELD else 1.0

    gas_fvf_2d = np.empty((n_p, n_rv), dtype=dtype)
    gas_viscosity_2d = np.empty((n_p, n_rv), dtype=dtype)

    for i, pressure_key in enumerate(pressure_keys):
        rows = sorted(pressure_to_rows[pressure_key], key=lambda row: row["vaporized_ogr"])
        rv_arr = np.array([row["vaporized_ogr"] for row in rows], dtype=dtype)
        gas_fvf_arr = np.array(
            [row["fvf"] * bbl_to_ft3 / mscf_to_scf for row in rows], dtype=dtype
        )
        gas_viscosity_arr = np.array([row["viscosity"] for row in rows], dtype=dtype)

        if np.any(gas_fvf_arr <= 0):
            raise ValidationError(
                f"`PVTG` Bg values must be positive at pressure {pressure_key} psi."
            )
        if np.any(gas_viscosity_arr <= 0):
            raise ValidationError(
                f"`PVTG` viscosity values must be positive at pressure {pressure_key} psi."
            )

        # If this pressure group has only one Rv point, broadcast it
        if len(rv_arr) == 1:
            gas_fvf_2d[i, :] = gas_fvf_arr[0]
            gas_viscosity_2d[i, :] = gas_viscosity_arr[0]
        else:
            gas_fvf_2d[i, :] = interp1d(
                rv_arr,
                gas_fvf_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(gas_fvf_arr[0], gas_fvf_arr[-1]),
            )(rv_values)
            gas_viscosity_2d[i, :] = interp1d(
                rv_arr,
                gas_viscosity_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(gas_viscosity_arr[0], gas_viscosity_arr[-1]),
            )(rv_values)

    if n_rv < 2:
        raise ValidationError(f"`PVTG` table requires at least 2 distinct Rv values; got {n_rv}.")

    # Saturated (dew-curve) state at each tabulated pressure: the largest Rv listed in a
    # pressure group is that pressure's saturated Rv, so `(Rv_sat(P), P)` traces the dew-point
    # curve, the gas-side analogue of Pb(Rs) for oil. The saturated FVF and viscosity are the
    # group's values at that Rv, which the common Rv grid contains exactly.
    rv_max_per_pressure = np.array(
        [
            max(row["vaporized_ogr"] for row in pressure_to_rows[pressure_key])
            for pressure_key in pressure_keys
        ],
        dtype=dtype,
    )
    saturated_column = np.searchsorted(rv_values, rv_max_per_pressure)
    pressure_index = np.arange(n_p)
    saturated_gas_fvf = gas_fvf_2d[pressure_index, saturated_column]
    saturated_gas_viscosity = gas_viscosity_2d[pressure_index, saturated_column]

    # Pdew(Rv): invert Rv_sat(P) on its running maximum, taking the lowest pressure that
    # reaches each Rv, so the inversion stays single-valued even if Rv_sat is not monotonic
    # in P (retrograde behaviour). Gas with an Rv below the lowest Rv_sat is undersaturated
    # over the whole table, so it maps to the lowest tabulated pressure.
    if not np.all(np.diff(rv_max_per_pressure) > 0):
        warnings.warn(
            "`PVTG` saturated Rv (largest Rv per tabulated pressure) does not increase "
            "strictly with pressure. Dew-point pressure lookups use the lowest pressure "
            "that reaches each Rv.",
            UserWarning,
            stacklevel=4,
        )
    rv_envelope, first_index = np.unique(
        np.maximum.accumulate(rv_max_per_pressure), return_index=True
    )
    dew_point_pressure_of_rv = np.interp(rv_values, rv_envelope, pressure_values[first_index])
    dew_point_pressure_2d = np.tile(dew_point_pressure_of_rv[:, np.newaxis], (1, n_t)).astype(
        dtype, copy=False
    )

    # Resolve reference densities
    stock_tank_gas_density: Number | None = None
    stock_tank_oil_density: Number | None = None

    if density_record is not None:
        stock_tank_gas_density = density_record.get("gas")
        stock_tank_oil_density = density_record.get("oil")

    # Density along the dew curve: ρg = (ρg,SC + Rv_sat·ρo,SC · f) / Bg_sat  [wet gas],
    # f = ft³/STB in FIELD units; ρg = ρg,SC / Bg [no ρo,SC available]
    gas_density_2d: npt.NDArray | None = None
    if stock_tank_gas_density is not None:
        if stock_tank_oil_density is not None:
            stb_to_volume = get_stb_to_volume_factor(unit_system)
            saturated_gas_density = (
                stock_tank_gas_density
                + rv_max_per_pressure * stock_tank_oil_density * stb_to_volume
            ) / saturated_gas_fvf
        else:
            saturated_gas_density = stock_tank_gas_density / saturated_gas_fvf
        gas_density_2d = _broadcast_to_2d(saturated_gas_density.astype(dtype, copy=False), n_t)

    # Compressibility along the dew curve: cg = -(1/Bg_sat)·(dBg_sat/dP)
    dbg_dp = PchipInterpolator(pressure_values, saturated_gas_fvf).derivative(1)(pressure_values)
    gas_compressibility_2d = _broadcast_to_2d(
        (-(1.0 / saturated_gas_fvf) * dbg_dp).astype(dtype, copy=False), n_t
    )
    clip_compressibility(
        gas_compressibility_2d,
        dtype=dtype,
        unit_system=unit_system,
        pressure=pressure_values[:, np.newaxis],
        context="`PVTG` gas compressibility",
    )

    # The tables are isothermal: broadcast over the temperature axis
    undersaturated_gas_fvf_3d = np.repeat(gas_fvf_2d[:, np.newaxis, :], n_t, axis=1)
    undersaturated_gas_viscosity_3d = np.repeat(gas_viscosity_2d[:, np.newaxis, :], n_t, axis=1)
    return PVTData(
        phase=FluidPhase.GAS,
        pressures=typing.cast(FloatArray[OneDimension], pressure_values),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        vaporized_oil_ratios=typing.cast(FloatArray[OneDimension], rv_values),
        formation_volume_factor_table=typing.cast(
            FloatArray[TwoDimensions], _broadcast_to_2d(saturated_gas_fvf, n_t)
        ),
        viscosity_table=typing.cast(
            FloatArray[TwoDimensions], _broadcast_to_2d(saturated_gas_viscosity, n_t)
        ),
        vaporized_oil_ratio_table=typing.cast(
            FloatArray[TwoDimensions], _broadcast_to_2d(rv_max_per_pressure, n_t)
        ),
        undersaturated_formation_volume_factor_table=typing.cast(
            FloatArray[ThreeDimensions], undersaturated_gas_fvf_3d
        ),
        undersaturated_viscosity_table=typing.cast(
            FloatArray[ThreeDimensions], undersaturated_gas_viscosity_3d
        ),
        density_table=typing.cast(FloatArray[TwoDimensions], gas_density_2d)
        if gas_density_2d is not None
        else None,
        compressibility_table=typing.cast(FloatArray[TwoDimensions], gas_compressibility_2d),
        dew_point_pressures=typing.cast(FloatArray[TwoDimensions], dew_point_pressure_2d),
        dtype=dtype,
        unit_system=unit_system,
    )


def build_water_data_from_pvtw(
    pvtw_record: dict[str, Number],
    density_record: dict[str, Number] | None,
    temperature: TemperatureSpec,
    unit_system: UnitSystem,
    salinity: Number = 0.0,
    n_pressure_points: int = 50,
    interpolation_method: InterpolationMethod = "linear",
    depth_range: tuple[Number, Number] | None = None,
    dtype: npt.DTypeLike = None,
) -> PVTData:
    """
    Build water `PVTData` from a `PVTW` analytical record.

    `PVTW` provides four scalars per region - reference pressure, Bw, cw,
    μw, and optionally cv (viscosibility). Bw(P) and μw(P) are evaluated
    analytically on a pressure grid and stored as tables so all subsequent
    lookups are interpolator calls.

    The models used are Eclipse's second-order series (in place of the exponential, which
    it approximates). With `X = cw · (P - P_ref)` and `Y = (cw - cv) · (P - P_ref)`:

    - `Bw(P) = Bw_ref / (1 + X + X²/2)`
    - `μw(P) = μw_ref · (1 + X + X²/2) / (1 + Y + Y²/2)`

    The viscosity form follows from Eclipse evaluating the product `μw · Bw` as
    `(μw_ref · Bw_ref) / (1 + Y + Y²/2)`. It gives `cv = (1/μw) · (dμw/dP)` at the
    reference pressure, and a constant viscosity when `cv = 0`.

    The pressure grid spans `[P_ref/10, P_ref x 10]` so that
    the reference pressure always sits comfortably within the table bounds.

    :param pvtw_record: Dict with keys `"reference_pressure"`, `"fvf"`, `"compressibility"`,
        `"viscosity"`, and optionally `"viscosibility"` (default 0).
    :param density_record: `DENSITY` record; `"water"` key used for ρw,SC.
    :param temperature: Reservoir temperature.
    :param salinity: Water salinity (ppm NaCl).
    :param n_pressure_points: Points in the synthetic pressure grid.
    :param dtype: Array dtype; defaults to `get_dtype()`.
    :returns: `PVTData` for the water phase.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    temperatures = generate_temperature_axis(
        temperature,
        dtype=dtype,
        interpolation_method=interpolation_method,
        depth_range=depth_range,
    )
    n_t = len(temperatures)
    salinities = np.array([salinity], dtype=dtype)

    reference_pressure = pvtw_record["reference_pressure"]
    reference_water_fvf = pvtw_record["fvf"]
    water_compressibility = pvtw_record["compressibility"]
    reference_water_viscosity = pvtw_record["viscosity"]
    water_viscosibility = pvtw_record.get("viscosibility", 0.0)

    if reference_water_fvf <= 0:
        raise ValidationError("`PVTW` Bw must be positive.")
    if water_compressibility < 0:
        raise ValidationError("`PVTW` cw (compressibility) must be non-negative.")
    if reference_water_viscosity <= 0:
        raise ValidationError("`PVTW` viscosity must be positive.")

    min_pressure = max(0.0, reference_pressure / 10.0)
    max_pressure = reference_pressure * 10.0
    pressures = np.linspace(min_pressure, max_pressure, n_pressure_points, dtype=dtype)
    n_p = len(pressures)

    delta_p = pressures - reference_pressure
    # True Exponential approach (May give negatives by more accurate)
    # water_fvf_1d = (
    #     reference_water_fvf * np.exp(-water_compressibility * delta_p)
    # ).astype(dtype, copy=False)

    # Taylor's exponential approximation (more stable). Used by Eclipse.
    x = water_compressibility * delta_p
    water_fvf_1d = (reference_water_fvf / (1.0 + x + 0.5 * x * x)).astype(dtype, copy=False)

    # Eclipse evaluates the product μw·Bw as (μw_ref·Bw_ref) / (1 + Y + Y²/2) with
    # Y = (cw - cv)·ΔP; dividing by Bw(P) = Bw_ref / (1 + X + X²/2) gives μw.
    # cv = 0 leaves μw constant and cv is the true (1/μw)·(dμw/dP) at the reference pressure.
    y = (water_compressibility - water_viscosibility) * delta_p
    water_viscosity_1d = (
        reference_water_viscosity * (1.0 + x + 0.5 * x * x) / (1.0 + y + 0.5 * y * y)
    ).astype(dtype, copy=False)

    water_fvf_2d = _broadcast_to_2d(water_fvf_1d, n_t)
    water_viscosity_2d = _broadcast_to_2d(water_viscosity_1d, n_t)
    # Expand to 3-D: (n_p, n_t, n_s=1)
    water_fvf_3d = water_fvf_2d[:, :, np.newaxis].astype(dtype, copy=False)
    water_viscosity_3d = water_viscosity_2d[:, :, np.newaxis].astype(dtype, copy=False)

    stock_tank_water_density: Number | None = None
    if density_record is not None:
        stock_tank_water_density = density_record.get("water")

    # Density: ρw = ρw,SC / Bw
    water_density_3d: npt.NDArray | None = None
    if stock_tank_water_density is not None:
        water_density_3d = (stock_tank_water_density / water_fvf_3d).astype(dtype, copy=False)

    # Compressibility: cw is constant for this model - store as a uniform table
    # so the lookup API is consistent with oil and gas phases
    water_compressibility_3d = np.full((n_p, n_t, 1), water_compressibility, dtype=dtype)

    return PVTData(
        phase=FluidPhase.WATER,
        pressures=typing.cast(FloatArray[OneDimension], pressures),
        temperatures=typing.cast(FloatArray[OneDimension], temperatures),
        salinities=typing.cast(FloatArray[OneDimension], salinities),
        formation_volume_factor_table=typing.cast(FloatArray[ThreeDimensions], water_fvf_3d),
        viscosity_table=typing.cast(FloatArray[ThreeDimensions], water_viscosity_3d),
        density_table=typing.cast(FloatArray[ThreeDimensions], water_density_3d)
        if water_density_3d is not None
        else None,
        compressibility_table=typing.cast(FloatArray[ThreeDimensions], water_compressibility_3d),
        gas_free_water_fvf_table=typing.cast(FloatArray[TwoDimensions], water_fvf_2d),
        dtype=dtype,
        unit_system=unit_system,
    )


def load_pvt_regions(
    deck_file: DeckFile,
    temperature: Temperature,
    *,
    interpolation_method: InterpolationMethod = "linear",
    validate: bool = True,
    warn_on_extrapolation: bool = False,
    dtype: npt.DTypeLike = None,
) -> dict[int, PVTRegion]:
    """
    Build a `PVT` object from a parsed `DeckFile`.

    Detects which Eclipse PVT keywords are present and builds one
    `PVTRegion` per `PVTNUM` region:

    - Oil: `PVTO` (live oil, tabulated) -> `PVCO` (live oil, constant undersaturated
      compressibility and viscosibility) -> `PVDO` (dead oil, tabulated) -> `PVCDO`
      (dead oil, constant compressibility and viscosibility).
    - Gas: `PVTG` (wet gas, preferred) -> `PVDG` (dry gas).
    - Water: `PVTW` (always analytical; converted to a table internally).

    `DENSITY` records supply the stock-tank reference densities used to
    derive density tables and, through each region's `StaticPVT`, to recompute
    undersaturated oil density at simulation time.

    :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
    :param temperature: `Temperature` instance.
    :param interpolation_method: `"linear"` (default) or `"cubic"`.
    :param validate: Run physical-consistency checks.
    :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
    :returns: Mapping of 1-based `PVTNUM` region index to corresponding `PVTRegion`.
    :raises ValidationError: If no recognisable PVT keyword is found.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    # Retrieve deck records (each is a list-of-lists: outer = regions)
    pvto_records: list | None = deck_file.get("PVTO")
    pvdo_records: list | None = deck_file.get("PVDO")
    pvco_records: list | None = deck_file.get("PVCO")
    pvcdo_records: list | None = deck_file.get("PVCDO")
    pvtg_records: list | None = deck_file.get("PVTG")
    pvdg_records: list | None = deck_file.get("PVDG")
    pvtw_records: list | None = deck_file.get("PVTW")
    density_records: list | None = deck_file.get("DENSITY")

    if (
        pvto_records is None
        and pvdo_records is None
        and pvco_records is None
        and pvcdo_records is None
    ):
        raise ValidationError(
            "No oil PVT keyword found in DeckFile. "
            "Expected one of: `PVTO`, `PVDO`, `PVCO`, `PVCDO`."
        )

    # Number of regions is the maximum length across all keyword lists
    n_regions = max(
        len(records)
        for records in [
            pvto_records,
            pvdo_records,
            pvco_records,
            pvcdo_records,
            pvtg_records,
            pvdg_records,
            pvtw_records,
        ]
        if records is not None
    )
    unit_system = deck_file.unit_system
    table_kwargs: dict[str, typing.Any] = dict(
        interpolation_method=interpolation_method,
        validate=validate,
        warn_on_extrapolation=warn_on_extrapolation,
        dtype=dtype,
    )
    regions: dict[int, PVTRegion] = {}
    if unit_system != temperature.unit_system:
        temperature = temperature.convert(unit_system)

    for region_idx in range(n_regions):
        pvtnum = region_idx + 1  # 1-based

        # Density record for this region
        density_record: dict[str, Number] | None = None
        if density_records is not None and region_idx < len(density_records):
            # Each DENSITY region entry is a list containing one row dict
            region_rows = density_records[region_idx]
            if region_rows:
                density_record = region_rows[0]

        # Oil Phase
        oil_data: PVTData | None = None
        if pvto_records is not None and region_idx < len(pvto_records):
            oil_data = build_oil_data_from_pvto(
                pvto_records=pvto_records[region_idx],
                density_record=density_record,
                temperature=temperature.region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
        elif pvco_records is not None and region_idx < len(pvco_records):
            # PVCO: live oil, one row per bubble point (constant undersaturated co / cv)
            oil_data = build_oil_data_from_pvco(
                pvco_records=pvco_records[region_idx],
                density_record=density_record,
                temperature=temperature.region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
        elif pvdo_records is not None and region_idx < len(pvdo_records):
            oil_data = build_oil_data_from_pvdo(
                pvdo_records=pvdo_records[region_idx],
                density_record=density_record,
                temperature=temperature.region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
        elif pvcdo_records is not None and region_idx < len(pvcdo_records):
            # PVCDO: single-record dead oil (constant compressibility / viscosibility)
            pvcdo_record = pvcdo_records[region_idx]
            if pvcdo_record:
                oil_data = build_oil_data_from_pvcdo(
                    pvcdo_record=pvcdo_record[0],
                    density_record=density_record,
                    temperature=temperature.region(pvtnum),
                    unit_system=unit_system,
                    interpolation_method=interpolation_method,
                    dtype=dtype,
                )

        # Gas Phase
        gas_data: PVTData | None = None
        if pvtg_records is not None and region_idx < len(pvtg_records):
            gas_data = build_gas_data_from_pvtg(
                pvtg_records=pvtg_records[region_idx],
                density_record=density_record,
                temperature=temperature.region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )
        elif pvdg_records is not None and region_idx < len(pvdg_records):
            gas_data = build_gas_data_from_pvdg(
                pvdg_records=pvdg_records[region_idx],
                density_record=density_record,
                temperature=temperature.region(pvtnum),
                unit_system=unit_system,
                interpolation_method=interpolation_method,
                dtype=dtype,
            )

        # Water
        water_data: PVTData | None = None
        salinity = 0.0  # Salinity is not stored in the PVTW record; default to 0 ppm
        if pvtw_records is not None and region_idx < len(pvtw_records):
            pvtw_rows = pvtw_records[region_idx]
            if pvtw_rows:
                water_data = build_water_data_from_pvtw(
                    pvtw_record=pvtw_rows[0],
                    density_record=density_record,
                    temperature=temperature.region(pvtnum),
                    unit_system=unit_system,
                    salinity=salinity,
                    interpolation_method=interpolation_method,
                    dtype=dtype,
                )

        # `StaticPVT` for this region
        # Resolve stock-tank densities from DENSITY record
        stock_tank_oil_density = density_record["oil"] if density_record is not None else None
        stock_tank_water_density = density_record["water"] if density_record is not None else None
        stock_tank_gas_density = density_record["gas"] if density_record is not None else None

        # PVTW scalars for this region
        water_reference_pressure: Number | None = None
        water_reference_fvf: Number | None = None
        water_reference_viscosity: Number | None = None
        water_reference_compressibility: Number | None = None
        water_viscosibility: Number | None = None
        if pvtw_records is not None and region_idx < len(pvtw_records):
            pvtw_rows = pvtw_records[region_idx]
            if pvtw_rows:
                pvtw_record = pvtw_rows[0]
                water_reference_pressure = pvtw_record["reference_pressure"]
                water_reference_fvf = pvtw_record["fvf"]
                water_reference_compressibility = pvtw_record["compressibility"]
                water_reference_viscosity = pvtw_record["viscosity"]
                water_viscosibility = pvtw_record.get("viscosibility", 0.0)

        static = StaticPVT(
            stock_tank_oil_density=stock_tank_oil_density,
            water_reference_pressure=water_reference_pressure,
            water_reference_fvf=water_reference_fvf,
            water_reference_viscosity=water_reference_viscosity,
            water_reference_compressibility=water_reference_compressibility,
            stock_tank_water_density=stock_tank_water_density,
            stock_tank_gas_density=stock_tank_gas_density,
            water_viscosibility=water_viscosibility,
            water_salinity=salinity,
            unit_system=unit_system,
        )

        # Assemble `PVTRegion`
        dataset = PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)
        tables = PVTTables.from_dataset(dataset, pvt=static, **table_kwargs)
        regions[pvtnum] = PVTRegion(static=static, tables=tables, unit_system=unit_system)

        logger.debug(
            "Built PVT tables and properties for region %d: oil=%s, gas=%s, water=%s, salinity=%.0f ppm",
            pvtnum,
            "`PVTO`"
            if pvto_records and region_idx < len(pvto_records)
            else "`PVCO`"
            if pvco_records and region_idx < len(pvco_records)
            else "`PVDO`"
            if pvdo_records and region_idx < len(pvdo_records)
            else "`PVCDO`"
            if pvcdo_records and region_idx < len(pvcdo_records)
            else "none",
            "`PVTG`"
            if pvtg_records and region_idx < len(pvtg_records)
            else "`PVDG`"
            if pvdg_records and region_idx < len(pvdg_records)
            else "none",
            "`PVTW`" if pvtw_records and region_idx < len(pvtw_records) else "none",
            salinity,
        )
    return regions
