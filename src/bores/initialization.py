"""
Utilities for reservoir equilibrium state initialization from EQUIL, explicit deck arrays, or user-supplied
arrays.

Precedence per field is:

```
RESTART (not yet supported but planned) -> explicit array/keyword -> EQUIL
```
"""

import typing
import warnings

import numpy as np
import numpy.typing as npt

from bores.blackoil.pvt.regions import PVT
from bores.blackoil.pvt.tables import PVTTable
from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
)
from bores.blackoil.satfunc.regions import SatFunc
from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.model import Reservoir
from bores.reservoir.state.base import Hysteresis, ReservoirState
from bores.reservoir.state.equilibrium import (
    DepthTable,
    Equilibrium,
    EquilibriumRegion,
)
from bores.reservoir.temperature import (
    Temperature,
    TemperatureGradient,
    TemperatureTable,
)
from bores.types import (
    CellArray,
    IntArray,
    IntCellArray,
    Number,
    NumberArray,
    OneDimension,
    UnitSystem,
)
from bores.utils import get_hydrostatic_gradient_factor

__all__ = ["EquilibriumArrays", "initialize_equilibrium_arrays", "initialize_reservoir_state"]

N_SATURATION_SAMPLES = 100
"""
Default sample count for the capillary-pressure inversion grid (see
`_invert_monotonic_curve`); overridable per-call via `saturation_samples`.
"""


class EquilibriumArrays(typing.NamedTuple):
    """Per-cell arrays produced by an equilibration algorithm for one region."""

    pressure: CellArray
    water_saturation: CellArray
    gas_saturation: CellArray
    solution_gor: CellArray
    oil_bubble_point_pressure: CellArray
    vaporized_oil_to_gas_ratio: CellArray
    gas_dew_point_pressure: CellArray


def _rk4_march(
    density_fn: typing.Callable[[Number, Number], Number],
    reference_depth: Number,
    reference_pressure: Number,
    endpoint_depth: Number,
    gradient_factor: Number,
    step: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Integrate `dP/dz = gradient_factor * density_fn(pressure, depth)` from `reference_depth` to
    `endpoint_depth` via fixed-step RK4.

    :returns: `(depths, pressures)`, both ascending in depth and spanning
        `[min(reference_depth, endpoint_depth), max(reference_depth, endpoint_depth)]`.
    """
    if np.isclose(reference_depth, endpoint_depth):
        return np.array([reference_depth], dtype=dtype), np.array(
            [reference_pressure], dtype=dtype
        )

    direction = 1.0 if endpoint_depth > reference_depth else -1.0
    n_steps = max(1, int(np.ceil(abs(endpoint_depth - reference_depth) / step)))
    h = direction * abs(endpoint_depth - reference_depth) / n_steps

    depths = np.empty(n_steps + 1, dtype=dtype)
    pressures = np.empty(n_steps + 1, dtype=dtype)
    depths[0], pressures[0] = reference_depth, reference_pressure
    z, p = reference_depth, reference_pressure
    for i in range(n_steps):
        k1 = gradient_factor * density_fn(p, z)
        k2 = gradient_factor * density_fn(p + 0.5 * h * k1, z + 0.5 * h)
        k3 = gradient_factor * density_fn(p + 0.5 * h * k2, z + 0.5 * h)
        k4 = gradient_factor * density_fn(p + h * k3, z + h)
        p = p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + h
        depths[i + 1], pressures[i + 1] = z, p

    if direction < 0:
        depths = depths[::-1]
        pressures = pressures[::-1]
    return depths, pressures


def _march_full_range(
    density_fn: typing.Callable[[Number, Number], Number],
    reference_depth: Number,
    reference_pressure: Number,
    min_depth: Number,
    max_depth: Number,
    gradient_factor: Number,
    step: Number,
    dtype: npt.DTypeLike,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    March `density_fn` outward from `reference_depth` in both directions, covering
    `[min_depth, max_depth]` in one sorted, concatenated profile.
    """
    depths_up, pressures_up = _rk4_march(
        density_fn=density_fn,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        endpoint_depth=min_depth,
        gradient_factor=gradient_factor,
        step=step,
        dtype=dtype,
    )
    depths_down, pressures_down = _rk4_march(
        density_fn=density_fn,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        endpoint_depth=max_depth,
        gradient_factor=gradient_factor,
        step=step,
        dtype=dtype,
    )
    depths = np.concatenate([depths_up[:-1], depths_down], dtype=dtype)
    pressures = np.concatenate([pressures_up[:-1], pressures_down], dtype=dtype)
    return typing.cast(NumberArray[OneDimension], depths), typing.cast(
        NumberArray[OneDimension], pressures
    )


def _invert_monotonic_curve(
    saturation_grid: NumberArray[OneDimension],
    capillary_pressure_grid: NumberArray[OneDimension],
    target_capillary_pressure: Number,
) -> Number:
    """
    Invert a (assumed monotonic) Pc(saturation) curve at a target Pc value.

    Sorts by `capillary_pressure_grid` before interpolating so this works regardless of
    whether Pc increases or decreases with saturation. This is not meaningful for
    non-monotonic curves and is not checked here, since every capillary pressure
    model in this codebase is monotonic in saturation by construction.
    """
    order = np.argsort(capillary_pressure_grid)
    return np.interp(
        target_capillary_pressure,
        capillary_pressure_grid[order],
        saturation_grid[order],
    )


def get_saturations_from_capillary_pressure(
    *,
    depths: CellArray,
    pressure: CellArray,
    connate_water_saturation: CellArray,
    residual_oil_saturation_water: CellArray,
    residual_oil_saturation_gas: CellArray,
    residual_gas_saturation: CellArray,
    capillary_pressure: CapillaryPressureTable,
    water_profile: tuple[npt.NDArray, npt.NDArray] | None,
    gas_profile: tuple[npt.NDArray, npt.NDArray] | None,
    dtype: npt.DTypeLike,
    saturation_samples: int = N_SATURATION_SAMPLES,
) -> tuple[CellArray, CellArray]:
    """
    Invert the region's `Pcow`/`Pcgo` curves against depth-varying `Pcow(z)`/
    `Pcgo(z)` to get a smooth `Sw(z)`/`Sg(z)`, rather than a sharp step at the
    contacts.

    `water_profile`/`gas_profile` are full-column `(depth, pressure)` `Pw(z)`/
    `Pg(z)` profiles (see `_march_full_range`); `None` when the region has no
    WOC/GOC respectively.

    :param saturation_samples: Number of samples used to grid the
        capillary-pressure inversion curve. Higher values give a smoother,
        more accurate `Sw(z)`/`Sg(z)` inversion at proportionally higher cost;
        defaults to `N_SATURATION_SAMPLES`.
    """
    n = len(depths)
    swc = connate_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    water_saturation = swc.astype(dtype, copy=False).copy()
    gas_saturation = np.zeros(n, dtype=dtype)

    if water_profile is not None:
        water_pressure_at_depth = np.interp(depths, water_profile[0], water_profile[1])
        target_pcow = pressure - water_pressure_at_depth  # Pcow = Po - Pw
        fractions = np.linspace(0.0, 1.0, saturation_samples, dtype=dtype)
        water_saturation_grid = swc[:, None] + fractions[None, :] * (1.0 - swc - sorw)[:, None]
        for i in range(n):
            pcow_grid = capillary_pressure.evaluate(
                water_saturation_grid[i],
                1.0 - water_saturation_grid[i],
                np.zeros(saturation_samples, dtype=dtype),
                irreducible_water_saturation=swc[i],
                residual_oil_saturation_water=sorw[i],
                residual_oil_saturation_gas=sorg[i],
                residual_gas_saturation=sgr[i],
            )["oil_water"]
            pcow_grid = typing.cast(NumberArray[OneDimension], pcow_grid)
            water_saturation[i] = _invert_monotonic_curve(
                water_saturation_grid[i], pcow_grid, target_pcow[i]
            )

    if gas_profile is not None:
        gas_pressure_at_depth = np.interp(depths, gas_profile[0], gas_profile[1])
        target_pcgo = gas_pressure_at_depth - pressure  # Pcgo = Pg - Po
        fractions = np.linspace(0.0, 1.0, saturation_samples, dtype=dtype)
        gas_saturation_grid = sgr[:, None] + fractions[None, :] * (1.0 - swc - sorg - sgr)[:, None]
        for i in range(n):
            pcgo_grid = capillary_pressure.evaluate(
                np.full(saturation_samples, swc[i], dtype=dtype),
                1.0 - swc[i] - gas_saturation_grid[i],
                gas_saturation_grid[i],
                irreducible_water_saturation=swc[i],
                residual_oil_saturation_water=sorw[i],
                residual_oil_saturation_gas=sorg[i],
                residual_gas_saturation=sgr[i],
            )["gas_oil"]
            pcgo_grid = typing.cast(NumberArray[OneDimension], pcgo_grid)
            gas_saturation[i] = _invert_monotonic_curve(
                gas_saturation_grid[i], pcgo_grid, target_pcgo[i]
            )

    return (
        typing.cast(CellArray, water_saturation),
        typing.cast(CellArray, gas_saturation),
    )


def initialize_center_point_equilibrium(
    *,
    region: EquilibriumRegion,
    depths: CellArray,
    connate_water_saturation: CellArray,
    oil_table: PVTTable,
    gas_table: PVTTable | None,
    water_table: PVTTable | None,
    rsvd_table: DepthTable | None,
    rvvd_table: DepthTable | None,
    temperature: CellArray,
    unit_system: UnitSystem,
    capillary_pressure: CapillaryPressureTable | None = None,
    residual_oil_saturation_water: CellArray | None = None,
    residual_oil_saturation_gas: CellArray | None = None,
    residual_gas_saturation: CellArray | None = None,
    depth_step: Number = 1.0,
    dtype: npt.DTypeLike = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Center-point `EQUIL` initialization: evaluate pressure and saturation once
    at each cell's centroid depth (`region.accuracy_flag == 0`).

    :param saturation_samples: Forwarded to
        `get_saturations_from_capillary_pressure` when `capillary_pressure`
        is supplied; ignored for sharp-contact initialization.

    :raises ValidationError: If `datum_depth` is outside the oil zone, or a
        contact is present without its corresponding PVT table.
    :raises NotImplementedError: If the gas table is wet-gas (`PVTG`-based)
        and no `rvvd_table` is supplied.
    """
    is_wet_gas = gas_table is not None and gas_table.has("vaporized_oil_to_gas_ratio")
    if is_wet_gas and rvvd_table is None:
        raise NotImplementedError(
            "Wet-gas / gas-condensate `EQUIL` initialization without an `RVVD` table "
            "is not yet supported (no way to determine Rv(depth) to query the "
            "gas PVT table's Rv-indexed axis). Supply an `RVVD` table for this "
            "region, use a `PVDG` (dry-gas) table, or supply pressure/saturations "
            "explicitly for this region."
        )

    gradient_factor = get_hydrostatic_gradient_factor(unit_system)
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    oil_zone_low = region.goc_depth if region.has_goc else min(depths.min(), region.datum_depth)
    oil_zone_high = region.woc_depth if region.has_woc else max(depths.max(), region.datum_depth)
    if not (oil_zone_low <= region.datum_depth <= oil_zone_high):
        raise ValidationError(
            f"`EQUIL` `datum_depth` ({region.datum_depth}) must lie within the oil "
            f"zone [{oil_zone_low}, {oil_zone_high}] for center-point "
            "initialization; datum points in the gas cap or aquifer are not "
            "yet supported."
        )

    order = np.argsort(depths)
    sorted_depths = depths[order]
    sorted_temperature = temperature[order]

    def get_temperature_at(depth: Number) -> Number:
        return np.interp(depth, sorted_depths, sorted_temperature)

    def compute_oil_density(pressure: Number, depth: Number) -> Number:
        return oil_table.density(pressure, get_temperature_at(depth)).astype(  # type: ignore[union-attr, return-value]
            dtype, copy=False
        )

    def compute_gas_density(pressure: Number, depth: Number) -> Number:
        assert gas_table is not None
        return gas_table.density(pressure, get_temperature_at(depth)).astype(  # type: ignore[union-attr, return-value]
            dtype, copy=False
        )

    def compute_water_density(pressure: Number, depth: Number) -> Number:
        assert water_table is not None
        return water_table.density(pressure, get_temperature_at(depth)).astype(  # type: ignore[union-attr, return-value]
            dtype, copy=False
        )

    min_depth = min(depths.min(), region.datum_depth, oil_zone_low)
    max_depth = max(depths.max(), region.datum_depth, oil_zone_high)

    # Oil zone: integrate from datum outward to the zone edges.
    oil_depths_up, oil_pressures_up = _rk4_march(
        density_fn=compute_oil_density,
        reference_depth=region.datum_depth,
        reference_pressure=region.datum_pressure,
        endpoint_depth=oil_zone_low,
        gradient_factor=gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    oil_depths_down, oil_pressures_down = _rk4_march(
        density_fn=compute_oil_density,
        reference_depth=region.datum_depth,
        reference_pressure=region.datum_pressure,
        endpoint_depth=oil_zone_high,
        gradient_factor=gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    all_depths = [oil_depths_up[:-1], oil_depths_down]
    all_pressures = [oil_pressures_up[:-1], oil_pressures_down]

    # Gas cap: full-range Pg(z) march from GOC, re-referenced to Po via pcgo_at_goc.
    # Full range (not just the gas-cap zone) so Pcgo(z) is defined everywhere,
    # which the capillary-pressure saturation path needs.
    gas_profile: tuple[npt.NDArray, npt.NDArray] | None = None
    if region.has_goc:
        if gas_table is None:
            raise ValidationError(
                "Region has a gas-oil contact but no gas PVT table was supplied."
            )
        gas_pressure_at_goc = (
            np.interp(region.goc_depth, oil_depths_up, oil_pressures_up) + region.pcgo_at_goc
        )
        gas_depths_full, gas_pressures_full = _march_full_range(
            density_fn=compute_gas_density,
            reference_depth=region.goc_depth,
            reference_pressure=gas_pressure_at_goc,
            min_depth=min_depth,
            max_depth=max_depth,
            gradient_factor=gradient_factor,
            step=depth_step,
            dtype=dtype,
        )
        gas_profile = (gas_depths_full, gas_pressures_full)
        in_gas_zone = gas_depths_full < region.goc_depth
        if min_depth < region.goc_depth:
            all_depths.append(gas_depths_full[in_gas_zone])
            all_pressures.append(gas_pressures_full[in_gas_zone] - region.pcgo_at_goc)

    # Aquifer: full-range Pw(z) march from WOC, re-referenced to Po via pcow_at_woc.
    water_profile: tuple[npt.NDArray, npt.NDArray] | None = None
    if region.has_woc:
        if water_table is None:
            raise ValidationError(
                "Region has a water-oil contact but no water PVT table was supplied."
            )
        water_pressure_at_woc = (
            np.interp(region.woc_depth, oil_depths_down, oil_pressures_down) - region.pcow_at_woc
        )
        water_depths_full, water_pressures_full = _march_full_range(
            density_fn=compute_water_density,
            reference_depth=region.woc_depth,
            reference_pressure=water_pressure_at_woc,
            min_depth=min_depth,
            max_depth=max_depth,
            gradient_factor=gradient_factor,
            step=depth_step,
            dtype=dtype,
        )
        water_profile = (water_depths_full, water_pressures_full)
        in_aquifer = water_depths_full > region.woc_depth
        if max_depth > region.woc_depth:
            all_depths.append(water_depths_full[in_aquifer])
            all_pressures.append(water_pressures_full[in_aquifer] + region.pcow_at_woc)

    grid_depths = np.concatenate(all_depths, dtype=dtype)
    grid_pressures = np.concatenate(all_pressures, dtype=dtype)
    grid_order = np.argsort(grid_depths)
    grid_depths = grid_depths[grid_order]
    grid_pressures = grid_pressures[grid_order]

    pressure = typing.cast(
        CellArray,
        np.interp(depths, grid_depths, grid_pressures).astype(dtype, copy=False),
    )

    n = len(depths)
    if capillary_pressure is not None:
        water_saturation, gas_saturation = get_saturations_from_capillary_pressure(
            depths=depths,
            pressure=pressure,
            connate_water_saturation=connate_water_saturation,
            residual_oil_saturation_water=(
                residual_oil_saturation_water
                if residual_oil_saturation_water is not None
                else np.zeros(n, dtype=dtype)
            ),
            residual_oil_saturation_gas=(
                residual_oil_saturation_gas
                if residual_oil_saturation_gas is not None
                else np.zeros(n, dtype=dtype)
            ),
            residual_gas_saturation=(
                residual_gas_saturation
                if residual_gas_saturation is not None
                else np.zeros(n, dtype=dtype)
            ),
            capillary_pressure=capillary_pressure,
            water_profile=water_profile,
            gas_profile=gas_profile,
            dtype=dtype,
            saturation_samples=saturation_samples,
        )
    else:
        # Sharp contact: Sw = 1 below WOC, Sw = swc in the oil zone,
        # Sg = 1 - swc above GOC. No smooth transition zone.
        water_saturation = connate_water_saturation.astype(dtype, copy=False).copy()
        gas_saturation = np.zeros(n, dtype=dtype)
        if region.has_goc:
            in_gas_cap = depths < region.goc_depth
            gas_saturation[in_gas_cap] = 1.0 - connate_water_saturation[in_gas_cap]
            water_saturation[in_gas_cap] = connate_water_saturation[in_gas_cap]
        if region.has_woc:
            in_aquifer = depths > region.woc_depth
            water_saturation[in_aquifer] = 1.0
            gas_saturation[in_aquifer] = 0.0

    has_oil = (1.0 - water_saturation - gas_saturation) > 0.0

    # Solution GOR / bubble-point pressure
    solution_gor = np.zeros(n, dtype=dtype)
    oil_bubble_point_pressure = np.zeros(n, dtype=dtype)
    if np.any(has_oil):
        oil_pressure = pressure[has_oil]
        oil_temperature = temperature[has_oil]
        if rsvd_table is not None:
            gor = rsvd_table.at_depth(depths[has_oil]).astype(  # type: ignore[arg-type]
                dtype, copy=False
            )
            bubble_point_pressure = oil_table.bubble_point_pressure(
                temperature=oil_temperature, solution_gor=gor
            ).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            # No RSVD: assume saturated oil (Pbub = P) everywhere in the oil
            # zone. Forcing `bubble_point_pressure=oil_pressure` makes the saturated
            # branch always trigger, reading Rs straight off the table at P.
            gor = oil_table.solution_gas_to_oil_ratio(
                pressure=oil_pressure,
                temperature=oil_temperature,
                bubble_point_pressure=oil_pressure,
            ).astype(dtype, copy=False)  # type: ignore[union-attr]
            bubble_point_pressure = oil_pressure.copy()

        solution_gor[has_oil] = gor
        oil_bubble_point_pressure[has_oil] = bubble_point_pressure

    # Vaporized oil ratio / dew-point pressure (dry-gas: Rv = 0 unless RVVD)
    vaporized_oil_to_gas_ratio = np.zeros(n, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n, dtype=dtype)
    has_free_gas = gas_saturation > 0.0
    if rvvd_table is not None and np.any(has_free_gas):
        vaporized_oil_to_gas_ratio[has_free_gas] = rvvd_table.at_depth(
            depths[has_free_gas]  # type: ignore[arg-type]
        ).astype(dtype, copy=False)  # type: ignore[union-attr]

        if gas_table is not None:
            dew_point = gas_table.dew_point_pressure(temperature=temperature[has_free_gas])
            if dew_point is not None:
                gas_dew_point_pressure[has_free_gas] = dew_point.astype(  # type: ignore[union-attr]
                    dtype, copy=False
                )

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_to_gas_ratio=typing.cast(CellArray, vaporized_oil_to_gas_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


def initialize_horizontal_subdivision_equilibrium(
    *,
    region: EquilibriumRegion,
    cell_top: CellArray,
    cell_bottom: CellArray,
    n_subdivisions: int | None = None,
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Vertical-equilibrium averaging via `N` horizontal sub-divisions per cell
    (`region.accuracy_flag > 0`).

    Samples `N` depths between each cell's top/bottom (from the grid's
    per-cell AABB), evaluates `initialize_center_point_equilibrium` at all
    of them, and averages back down to one value per cell. Arithmetic mean
    assumes uniform cross-sectional area across depth - doesn't account for
    cell dip (see the tilted variant).
    """
    n_sub = n_subdivisions or abs(region.accuracy_flag)
    n_cells = len(cell_top)
    fractions = (np.arange(n_sub, dtype=cell_top.dtype) + 0.5) / n_sub
    sub_depth = cell_top[:, None] + fractions[None, :] * (cell_bottom - cell_top)[:, None]

    repeated_kwargs = {
        key: np.repeat(value, n_sub)
        if isinstance(value, np.ndarray) and value.shape == (n_cells,)
        else value
        for key, value in kwargs.items()
    }
    sub_arrays = initialize_center_point_equilibrium(
        region=region,
        depths=sub_depth.ravel(),
        **repeated_kwargs,  # type: ignore[arg-type]
    )

    def get_average(field: CellArray) -> CellArray:
        return typing.cast(CellArray, field.reshape(n_cells, n_sub).mean(axis=1))

    return EquilibriumArrays(**{
        name: get_average(getattr(sub_arrays, name)) for name in EquilibriumArrays._fields
    })


def get_dip_aware_top_bottom_faces(
    grid: Grid, cell_indices: IntArray[OneDimension]
) -> tuple[
    NumberArray[OneDimension],
    NumberArray[OneDimension],
    NumberArray[OneDimension],
    NumberArray[OneDimension],
]:
    """
    For each cell, find its true top/bottom face depths and areas from grid
    geometry, rather than the cell's axis-aligned bounding box.

    Each cell's own faces are inspected (via `cell_face_offsets`/
    `cell_face_indices`) and classified as top/bottom candidates by their
    outward-normal z-component (`Grid.get_face_normal_for_cell`): a face
    whose normal is within 45 degrees of vertical is a candidate, since
    `z` is positive downward, an upward-pointing (negative-z) normal marks a
    top face and a downward-pointing (positive-z) normal marks a bottom
    face. Candidates are split into a shallow group and a deep group at
    their midpoint depth, and each group's area-weighted centroid depth and
    total area become that cell's top/bottom depth/area. This resolves
    correctly for a single quad top/bottom face (structured/corner-point
    hexahedra) as well as a top/bottom face triangulated into several
    polygons (general unstructured cells).

    A dipping cell's true top-to-bottom depth span (and area taper) along
    its own face-normal direction can differ materially from its AABB span,
    which is inflated by the lateral spread of all of its vertices; using
    the actual faces is what makes this dip-aware.

    Falls back to the cell's AABB z-range with equal top/bottom areas
    (i.e. uniform-weighting downstream) for any cell with fewer than two
    vertical-ish faces - e.g. a degenerate sliver cell, or a 2-D grid with
    no k-direction faces at all.

    :returns: `(top_depth, bottom_depth, top_area, bottom_area)`, each shape
        `(len(cell_indices),)`, `float64`.
    """
    n = len(cell_indices)
    # Use float64 here since the grid arrays are mostly in float64
    # also the logic here is precision snensitive. W ecan cast to
    # the preferred dtype in the caller
    top_depth = np.empty(n, dtype=np.float64)
    bottom_depth = np.empty(n, dtype=np.float64)
    top_area = np.empty(n, dtype=np.float64)
    bottom_area = np.empty(n, dtype=np.float64)

    min_abs_normal_z = np.cos(np.deg2rad(45.0))

    for i in range(n):
        cell_index = int(cell_indices[i])
        start = grid.cell_face_offsets[cell_index]
        end = grid.cell_face_offsets[cell_index + 1]
        faces = grid.cell_face_indices[start:end]

        candidate_depths = []
        candidate_areas = []
        for face_index in faces:
            face_index = int(face_index)
            normal = grid.get_face_normal_for_cell(face_index, cell_index)
            if abs(normal[2]) >= min_abs_normal_z:
                candidate_depths.append(grid.face_centroids[face_index, 2])
                candidate_areas.append(grid.face_areas[face_index])

        if len(candidate_depths) >= 2:
            depths_arr = np.asarray(candidate_depths, dtype=np.float64)
            areas_arr = np.asarray(candidate_areas, dtype=np.float64)
            midpoint = 0.5 * (depths_arr.min() + depths_arr.max())
            is_top = depths_arr <= midpoint
            if not np.any(is_top) or np.all(is_top):
                # All candidates landed on one side of the midpoint (e.g.
                # a single degenerate face), so we split by the extremes instead.
                is_top = depths_arr == depths_arr.min()
            is_bottom = ~is_top

            top_depth[i] = np.average(depths_arr[is_top], weights=areas_arr[is_top])
            bottom_depth[i] = np.average(depths_arr[is_bottom], weights=areas_arr[is_bottom])
            top_area[i] = areas_arr[is_top].sum()
            bottom_area[i] = areas_arr[is_bottom].sum()
        else:
            top_depth[i] = grid.cell_min_xyz[cell_index, 2]
            bottom_depth[i] = grid.cell_max_xyz[cell_index, 2]
            top_area[i] = bottom_area[i] = 1.0

    return (
        typing.cast(NumberArray[OneDimension], top_depth),
        typing.cast(NumberArray[OneDimension], bottom_depth),
        typing.cast(NumberArray[OneDimension], top_area),
        typing.cast(NumberArray[OneDimension], bottom_area),
    )


def initialize_tilted_subdivision_equilibrium(
    *,
    region: EquilibriumRegion,
    grid: Grid,
    cell_indices: IntArray[OneDimension],
    n_subdivisions: int | None = None,
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Tilted-cell variant of horizontal subdivision, additionally accounting
    for cell dip (`region.accuracy_flag < 0`).

    Uses each cell's actual top/bottom face geometry (see
    `get_dip_aware_top_bottom_faces`) rather than its AABB to get a
    dip-aware depth range, then weights the `N` sub-samples by a linear
    taper between the top and bottom face areas instead of averaging them
    with equal weight. This is a volume-weighted average under a
    linear-taper (frustum-like) cross-section model, which is exact for a
    cell whose cross-sectional area truly varies linearly from top to
    bottom and a first-order improvement otherwise; it is not a full
    polyhedron/plane-intersection volume average; genuinely irregular
    (e.g. non-monotonic) cross-sections along the cell's vertical axis are
    still only approximated.
    """
    n_sub = n_subdivisions or abs(region.accuracy_flag)
    n_cells = len(cell_indices)

    top_depth, bottom_depth, top_area, bottom_area = get_dip_aware_top_bottom_faces(
        grid=grid, cell_indices=cell_indices
    )
    dtype = kwargs["dtype"]
    top_depth = top_depth.astype(dtype, copy=False)
    bottom_depth = bottom_depth.astype(dtype, copy=False)
    top_area = top_area.astype(dtype, copy=False)
    bottom_area = bottom_area.astype(dtype, copy=False)

    fractions = (np.arange(n_sub, dtype=dtype) + 0.5) / n_sub
    sub_depth = top_depth[:, None] + fractions[None, :] * (bottom_depth - top_depth)[:, None]
    area_weights = top_area[:, None] + fractions[None, :] * (bottom_area - top_area)[:, None]
    # Guards against a degenerate zero/negative taper (e.g. a pinched-out
    # face) collapsing the weighted average to NaN.
    area_weights = np.clip(area_weights, np.finfo(dtype).tiny, None)

    repeated_kwargs = {
        key: np.repeat(value, n_sub)
        if isinstance(value, np.ndarray) and value.shape == (n_cells,)
        else value
        for key, value in kwargs.items()
    }
    sub_arrays = initialize_center_point_equilibrium(
        region=region,
        depths=sub_depth.ravel(),
        **repeated_kwargs,  # type: ignore[arg-type]
    )

    def get_weighted_average(field: CellArray) -> CellArray:
        field_2d = field.reshape(n_cells, n_sub)
        return typing.cast(
            CellArray,
            np.sum(field_2d * area_weights, axis=1) / np.sum(area_weights, axis=1),
        )

    return EquilibriumArrays(**{
        name: get_weighted_average(getattr(sub_arrays, name)) for name in EquilibriumArrays._fields
    })


def initialize_equilibrium_arrays(
    reservoir: Reservoir,
    pvt: PVT,
    equilibrium: Equilibrium,
    temperature: CellArray,
    *,
    satfunc: SatFunc | None = None,
    depth_step: Number = 1.0,
    dtype: npt.DTypeLike = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
) -> EquilibriumArrays:
    """
    Compute `EQUIL`-derived pressure/saturation/Rs/Rv arrays for every cell.

    Dispatches each `EQLNUM` region to the algorithm selected by that
    region's `accuracy_flag` and assembles the full-grid arrays.

    :param satfunc: Optional `SatFunc`; when given, saturations
        are computed via capillary-pressure inversion instead of a sharp
        contact (see `get_saturations_from_capillary_pressure`). Doing so
        requires a `SATNUM` region per cell to select the right
        capillary-pressure curve; if `reservoir.regions.saturation_region`
        is unavailable, every cell defaults to saturation region 1 and a
        `UserWarning` is raised (see `Warns`).
    :param saturation_samples: Sample count for the capillary-pressure
        inversion grid, forwarded to
        `get_saturations_from_capillary_pressure` whenever `satfunc` is
        supplied; ignored for sharp-contact initialization. Defaults to
        `N_SATURATION_SAMPLES`.
    :raises ValidationError: If `reservoir`, `pvt`, `equilibrium`, or
        `satfunc` (when supplied) don't all share the same unit system;
        if a cell's `EQLNUM` has no matching `EquilibriumRegion`, an
        `EQLNUM` region spans more than one `PVTNUM`/`SATNUM` region, or a
        required PVT table is missing.
    :warns UserWarning: If `satfunc` is supplied but
        `reservoir.regions.saturation_region` (`SATNUM`) is unavailable;
        every cell is assigned to saturation region 1 in that case. Supply
        `reservoir.regions.saturation_region` explicitly (e.g.
        `np.ones(n_cells, dtype=np.int32)` for a single-region reservoir)
        to make the assignment explicit and silence the warning.
    """
    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    if pvt.unit_system != reservoir.unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({reservoir.unit_system.value!r}) does not "
            f"match `pvt.unit_system` ({pvt.unit_system.value!r})."
        )
    if equilibrium.unit_system != reservoir.unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({reservoir.unit_system.value!r}) does not "
            f"match `equilibrium.unit_system` ({equilibrium.unit_system.value!r})."
        )
    if (
        satfunc is not None
        and satfunc.unit_system is not None
        and satfunc.unit_system != reservoir.unit_system
    ):
        raise ValidationError(
            f"`reservoir.unit_system` ({reservoir.unit_system.value!r}) does not "
            f"match `satfunc.unit_system` ({satfunc.unit_system.value!r})."
        )
    equilibrium_region_index: IntCellArray = (
        reservoir.regions.equilibrium_region
        if reservoir.regions is not None and reservoir.regions.equilibrium_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    pvt_region_index: IntCellArray = (
        reservoir.regions.pvt_region
        if reservoir.regions is not None and reservoir.regions.pvt_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    saturation_region: IntCellArray | None = (
        reservoir.regions.saturation_region if reservoir.regions is not None else None
    )
    if satfunc is not None and saturation_region is None:
        warnings.warn(
            "`satfunc` was supplied but `reservoir.regions.saturation_region` "
            "(`SATNUM`) is unavailable; defaulting every cell to saturation region 1. "
            "Set `reservoir.regions.saturation_region` explicitly (e.g. "
            "`np.ones(reservoir.n_cells, dtype=np.int32)` for a single-region "
            "reservoir) to make this assignment explicit and silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        saturation_region = typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))

    depth = reservoir.depth
    connate_water_saturation = reservoir.rock.connate_water_saturation
    residual_oil_saturation_water = reservoir.rock.residual_oil_saturation_water_flood
    residual_oil_saturation_gas = reservoir.rock.residual_oil_saturation_gas_flood
    residual_gas_saturation = reservoir.rock.residual_gas_saturation

    pressure = np.zeros(n_cells, dtype=dtype)
    water_saturation = np.zeros(n_cells, dtype=dtype)
    gas_saturation = np.zeros(n_cells, dtype=dtype)
    solution_gor = np.zeros(n_cells, dtype=dtype)
    oil_bubble_point_pressure = np.zeros(n_cells, dtype=dtype)
    vaporized_oil_to_gas_ratio = np.zeros(n_cells, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n_cells, dtype=dtype)

    for eqlnum in np.unique(equilibrium_region_index):
        mask = equilibrium_region_index == eqlnum
        equilibrium_region = equilibrium.region(int(eqlnum))

        region_pvtnum = np.unique(pvt_region_index[mask])
        if len(region_pvtnum) != 1:
            raise ValidationError(
                f"`EQLNUM` {eqlnum} cells span multiple `PVTNUM` regions "
                f"({sorted(region_pvtnum.tolist())}); each equilibration "
                "region must map to exactly one PVT region for "
                "`initialize_equilibrium_arrays`."
            )
        pvt_region = pvt.region(region_pvtnum[0])

        capillary_pressure: CapillaryPressureTable | None = None
        if satfunc is not None:
            # Guaranteed non-`None` here: defaulted to region 1 (with a
            # warning) above whenever `satfunc` is supplied.
            assert saturation_region is not None
            region_satnum = np.unique(saturation_region[mask])
            if len(region_satnum) != 1:
                raise ValidationError(
                    f"`EQLNUM` {eqlnum} cells span multiple `SATNUM` regions "
                    f"({sorted(region_satnum.tolist())}); each equilibration "
                    "region must map to exactly one saturation-function "
                    "region for capillary-pressure-based initialization."
                )
            capillary_pressure = satfunc.region(region_satnum[0]).capillary_pressure

        rsvd_table = (
            equilibrium.rsvd_tables.get(equilibrium_region.rsvd_table)
            if equilibrium_region.uses_rsvd and equilibrium.rsvd_tables
            else None
        )
        rvvd_table = (
            equilibrium.rvvd_tables.get(equilibrium_region.rvvd_table)
            if equilibrium_region.uses_rvvd and equilibrium.rvvd_tables
            else None
        )

        common_kwargs: dict[str, typing.Any] = dict(
            region=equilibrium_region,
            connate_water_saturation=connate_water_saturation[mask],
            oil_table=pvt_region.tables.oil,
            gas_table=pvt_region.tables.gas,
            water_table=pvt_region.tables.water,
            rsvd_table=rsvd_table,
            rvvd_table=rvvd_table,
            temperature=temperature[mask],
            unit_system=reservoir.unit_system,
            capillary_pressure=capillary_pressure,
            residual_oil_saturation_water=residual_oil_saturation_water[mask],
            residual_oil_saturation_gas=residual_oil_saturation_gas[mask],
            residual_gas_saturation=residual_gas_saturation[mask],
            depth_step=depth_step,
            dtype=dtype,
            saturation_samples=saturation_samples,
        )

        if equilibrium_region.accuracy_flag == 0:
            region_arrays = initialize_center_point_equilibrium(
                depths=depth[mask],  # type: ignore[arg-type]
                **common_kwargs,
            )
        elif equilibrium_region.accuracy_flag > 0:
            region_arrays = initialize_horizontal_subdivision_equilibrium(
                cell_top=reservoir.grid.cell_min_xyz[mask, 2],  # type: ignore[arg-type]
                cell_bottom=reservoir.grid.cell_max_xyz[mask, 2],  # type: ignore[arg-type]
                **common_kwargs,
            )
        else:
            region_arrays = initialize_tilted_subdivision_equilibrium(
                grid=reservoir.grid,
                cell_indices=np.nonzero(mask)[0],  # type: ignore[arg-type]
                **common_kwargs,
            )

        pressure[mask] = region_arrays.pressure
        water_saturation[mask] = region_arrays.water_saturation
        gas_saturation[mask] = region_arrays.gas_saturation
        solution_gor[mask] = region_arrays.solution_gor
        oil_bubble_point_pressure[mask] = region_arrays.oil_bubble_point_pressure
        vaporized_oil_to_gas_ratio[mask] = region_arrays.vaporized_oil_to_gas_ratio
        gas_dew_point_pressure[mask] = region_arrays.gas_dew_point_pressure

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_to_gas_ratio=typing.cast(CellArray, vaporized_oil_to_gas_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


def get_temperature_array_from_regions(
    temperature: Temperature,
    pvt_region_index: IntCellArray,
    depth: CellArray,
    dtype: npt.DTypeLike = None,
) -> CellArray:
    """
    Evaluate a `Temperature` at every cell's depth, grouped by `PVTNUM` (the
    same grain `PVT.from_deck` resolves `Temperature` against, so PVT
    tables and this array stay consistent).

    Duck-types each region's spec. A bare number is broadcast, anything
    with an `at_depth` method (e.g. a depth-dependent table) is evaluated
    per cell.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    result = np.empty(len(pvt_region_index), dtype=dtype)
    for pvtnum in np.unique(pvt_region_index):
        mask = pvt_region_index == pvtnum
        spec = temperature.region(int(pvtnum))
        if isinstance(spec, (TemperatureGradient, TemperatureTable)):
            result[mask] = spec.at_depth(depth[mask]).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            result[mask] = spec  # type: ignore[arg-type]
    return typing.cast(CellArray, result)


def resolve_temperature(
    reservoir: Reservoir,
    deck_file: DeckFile | None,
    temperature: Temperature | Number | None,
    dtype: npt.DTypeLike = None,
) -> CellArray:
    """
    Resolve per-cell temperature. Explicit `temperature` kwarg > deck
    `RTEMP`/`TEMPVD` > error. (Restart temperature is not yet supported.)
    """
    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    source: Temperature | None = None
    if isinstance(temperature, Temperature):
        source = temperature
    elif temperature is not None:
        return typing.cast(CellArray, np.full(n_cells, temperature, dtype=dtype))
    elif deck_file is not None and (
        deck_file.has("RTEMP") or deck_file.has("TEMPVD") or deck_file.has("RTEMPVD")
    ):
        source = Temperature.from_deck(deck_file, dtype=dtype)
    else:
        raise ValidationError(
            "No temperature source available. Pass `temperature=` explicitly "
            "(a constant, or a `Temperature`), or supply a `deck_file` "
            "with `RTEMP`/`TEMPVD`."
        )

    pvt_region_index: IntCellArray = (
        reservoir.regions.pvt_region
        if reservoir.regions is not None and reservoir.regions.pvt_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    return get_temperature_array_from_regions(
        source,
        pvt_region_index=pvt_region_index,
        depth=reservoir.depth,
        dtype=dtype,
    )


def initialize_reservoir_state(
    reservoir: Reservoir,
    pvt: PVT,
    *,
    deck_file: DeckFile | None = None,
    equilibrium: Equilibrium | None = None,
    satfunc: SatFunc | None = None,
    temperature: Temperature | Number | None = None,
    pressure: CellArray | None = None,
    water_saturation: CellArray | None = None,
    gas_saturation: CellArray | None = None,
    solution_gor: CellArray | None = None,
    vaporized_oil_to_gas_ratio: CellArray | None = None,
    depth_step: Number = 1.0,
    with_hysteresis: bool = False,
    dtype: npt.DTypeLike = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
) -> ReservoirState:
    """
    Build a complete `ReservoirState` from equilibration data (e.g `EQUIL`) and/or explicit arrays.

    "Explicit array/keyword" means either the corresponding kwarg here, or
    (if the kwarg is `None`) the matching `PRESSURE`/`SWAT`/`SGAS`/`RS`/`RV`
    keyword in `deck_file`. Any field left uncovered by both is filled in
    from equilibration, which therefore only runs at all if at least one
    field needs it. `oil_saturation` is always derived as
    `1 - water_saturation - gas_saturation`, never taken from an explicit array.

    `oil_bubble_point_pressure` and `gas_dew_point_pressure` are taken from
    equilibration when it runs. Otherwise (fully explicit arrays, no `EQUIL`
    involved), each `PVTNUM` region's own PVT tables are queried per-cell -
    `oil_bubble_point_pressure` from `PVTTable.bubble_point_pressure(temperature,
    solution_gor)` on the oil table, `gas_dew_point_pressure` from
    `PVTTable.dew_point_pressure(temperature)` on the gas table - and only
    fall back to an assumption where a table has no such data: saturated oil
    (`oil_bubble_point_pressure = pressure`) and no free gas ever
    (`gas_dew_point_pressure = 0`).

    Note:
    Ensure the unit system of all inputs (`reservoir`, `pvt`, ..., and any explicit arrays) is consistent.
    The function will raise a `ValidationError` if there is a detected mismatch.

    :param reservoir: Reservoir geometry and rock properties.
    :param pvt: PVT tables for all regions.
    :param deck_file: Optional `DeckFile` to read explicit arrays from.
    :param equilibrium: Optional `Equilibrium` for any fields not covered by an
        explicit array/keyword.
    :param satfunc: Optional `SatFunc` for capillary-pressure-based
        saturations instead of a sharp contact. If supplied and
        `reservoir.regions.saturation_region` (SATNUM) is unavailable, every
        cell defaults to saturation region 1 and a `UserWarning` is raised
        (see `initialize_equilibrium_arrays`).
    :param temperature: Optional per-cell temperature (constant or `Temperature`)
        for equilibration. If `None`, will be read from `deck_file` if available.
    :param pressure: Optional explicit pressure array (overrides `PRESSURE` keyword).
    :param water_saturation: Optional explicit water saturation array (overrides `SWAT` keyword).
    :param gas_saturation: Optional explicit gas saturation array (overrides `SGAS` keyword).
    :param solution_gor: Optional explicit solution GOR array (overrides `RS` keyword).
    :param vaporized_oil_to_gas_ratio: Optional explicit vaporized oil ratio array (overrides `RV` keyword).
    :param depth_step: Step size for hydrostatic integration in equilibration.
    :param with_hysteresis: If `True`, initialize hysteresis state from the
        resolved saturations (`Hysteresis.from_initial_saturation`) for
        later scanning-curve tracking. Works regardless of whether
        saturations came from a sharp contact or capillary-pressure
        inversion; `satfunc` is not itself required.
    :param dtype: Preferred dtype for the returned `ReservoirState`. Defaults to `get_dtype()`
    :param saturation_samples: Sample count for the capillary-pressure
        inversion grid, forwarded to `initialize_equilibrium_arrays` when
        `satfunc` is supplied. Defaults to `N_SATURATION_SAMPLES`.
    :raises NotImplementedError: If `deck_file` has a `RESTART` keyword
        (not yet supported).
    :raises ValidationError: If some field is covered by neither an
        explicit array/keyword nor `equilibrium`/`deck_file`, if `reservoir`
        and `pvt` unit systems disagree, or if `equilibrium`/`satfunc`
        (when supplied) don't share that same unit system, or if
        saturations are physically inconsistent.
    :warns UserWarning: If `satfunc` is supplied but
        `reservoir.regions.saturation_region` (`SATNUM`) is unavailable; see
        `initialize_equilibrium_arrays`. Also if exactly one of
        `pressure`/`solution_gor` (or `pressure`/`vaporized_oil_to_gas_ratio`) is
        explicit while the other is equilibration-derived, since `EQUIL`
        derives `solution_gor`/`vaporized_oil_to_gas_ratio` from its own
        internally-integrated hydrostatic pressure profile, not from an
        explicit `pressure` override as the two may end up off the oil
        bubble-point / gas dew-point curve together.
    """
    if equilibrium is None and deck_file is not None and deck_file.has("RESTART"):
        raise NotImplementedError(
            "Restart-file state initialization is not yet supported. "
            "Supply `equilibrium` and/or explicit arrays instead."
        )

    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    unit_system = reservoir.unit_system
    if pvt.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `pvt.unit_system` ({pvt.unit_system.value!r})."
        )
    if equilibrium is not None and equilibrium.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `equilibrium.unit_system` ({equilibrium.unit_system.value!r})."
        )
    if (
        satfunc is not None
        and satfunc.unit_system is not None
        and satfunc.unit_system != unit_system
    ):
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `satfunc.unit_system` ({satfunc.unit_system.value!r})."
        )

    temperature_array = resolve_temperature(
        reservoir=reservoir,
        deck_file=deck_file,
        temperature=temperature,
        dtype=dtype,
    )
    explicit: dict[str, CellArray | None] = {
        "pressure": pressure,
        "water_saturation": water_saturation,
        "gas_saturation": gas_saturation,
        "solution_gor": solution_gor,
        "vaporized_oil_to_gas_ratio": vaporized_oil_to_gas_ratio,
    }
    field_keywords = {
        "pressure": "PRESSURE",
        "water_saturation": "SWAT",
        "gas_saturation": "SGAS",
        "solution_gor": "RS",
        "vaporized_oil_to_gas_ratio": "RV",
    }
    if deck_file is not None:
        for field, keyword in field_keywords.items():
            if explicit[field] is None:
                deck_array = deck_file.get(keyword)
                if deck_array is not None:
                    explicit[field] = typing.cast(CellArray, deck_array)

    equilibrium_arrays: EquilibriumArrays | None = None
    if any(value is None for value in explicit.values()):
        if equilibrium is None:
            if deck_file is not None and deck_file.get("EQUIL"):
                equilibrium = Equilibrium.from_deck(deck_file)
            else:
                missing = [field for field, value in explicit.items() if value is None]
                raise ValidationError(
                    f"No explicit array/keyword and no EQUIL data available "
                    f"for: {missing}. Supply `equilibrium`, a "
                    "`deck_file` with an `EQUIL` keyword, or explicit arrays "
                    "for these fields."
                )
        equilibrium_arrays = initialize_equilibrium_arrays(
            reservoir=reservoir,
            pvt=pvt,
            equilibrium=equilibrium,
            temperature=temperature_array,
            satfunc=satfunc,
            depth_step=depth_step,
            dtype=dtype,
            saturation_samples=saturation_samples,
        )

        # `pressure` and `solution_gor`/`vaporized_oil_to_gas_ratio` are physically
        # coupled through the oil bubble-point / gas dew-point curve. EQUIL
        # derives Rs/Rv from its own internally-integrated hydrostatic
        # pressure profile, not from an explicit `pressure` override, so
        # taking exactly one of the pair from EQUIL and the other from an
        # explicit array/keyword can silently produce a state that isn't on
        # that curve (e.g. an Rs inconsistent with the stored pressure).
        # Supplying both explicitly, or neither, keeps them consistent by
        # construction; this only warns (rather than raising) since a
        # deliberately mismatched pair - e.g. a measured `pressure` history-
        # matched against an EQUIL-only Rs/Rv profile, is a legitimate,
        # if unusual, use of the per-field override contract above.
        pressure_is_explicit = explicit["pressure"] is not None
        for gas_ratio_field in ("solution_gor", "vaporized_oil_to_gas_ratio"):
            if pressure_is_explicit != (explicit[gas_ratio_field] is not None):
                warnings.warn(
                    f"`pressure` is explicit but `{gas_ratio_field}` is "
                    "EQUIL-derived (or vice versa); the two may be "
                    "physically inconsistent since EQUIL derives "
                    f"`{gas_ratio_field}` from its own internally-integrated "
                    "hydrostatic pressure profile, not from the explicit "
                    "`pressure` array. Supply both explicitly, or neither, "
                    "to guarantee a state consistent with the oil "
                    "bubble-point / gas dew-point curve.",
                    UserWarning,
                    stacklevel=2,
                )

    def resolve(field: str, equilibrium_value: CellArray | None) -> CellArray:
        value = explicit[field]
        if value is not None:
            array = np.asarray(value, dtype=dtype)
            if array.shape != (n_cells,):
                raise ValidationError(
                    f"Explicit `{field}` has shape {array.shape}; expected ({n_cells},)."
                )
            return typing.cast(CellArray, array)
        assert equilibrium_value is not None  # guaranteed by the branch above
        return equilibrium_value

    pressure_array = resolve(
        "pressure",
        equilibrium_arrays.pressure if equilibrium_arrays is not None else None,
    )
    water_saturation_array = resolve(
        "water_saturation",
        equilibrium_arrays.water_saturation if equilibrium_arrays is not None else None,
    )
    gas_saturation_array = resolve(
        "gas_saturation",
        equilibrium_arrays.gas_saturation if equilibrium_arrays is not None else None,
    )
    solution_gor_array = resolve(
        "solution_gor",
        equilibrium_arrays.solution_gor if equilibrium_arrays is not None else None,
    )
    vaporized_oil_ratio_array = resolve(
        "vaporized_oil_to_gas_ratio",
        equilibrium_arrays.vaporized_oil_to_gas_ratio if equilibrium_arrays is not None else None,
    )
    if equilibrium_arrays is not None:
        oil_bubble_point_pressure_array = equilibrium_arrays.oil_bubble_point_pressure
        gas_dew_point_pressure_array = equilibrium_arrays.gas_dew_point_pressure
    else:
        # Filled in per-PVTNUM below from each region's own PVT tables.
        # `oil_bubble_point_pressure_array` falls back to `pressure_array` (assume
        # saturated oil) and `gas_dew_point_pressure_array` falls back to zero
        # (assume no free gas) only where a region's table has no
        # bubble-/dew-point data of its own.
        oil_bubble_point_pressure_array = pressure_array.copy()
        gas_dew_point_pressure_array = np.zeros(n_cells, dtype=dtype)

    saturation_sum_excess = (water_saturation_array + gas_saturation_array) - 1.0
    if np.any(saturation_sum_excess > 1e-6):
        raise ValidationError(
            "water_saturation + gas_saturation exceeds 1.0 in at least one "
            f"cell (max excess {float(saturation_sum_excess.max()):.3e})."
        )
    oil_saturation_array = np.clip(
        1.0 - water_saturation_array - gas_saturation_array, 0.0, 1.0, dtype=dtype
    )

    pore_volumes = reservoir.pore_volumes
    pvt_region_index: IntCellArray = (
        reservoir.regions.pvt_region
        if reservoir.regions is not None and reservoir.regions.pvt_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )

    oil_mass = np.zeros(n_cells, dtype=dtype)
    water_mass = np.zeros(n_cells, dtype=dtype)
    free_gas_mass = np.zeros(n_cells, dtype=dtype)
    dissolved_gas_mass_in_oil = np.zeros(n_cells, dtype=dtype)
    vaporized_oil_mass_in_gas = np.zeros(n_cells, dtype=dtype)

    # `UnitSystem.FIELD` mixes two volume "families": oil/water are barrels (STB), gas is
    # cubic feet (SCF), and 1 barrel = 5.614583 ft3 (`c.BARRELS_TO_CUBIC_FEET`).
    # `solution_gor` (Rs) and `vaporized_oil_to_gas_ratio` (Rv) are both normalized
    # to SCF/STB by this codebase (see `equilibrium.py` / `pvt/regions.py`),
    # so each crosses those two families the same way: converting
    # `Rs * rho_g_sc` into an oil-mass-basis term (or `Rv * rho_o_sc` into a
    # gas-mass-basis term) needs an explicit ft3<->bbl correction; skipping
    # it overstates the mass by ~5.615x. `METRIC/SI/LAB` use a single volume unit
    # throughout (m3/m3, cc/cc) so no such correction applies for them.
    volume_correction = c.CUBIC_FEET_TO_STB if unit_system is UnitSystem.FIELD else 1.0

    for pvtnum in np.unique(pvt_region_index):
        mask = pvt_region_index == pvtnum
        pvt_region = pvt.region(int(pvtnum))
        static = pvt_region.static
        if static.stock_tank_oil_density is None:
            raise ValidationError(
                f"PVTNUM {pvtnum}: `stock_tank_oil_density` (DENSITY "
                "keyword) is required to assemble ReservoirState masses."
            )

        rho_o_sc = static.stock_tank_oil_density
        rho_g_sc = static.stock_tank_gas_density
        rho_w_sc = static.stock_tank_water_density

        p = pressure_array[mask]
        t = temperature_array[mask]
        so = oil_saturation_array[mask]
        sw = water_saturation_array[mask]
        sg = gas_saturation_array[mask]
        rs = solution_gor_array[mask]
        pv = pore_volumes[mask]

        bo = pvt_region.tables.oil.formation_volume_factor(  # type: ignore[union-attr]
            pressure=p, temperature=t, solution_gor=rs
        )
        if bo is None:
            raise ValidationError(
                f"PVTNUM {pvtnum}: oil formation volume factor table is unavailable."
            )
        oil_mass[mask] = so * pv / bo * rho_o_sc

        if equilibrium_arrays is None:
            oil_bubble_point = pvt_region.tables.oil.bubble_point_pressure(  # type: ignore[union-attr]
                temperature=t, solution_gor=rs
            )
            if oil_bubble_point is not None:
                oil_bubble_point_pressure_array[mask] = oil_bubble_point

            if pvt_region.tables.gas is not None:
                gas_dew_point = pvt_region.tables.gas.dew_point_pressure(temperature=t)
                if gas_dew_point is not None:
                    gas_dew_point_pressure_array[mask] = gas_dew_point

        if np.any(sg > 0.0):
            if pvt_region.tables.gas is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: free gas saturation is present "
                    "but no gas PVT table is available."
                )

            bg = pvt_region.tables.gas.formation_volume_factor(pressure=p, temperature=t)
            if bg is None or rho_g_sc is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: gas FVF table or "
                    "`stock_tank_gas_density` is unavailable but Sg > 0."
                )
            free_gas_mass[mask] = sg * pv / bg * rho_g_sc

            rv = vaporized_oil_ratio_array[mask]
            vaporized_oil_mass_in_gas[mask] = (
                rv * free_gas_mass[mask] * (rho_o_sc / rho_g_sc) * volume_correction
            )

        if np.any(sw > 0.0):
            if pvt_region.tables.water is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: water saturation is present "
                    "but no water PVT table is available."
                )

            bw = pvt_region.tables.water.formation_volume_factor(pressure=p, temperature=t)
            if bw is None or rho_w_sc is None:
                raise ValidationError(
                    f"PVTNUM {pvtnum}: water FVF table or "
                    "`stock_tank_water_density` is unavailable."
                )
            water_mass[mask] = sw * pv / bw * rho_w_sc

        if rho_g_sc is not None:
            dissolved_gas_mass_in_oil[mask] = (
                rs * oil_mass[mask] * (rho_g_sc / rho_o_sc) * volume_correction
            )

    zeros = np.zeros(n_cells, dtype=dtype)
    hysteresis = (
        Hysteresis.from_initial_saturation(water_saturation_array, gas_saturation_array)
        if with_hysteresis
        else None
    )
    return ReservoirState(
        pressure=pressure_array,
        temperature=temperature_array,
        oil_saturation=oil_saturation_array,  # type: ignore[arg-type]
        water_saturation=water_saturation_array,
        gas_saturation=gas_saturation_array,
        solution_gor=solution_gor_array,
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure_array),
        vaporized_oil_to_gas_ratio=vaporized_oil_ratio_array,
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure_array),
        gas_solubility_in_water=zeros,
        water_bubble_point_pressure=zeros,
        oil_mass=typing.cast(CellArray, oil_mass),
        water_mass=typing.cast(CellArray, water_mass),
        free_gas_mass=typing.cast(CellArray, free_gas_mass),
        dissolved_gas_mass_in_oil=typing.cast(CellArray, dissolved_gas_mass_in_oil),
        dissolved_gas_mass_in_water=zeros,
        vaporized_oil_mass_in_gas=typing.cast(CellArray, vaporized_oil_mass_in_gas),
        hysteresis=hysteresis,
        unit_system=unit_system,
    )
