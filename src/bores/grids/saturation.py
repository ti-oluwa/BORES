import typing
import warnings

import numpy as np

from bores._precision import get_dtype
from bores.constants import c
from bores.errors import ValidationError
from bores.relperm import RelativePermeabilityTable
from bores.types import FluidPhase, NDimension, NDimensionalGrid

__all__ = ["build_saturation_grids", "seed_phase_saturation"]


def build_saturation_grids(
    depth_grid: NDimensionalGrid[NDimension],
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    use_transition_zones: bool = False,
    gas_oil_transition_thickness: float = 5.0,
    oil_water_transition_thickness: float = 5.0,
    transition_curvature_exponent: float = 2.0,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Build three-phase saturation grids from fluid contact depths and reservoir properties.

    Creates physically realistic water, oil, and gas saturation distributions based on:
    - Fluid contact depths (gas-oil contact, oil-water contact)
    - Residual saturations appropriate to each zone
    - Optional capillary transition zones for smooth saturation gradients

    This function properly accounts for the fact that residual oil saturation depends
    on the displacing fluid:
    - **Gas cap zone**: Uses Sor_gas (typically 0.10-0.25) - gas displaces oil efficiently
    - **Water zone**: Uses Sor_water (typically 0.20-0.35) - water leaves more residual oil
    - **Oil zone**: Contains mobile oil with connate water and residual gas

    ### Reservoir Zonation

    ### Sharp Contacts (use_transition_zones=False):
    ```markdown
        ┌─────────────────────────────────────┐
        │   GAS CAP (depth < GOC)            │  Sg = 1 - Sor_gas - Swc
        │   Sg + So + Sw = 1.0               │  So = Sor_gas
        │   Gas has displaced oil            │  Sw = Swc
        ├─────────────────────────────────────┤ ← Gas-Oil Contact (GOC)
        │   OIL ZONE (GOC ≤ depth < OWC)    │  So = 1 - Swc - Sgr
        │   So + Sw + Sg = 1.0               │  Sw = Swc
        │   Original accumulation            │  Sg = Sgr
        ├─────────────────────────────────────┤ ← Oil-Water Contact (OWC)
        │   WATER ZONE (depth ≥ OWC)        │  Sw = 1 - Sor_water
        │   Sw + So + Sg = 1.0               │  So = Sor_water
        │   Water has displaced oil          │  Sg = 0
        └─────────────────────────────────────┘
    ```

    ### Transition Zones (use_transition_zones=True):
    ```markdown
        ┌─────────────────────────────────────┐
        │   GAS CAP                          │  Pure gas zone
        │   (depth < GOC - h_go/2)           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │   GAS-OIL TRANSITION               │  Smooth blend:
        │   (GOC - h_go/2 to GOC + h_go/2)   │  Sg: high → low
        │                                     │  So: low → high
        ├─────────────────────────────────────┤ ← Gas-Oil Contact (GOC)
        │   OIL ZONE                         │  Pure oil zone
        │   (GOC + h_go/2 to OWC - h_ow/2)   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │   OIL-WATER TRANSITION             │  Smooth blend:
        │   (OWC - h_ow/2 to OWC + h_ow/2)   │  Sw: low → high
        │                                     │  So: high → low
        ├─────────────────────────────────────┤ ← Oil-Water Contact (OWC)
        │   WATER ZONE                       │  Pure water zone
        │   (depth > OWC + h_ow/2)           │
        └─────────────────────────────────────┘
    ```

    ### Transition Zone Physics

    The optional transition zones mimic capillary pressure effects without explicit
    capillary pressure calculations. Saturation blending uses a power-law profile:

    ```LaTex
    $$w(z) = \left(\frac{z - z_{top}}{h_{transition}}\right)^{n}$$
    ```

    Where:
    - $w$ = blending weight (0 at top, 1 at bottom)
    - $z$ = depth within transition zone
    - $h_{transition}$ = transition zone thickness
    - $n$ = curvature exponent (controls profile shape)

    **Curvature exponent effects:**
    - n < 1: Sharper transition (approaching sharp contact)
    - n = 1: Linear interpolation
    - n = 2: Smooth, S-curved transition (recommended, realistic)
    - n > 3: Very gradual transition (may be unrealistic)

    ### Typical Parameter Values

    **Water-wet sandstone reservoir:**
    - Swc = 0.20-0.30 (connate water saturation)
    - Sor_water = 0.25-0.35 (residual oil to water displacement)
    - Sor_gas = 0.10-0.20 (residual oil to gas displacement)
    - Sgr = 0.03-0.08 (residual gas saturation)
    - GOC-OWC separation = 50-500 ft (typical oil column)
    - Transition thickness = 5-20 ft (if used)

    **Oil-wet carbonate reservoir:**
    - Swc = 0.15-0.25 (lower connate water)
    - Sor_water = 0.35-0.45 (higher residual oil)
    - Sor_gas = 0.15-0.30 (higher residual oil to gas)
    - Sgr = 0.05-0.10 (residual gas)

    :param depth_grid: Grid of cell-center depths (ft). Depth increases downward,
        where k=0 is the shallowest (top) layer and k=-1 is the deepest (bottom) layer.
        Shape: (nx, ny, nz).
    :param gas_oil_contact: Gas-oil contact depth (ft). Cells at depths less than this
        value are in the gas cap. Must be less than oil_water_contact.
    :param oil_water_contact: Oil-water contact depth (ft). Cells at depths greater than
        or equal to this value are in the water zone. Must be greater than gas_oil_contact.
    :param connate_water_saturation_grid: Connate (irreducible) water saturation for each
        cell (fraction, 0-1). This is the minimum water saturation that remains in the
        pore space even in the presence of hydrocarbons. Typical: 0.15-0.30.
        Shape: (nx, ny, nz).
    :param residual_oil_saturation_water_grid: Residual oil saturation when oil is
        displaced by water (fraction, 0-1). This is the oil that remains trapped after
        water imbibition. Higher in oil-wet systems. Typical: 0.20-0.35 (water-wet),
        0.35-0.45 (oil-wet). Shape: (nx, ny, nz).
    :param residual_oil_saturation_gas_grid: Residual oil saturation when oil is
        displaced by gas (fraction, 0-1). This is the oil that remains trapped after
        gas drainage. Typically lower than Sor_water due to favorable mobility ratio
        and wettability. Typical: 0.10-0.25. Shape: (nx, ny, nz).
    :param residual_gas_saturation_grid: Residual (trapped) gas saturation in the oil
        zone (fraction, 0-1). This is solution gas that has come out of solution but
        cannot flow. Typical: 0.03-0.08. Shape: (nx, ny, nz).
    :param porosity_grid: Porosity for each cell (fraction, 0-1). Used to identify
        active cells: cells with porosity <= 0 or NaN are considered inactive and
        set to zero saturation for all phases. Shape: (nx, ny, nz).
    :param use_transition_zones: If True, create smooth capillary-like transitions at
        fluid contacts. If False, use sharp (discontinuous) contacts. Default: False.
    :param gas_oil_transition_thickness: Thickness of the gas-oil transition zone (ft).
        The transition is centered on the GOC and extends ±thickness/2 around it.
        Only used if use_transition_zones=True. Typical: 5-20 ft. Default: 5.0.
    :param oil_water_transition_thickness: Thickness of the oil-water transition zone (ft).
        The transition is centered on the OWC and extends ±thickness/2 around it.
        Only used if use_transition_zones=True. Typical: 5-20 ft. Default: 5.0.
    :param transition_curvature_exponent: Power-law exponent controlling the shape of
        saturation gradients in transition zones (dimensionless). Higher values produce
        smoother, more realistic S-curves. Typical: 1.5-3.0. Default: 2.0.
        Only used if use_transition_zones=True.

    :return: Tuple of (water_saturation, oil_saturation, gas_saturation) grids, each
        with shape (nx, ny, nz). All three saturations sum to exactly 1.0 in active cells
        (where porosity > 0) and are 0.0 in inactive cells.

    :raises ValidationError: If inputs are invalid:
        - GOC >= OWC (contacts in wrong order)
        - Array shapes don't match
        - Saturation values outside [0, 1]
        - Physically impossible combinations (e.g., Swc + Sor > 1.0)
        - Transition zones overlap
        - Transition parameters are non-positive

    :warns UserWarning: If potentially problematic but not invalid:
        - Sor_gas > Sor_water (unusual wettability)
        - Oil column very thin (< 1 ft between transitions)

    Example:
    ```python
    import numpy as np
    from bores import build_depth_grid, build_saturation_grids

    # Create a 50x50x30 reservoir grid (30 layers, 10 ft each)
    nx, ny, nz = 50, 50, 30
    thickness = np.full((nx, ny, nz), 10.0)
    depth = build_depth_grid(thickness, datum=1000.0)  # Top at 1000 ft depth

    # Define reservoir properties (water-wet sandstone)
    porosity = np.full((nx, ny, nz), 0.22)  # 22% porosity
    swc = np.full((nx, ny, nz), 0.25)       # 25% connate water
    sor_w = np.full((nx, ny, nz), 0.30)     # 30% residual oil to water
    sor_g = np.full((nx, ny, nz), 0.15)     # 15% residual oil to gas
    sgr = np.full((nx, ny, nz), 0.05)       # 5% residual gas

    # Fluid contacts: GOC at 1100 ft, OWC at 1200 ft (100 ft oil column)
    goc = 1100.0
    owc = 1200.0

    # Sharp contacts (discontinuous)
    sw, so, sg = build_saturation_grids(
        depth_grid=depth,
        gas_oil_contact=goc,
        oil_water_contact=owc,
        connate_water_saturation_grid=swc,
        residual_oil_saturation_water_grid=sor_w,
        residual_oil_saturation_gas_grid=sor_g,
        residual_gas_saturation_grid=sgr,
        porosity_grid=porosity,
        use_transition_zones=False
    )

    # Verify saturation constraint
    assert np.allclose(sw + so + sg, 1.0, where=porosity > 0)

    # With capillary transition zones (smooth gradients)
    sw, so, sg = build_saturation_grids(
        depth_grid=depth,
        gas_oil_contact=goc,
        oil_water_contact=owc,
        connate_water_saturation_grid=swc,
        residual_oil_saturation_water_grid=sor_w,
        residual_oil_saturation_gas_grid=sor_g,
        residual_gas_saturation_grid=sgr,
        porosity_grid=porosity,
        use_transition_zones=True,
        gas_oil_transition_thickness=10.0,
        oil_water_transition_thickness=15.0,
        transition_curvature_exponent=2.0
    )
    ```

    Notes:
        - Depth increases downward: smaller depth = shallower (top), larger depth = deeper (bottom)
        - Gas cap: depth < GOC (shallowest zone, uses Sor_gas)
        - Oil zone: GOC ≤ depth < OWC (middle zone, uses Sgr)
        - Water zone: depth ≥ OWC (deepest zone, uses Sor_water)
        - Inactive cells (porosity ≤ 0 or NaN) have zero saturation for all phases
        - All saturations are normalized to sum to exactly 1.0 in active cells
        - For heterogeneous reservoirs, provide spatially varying property grids
        - Transition zones should not overlap: ensure (OWC - OWC) > (h_go + h_ow)/2

    See Also:
        - `build_depth_grid()`: Create depth grid from layer thicknesses
        - `build_elevation_grid()`: Create elevation grid (upward-positive)
        - Relative permeability models: Define how saturations affect flow

    References:
        - Craig, F. F. (1971). "The Reservoir Engineering Aspects of Waterflooding."
            Society of Petroleum Engineers. (Residual saturations)
        - Lake, L. W. (1989). "Enhanced Oil Recovery." Prentice Hall.
            (Wettability effects on Sor)
        - Leverett, M. C. (1941). "Capillary Behavior in Porous Solids."
            Transactions of the AIME, 142(01), 152-169. (Transition zones)
    """
    # Identify active cells based on porosity
    # If porosity is NaN or <= 0, cell is inactive
    active_mask = np.isfinite(porosity_grid) & (porosity_grid > 0)
    _validate_inputs(
        depth_grid=depth_grid,
        gas_oil_contact=gas_oil_contact,
        oil_water_contact=oil_water_contact,
        connate_water_saturation=connate_water_saturation_grid,
        residual_oil_saturation_water=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
        residual_gas_saturation=residual_gas_saturation_grid,
        porosity=porosity_grid,
        use_transitions=use_transition_zones,
        gas_oil_transition_thickness=gas_oil_transition_thickness,
        oil_water_transition_thickness=oil_water_transition_thickness,
        transition_curvature_exponent=transition_curvature_exponent,
        active=active_mask,
    )

    dtype = get_dtype()
    water_saturation = np.zeros_like(depth_grid, dtype=dtype)
    oil_saturation = np.zeros_like(depth_grid, dtype=dtype)
    gas_saturation = np.zeros_like(depth_grid, dtype=dtype)
    water_saturation = typing.cast(NDimensionalGrid[NDimension], water_saturation)
    oil_saturation = typing.cast(NDimensionalGrid[NDimension], oil_saturation)
    gas_saturation = typing.cast(NDimensionalGrid[NDimension], gas_saturation)

    if not use_transition_zones:
        water_saturation, oil_saturation, gas_saturation = _build_sharp_contacts(
            depth_grid=depth_grid,
            gas_oil_contact=gas_oil_contact,
            oil_water_contact=oil_water_contact,
            connate_water_saturation=connate_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            active=active_mask,
        )
    else:
        water_saturation, oil_saturation, gas_saturation = _build_transition_zones(
            depth_grid=depth_grid,
            gas_oil_contact=gas_oil_contact,
            oil_water_contact=oil_water_contact,
            connate_water_saturation=connate_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            gas_oil_transition_thickness=gas_oil_transition_thickness,
            oil_water_transition_thickness=oil_water_transition_thickness,
            transition_curvature_exponent=transition_curvature_exponent,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            active=active_mask,
        )

    # Normalize saturations to sum to 1.0 in active cells
    water_saturation, oil_saturation, gas_saturation = _normalize_saturations(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        active=active_mask,
    )
    return water_saturation, oil_saturation, gas_saturation  # type: ignore[return-value]


def _validate_inputs(
    depth_grid: np.typing.NDArray,
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation: np.typing.NDArray,
    residual_oil_saturation_water: np.typing.NDArray,
    residual_oil_saturation_gas: np.typing.NDArray,
    residual_gas_saturation: np.typing.NDArray,
    porosity: np.typing.NDArray,
    use_transitions: bool,
    gas_oil_transition_thickness: float,
    oil_water_transition_thickness: float,
    transition_curvature_exponent: float,
    active: np.typing.NDArray,
) -> None:
    """
    Validate all input parameters for saturation grid building.

    Performs comprehensive validation including:
    - Contact depth ordering (GOC must be above OWC)
    - Array shape consistency across all grids
    - Saturation value ranges (must be between 0 and 1)
    - Physical constraints (e.g., Swc + Sor_w <= 1.0, Swc + Sor_g <= 1.0)
    - Transition zone parameters (positive thicknesses, non-overlapping zones)
    - Oil column thickness warnings
    - Sor_w >= Sor_g check (water displacement typically leaves more residual oil)

    :param depth_grid: Depth grid array (ft).
    :param gas_oil_contact: Gas-oil contact depth (ft).
    :param oil_water_contact: Oil-water contact depth (ft).
    :param connate_water_saturation: Connate water saturation grid (fraction).
    :param residual_oil_saturation_water: Residual oil saturation to water displacement (fraction).
    :param residual_oil_saturation_gas: Residual oil saturation to gas displacement (fraction).
    :param residual_gas_saturation: Residual gas saturation grid (fraction).
    :param porosity: Porosity grid (fraction).
    :param use_transitions: Whether transition zones are enabled.
    :param gas_oil_transition_thickness: Gas-oil transition zone thickness (ft).
    :param oil_water_transition_thickness: Oil-water transition zone thickness (ft).
    :param transition_curvature_exponent: Transition curvature exponent (dimensionless).
    :param active: Boolean mask indicating active cells.
    :raises ValidationError: If any validation check fails.
    :raises UserWarning: If oil column is very thin (< 1.0 ft) or Sor_g > Sor_w.
    """

    # Check contact depth ordering
    if gas_oil_contact >= oil_water_contact:
        raise ValidationError(
            f"Gas-oil contact ({gas_oil_contact}) must be above oil-water contact ({oil_water_contact}). "
            "Depths increase downward."
        )

    # Check array shapes match
    if not (
        depth_grid.shape
        == connate_water_saturation.shape
        == residual_oil_saturation_water.shape
        == residual_oil_saturation_gas.shape
        == residual_gas_saturation.shape
        == porosity.shape
    ):
        raise ValidationError("All grid arrays must have the same shape.")

    # Check saturation ranges
    if np.any(
        (connate_water_saturation[active] < 0) | (connate_water_saturation[active] > 1)
    ):
        raise ValidationError("Connate water saturation must be in [0, 1].")
    if np.any(
        (residual_oil_saturation_water[active] < 0)
        | (residual_oil_saturation_water[active] > 1)
    ):
        raise ValidationError("Residual oil saturation (water) must be in [0, 1].")
    if np.any(
        (residual_oil_saturation_gas[active] < 0)
        | (residual_oil_saturation_gas[active] > 1)
    ):
        raise ValidationError("Residual oil saturation (gas) must be in [0, 1].")
    if np.any(
        (residual_gas_saturation[active] < 0) | (residual_gas_saturation[active] > 1)
    ):
        raise ValidationError("Residual gas saturation must be in [0, 1].")

    # Check for physically impossible saturation combinations
    if np.any(
        (connate_water_saturation[active] + residual_oil_saturation_gas[active]) > 1.0
    ):
        raise ValidationError(
            "Swc + Sor_gas exceeds 1.0 in some cells (gas zone constraint)."
        )
    if np.any(
        (connate_water_saturation[active] + residual_gas_saturation[active]) > 1.0
    ):
        raise ValidationError(
            "Swc + Sgr exceeds 1.0 in some cells (oil zone constraint)."
        )
    if np.any(
        (residual_oil_saturation_water[active] + residual_gas_saturation[active]) > 1.0
    ):
        raise ValidationError(
            "Sor_water + Sgr exceeds 1.0 in some cells (not physical in any zone)."
        )

    # Warn if Sor_gas > Sor_water (unusual but not impossible)
    if np.any(
        residual_oil_saturation_gas[active] > residual_oil_saturation_water[active]
    ):
        warnings.warn(
            "Sor_gas > Sor_water in some cells. Typically, water displacement leaves "
            "more residual oil than gas displacement due to wettability effects.",
            UserWarning,
        )

    oil_column_thickness = oil_water_contact - gas_oil_contact
    if oil_column_thickness < c.MIN_OIL_ZONE_THICKNESS:
        warnings.warn(
            f"Oil column is very thin ({oil_column_thickness:.2f} ft). "
            "Verify contact depths are correct.",
            UserWarning,
        )

    # Check transition zone parameters
    if use_transitions:
        if gas_oil_transition_thickness <= 0 or oil_water_transition_thickness <= 0:
            raise ValidationError("Transition thicknesses must be positive.")

        if transition_curvature_exponent <= 0:
            raise ValidationError("Transition curvature exponent must be positive.")

        # Check for overlapping transitions
        gas_oil_contact_bottom = gas_oil_contact + gas_oil_transition_thickness / 2
        oil_water_contact_top = oil_water_contact - oil_water_transition_thickness / 2

        if gas_oil_contact_bottom >= oil_water_contact_top:
            raise ValidationError(
                f"Transition zones overlap: GOC transition ends at {gas_oil_contact_bottom:.2f}, "
                f"but OWC transition starts at {oil_water_contact_top:.2f}. "
                "Reduce transition thicknesses or increase contact separation."
            )

        # Warn if oil column is thin
        oil_column_thickness = oil_water_contact_top - gas_oil_contact_bottom
        if oil_column_thickness < 1.0:
            warnings.warn(
                f"Oil column between transitions is very thin ({oil_column_thickness:.2f} units(ft)). "
                "Consider reducing transition thicknesses.",
                UserWarning,
            )


def _build_sharp_contacts(
    depth_grid: NDimensionalGrid[NDimension],
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation: NDimensionalGrid[NDimension],
    residual_oil_saturation_water: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas: NDimensionalGrid[NDimension],
    residual_gas_saturation: NDimensionalGrid[NDimension],
    water_saturation: NDimensionalGrid[NDimension],
    oil_saturation: NDimensionalGrid[NDimension],
    gas_saturation: NDimensionalGrid[NDimension],
    active: np.typing.NDArray,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Build saturations with sharp (discontinuous) fluid contacts.

    Creates three distinct zones with abrupt saturation changes at the contact depths.
    Uses different residual oil saturations depending on the displacing fluid.

    **Gas Zone (depth < GOC):**
        Gas has displaced oil, leaving residual oil trapped by gas:
        - Gas saturation: Sg = 1.0 - Sor_gas - Swc (mobile gas)
        - Oil saturation: So = Sor_gas (residual oil to gas displacement)
        - Water saturation: Sw = Swc (connate water)

    **Oil Zone (GOC <= depth < OWC):**
        Original oil accumulation with connate water:
        - Oil saturation: So = 1.0 - Swc - Sgr (mobile oil)
        - Water saturation: Sw = Swc (connate water)
        - Gas saturation: Sg = Sgr (residual gas, if any)

    **Water Zone (depth >= OWC):**
        Water has displaced oil, leaving residual oil trapped by water:
        - Water saturation: Sw = 1.0 - Sor_water (mobile water)
        - Oil saturation: So = Sor_water (residual oil to water displacement)
        - Gas saturation: Sg = 0.0 (no gas)

    :param depth_grid: Depth grid for each cell (ft).
    :param gas_oil_contact: Gas-oil contact depth (ft).
    :param oil_water_contact: Oil-water contact depth (ft).
    :param connate_water_saturation: Connate water saturation grid (fraction).
    :param residual_oil_saturation_water: Residual oil saturation to water (fraction).
    :param residual_oil_saturation_gas: Residual oil saturation to gas (fraction).
    :param residual_gas_saturation: Residual gas saturation grid (fraction).
    :param water_saturation: Water saturation grid to be populated (fraction).
    :param oil_saturation: Oil saturation grid to be populated (fraction).
    :param gas_saturation: Gas saturation grid to be populated (fraction).
    :param active: Boolean mask indicating active cells.
    :return: Tuple of (water_saturation, oil_saturation, gas_saturation) grids.
    """
    gas_zone = (depth_grid < gas_oil_contact) & active
    oil_zone = (
        (depth_grid >= gas_oil_contact) & (depth_grid < oil_water_contact) & active
    )
    water_zone = (depth_grid >= oil_water_contact) & active

    # Gas cap - uses Sor_gas (gas has displaced oil)
    gas_saturation[gas_zone] = (
        1.0 - residual_oil_saturation_gas[gas_zone] - connate_water_saturation[gas_zone]
    )
    oil_saturation[gas_zone] = residual_oil_saturation_gas[gas_zone]
    water_saturation[gas_zone] = connate_water_saturation[gas_zone]

    # Oil zone - original oil accumulation
    oil_saturation[oil_zone] = (
        1.0 - connate_water_saturation[oil_zone] - residual_gas_saturation[oil_zone]
    )
    water_saturation[oil_zone] = connate_water_saturation[oil_zone]
    gas_saturation[oil_zone] = residual_gas_saturation[oil_zone]

    # Water zone - uses Sor_water (water has displaced oil)
    water_saturation[water_zone] = 1.0 - residual_oil_saturation_water[water_zone]
    oil_saturation[water_zone] = residual_oil_saturation_water[water_zone]
    gas_saturation[water_zone] = 0.0
    return water_saturation, oil_saturation, gas_saturation


def _build_transition_zones(
    depth_grid: NDimensionalGrid[NDimension],
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation: NDimensionalGrid[NDimension],
    residual_oil_saturation_water: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas: NDimensionalGrid[NDimension],
    residual_gas_saturation: NDimensionalGrid[NDimension],
    gas_oil_transition_thickness: float,
    oil_water_transition_thickness: float,
    transition_curvature_exponent: float,
    water_saturation: NDimensionalGrid[NDimension],
    oil_saturation: NDimensionalGrid[NDimension],
    gas_saturation: NDimensionalGrid[NDimension],
    active: np.typing.NDArray,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Build saturations with smooth capillary transition zones at fluid contacts.

    Creates five distinct zones with gradual saturation changes in transition regions.
    Uses appropriate residual oil saturations based on the displacing fluid:
    - Gas cap uses Sor_gas
    - Water zone uses Sor_water
    - Transition zones blend between appropriate end-members

    **Gas Cap Zone (depth < GOC - gas_oil_transition_thickness/2):**
        Pure gas zone where gas has displaced oil:
        - Sg = 1.0 - Sor_gas - Swc, So = Sor_gas, Sw = Swc

    **Gas-Oil Transition Zone (GOC - gas_oil_transition_thickness/2 <= depth <= GOC + gas_oil_transition_thickness/2):**
        Smooth transition from gas-dominated to oil-dominated:
        - Uses power-law weighting: w = ((depth - goc_top) / gas_oil_transition_thickness)^exponent
        - Gas saturation decreases from (1 - Sor_gas - Swc) to Sgr
        - Oil saturation increases from Sor_gas to (1 - Swc - Sgr)
        - Water saturation remains at Swc

    **Oil Zone (GOC + gas_oil_transition_thickness/2 < depth < OWC - oil_water_transition_thickness/2):**
        Pure oil zone with original accumulation:
        - So = 1.0 - Swc - Sgr, Sw = Swc, Sg = Sgr

    **Oil-Water Transition Zone (OWC - oil_water_transition_thickness/2 <= depth <= OWC + oil_water_transition_thickness/2):**
        Smooth transition from oil-dominated to water-dominated:
        - Uses power-law weighting: w = ((depth - owc_top) / oil_water_transition_thickness)^exponent
        - Water saturation increases from Swc to (1 - Sor_water)
        - Oil saturation decreases from (1 - Swc - Sgr) to Sor_water
        - Gas saturation decreases from Sgr to 0

    **Water Zone (depth > OWC + oil_water_transition_thickness/2):**
        Pure water zone where water has displaced oil:
        - Sw = 1.0 - Sor_water, So = Sor_water, Sg = 0.0

    The transition curvature exponent controls the shape of the saturation profile:
    - exponent < 1: More abrupt change (approaching sharp contact)
    - exponent = 1: Linear interpolation
    - exponent > 1: Smoother, S-curved transition (more realistic capillary pressure effect)

    :param depth_grid: Depth grid for each cell (ft).
    :param gas_oil_contact: Gas-oil contact depth (ft).
    :param oil_water_contact: Oil-water contact depth (ft).
    :param connate_water_saturation: Connate water saturation grid (fraction).
    :param residual_oil_saturation_water: Residual oil to water displacement (fraction).
    :param residual_oil_saturation_gas: Residual oil to gas displacement (fraction).
    :param residual_gas_saturation: Residual gas saturation grid (fraction).
    :param gas_oil_transition_thickness: Gas-oil transition zone thickness (ft).
    :param oil_water_transition_thickness: Oil-water transition zone thickness (ft).
    :param transition_curvature_exponent: Power-law exponent controlling transition curvature (dimensionless).
    :param water_saturation: Water saturation grid to be populated (fraction).
    :param oil_saturation: Oil saturation grid to be populated (fraction).
    :param gas_saturation: Gas saturation grid to be populated (fraction).
    :param active: Boolean mask indicating active cells.
    :return: Tuple of (water_saturation, oil_saturation, gas_saturation) grids.
    """
    # Define transition boundaries
    gas_oil_contact_top = gas_oil_contact - gas_oil_transition_thickness / 2
    gas_oil_contact_bottom = gas_oil_contact + gas_oil_transition_thickness / 2
    oil_water_contact_top = oil_water_contact - oil_water_transition_thickness / 2
    oil_water_contact_bottom = oil_water_contact + oil_water_transition_thickness / 2

    # Gas cap (shallower than GOC transition) - uses Sor_gas
    gas_cap = (depth_grid < gas_oil_contact_top) & active
    gas_saturation[gas_cap] = (
        1.0 - residual_oil_saturation_gas[gas_cap] - connate_water_saturation[gas_cap]
    )
    oil_saturation[gas_cap] = residual_oil_saturation_gas[gas_cap]
    water_saturation[gas_cap] = connate_water_saturation[gas_cap]

    # Gas–oil transition: blend from (gas cap with Sor_gas) to (oil zone with Sgr)
    gas_oil_zone = (
        (depth_grid >= gas_oil_contact_top)
        & (depth_grid <= gas_oil_contact_bottom)
        & active
    )
    if np.any(gas_oil_zone):
        frac = (
            depth_grid[gas_oil_zone] - gas_oil_contact_top
        ) / gas_oil_transition_thickness
        # Clip frac to avoid numerical issues
        frac = np.clip(frac, 0.0, 1.0)
        weight = np.power(frac, transition_curvature_exponent)

        # Gas: from (1 - Sor_gas - Swc) to Sgr
        gas_saturation[gas_oil_zone] = (
            1.0
            - residual_oil_saturation_gas[gas_oil_zone]
            - connate_water_saturation[gas_oil_zone]
        ) * (1 - weight) + residual_gas_saturation[gas_oil_zone] * weight

        # Oil: from Sor_gas to (1 - Swc - Sgr)
        oil_saturation[gas_oil_zone] = (
            residual_oil_saturation_gas[gas_oil_zone] * (1 - weight)
            + (
                1.0
                - connate_water_saturation[gas_oil_zone]
                - residual_gas_saturation[gas_oil_zone]
            )
            * weight
        )

        # Water: remains at Swc throughout transition
        water_saturation[gas_oil_zone] = connate_water_saturation[gas_oil_zone]

    # Oil zone (between transitions) - original oil accumulation
    oil_zone = (
        (depth_grid > gas_oil_contact_bottom)
        & (depth_grid < oil_water_contact_top)
        & active
    )
    oil_saturation[oil_zone] = (
        1.0 - connate_water_saturation[oil_zone] - residual_gas_saturation[oil_zone]
    )
    water_saturation[oil_zone] = connate_water_saturation[oil_zone]
    gas_saturation[oil_zone] = residual_gas_saturation[oil_zone]

    # Oil–water transition: blend from (oil zone) to (water zone with Sor_water)
    oil_water_zone = (
        (depth_grid >= oil_water_contact_top)
        & (depth_grid <= oil_water_contact_bottom)
        & active
    )
    if np.any(oil_water_zone):
        frac = (
            depth_grid[oil_water_zone] - oil_water_contact_top
        ) / oil_water_transition_thickness
        # Clip frac to avoid numerical issues
        frac = np.clip(frac, 0.0, 1.0)
        weight = np.power(frac, transition_curvature_exponent)

        # Water: from Swc to (1 - Sor_water)
        water_saturation[oil_water_zone] = (
            connate_water_saturation[oil_water_zone] * (1 - weight)
            + (1.0 - residual_oil_saturation_water[oil_water_zone]) * weight
        )

        # Oil: from (1 - Swc - Sgr) to Sor_water
        oil_saturation[oil_water_zone] = (
            1.0
            - connate_water_saturation[oil_water_zone]
            - residual_gas_saturation[oil_water_zone]
        ) * (1 - weight) + residual_oil_saturation_water[oil_water_zone] * weight

        # Gas: from Sgr to 0
        gas_saturation[oil_water_zone] = residual_gas_saturation[oil_water_zone] * (
            1 - weight
        )

    # Water zone (deeper than OWC transition) - uses Sor_water
    water_zone = (depth_grid > oil_water_contact_bottom) & active
    water_saturation[water_zone] = 1.0 - residual_oil_saturation_water[water_zone]
    oil_saturation[water_zone] = residual_oil_saturation_water[water_zone]
    gas_saturation[water_zone] = 0.0
    return water_saturation, oil_saturation, gas_saturation


def _normalize_saturations(
    water_saturation: NDimensionalGrid[NDimension],
    oil_saturation: NDimensionalGrid[NDimension],
    gas_saturation: NDimensionalGrid[NDimension],
    active: np.typing.NDArray,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Normalize saturations to ensure they sum to 1.0 in all active cells.

    This function ensures mass balance by forcing the saturation constraint:
        Sw + So + Sg = 1.0

    The normalization process:
    1. Computes total saturation: total = Sw + So + Sg
    2. Identifies valid cells: active cells with total > 0
    3. Normalizes each phase: S_phase = S_phase / total
    4. Zeros out saturations in inactive cells

    Inactive cells (porosity <= 0 or NaN) are set to zero saturation for all phases.

    :param water_saturation: Water saturation grid (fraction).
    :param oil_saturation: Oil saturation grid (fraction).
    :param gas_saturation: Gas saturation grid (fraction).
    :param active: Boolean mask indicating active cells.
    :return: Tuple of normalized (water_saturation, oil_saturation, gas_saturation) grids.
    :note: All saturations in active cells are guaranteed to sum to exactly 1.0 after normalization.
    """
    total = water_saturation + oil_saturation + gas_saturation
    valid = active & (total > 0)

    water_saturation[valid] /= total[valid]
    oil_saturation[valid] /= total[valid]
    gas_saturation[valid] /= total[valid]

    # Ensure inactive cells are zero
    water_saturation[~active] = 0.0
    oil_saturation[~active] = 0.0
    gas_saturation[~active] = 0.0
    return water_saturation, oil_saturation, gas_saturation


def seed_phase_saturation(
    oil_saturation_grid: NDimensionalGrid[NDimension],
    cells: typing.Sequence[typing.Tuple[int, int, int]],
    phase: typing.Union[FluidPhase, typing.Literal["gas", "water"]],
    gas_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    seed_saturation: typing.Optional[float] = None,
    relperm_table: typing.Optional[RelativePermeabilityTable] = None,
    inplace: bool = False,
) -> typing.Tuple[
    typing.Optional[NDimensionalGrid[NDimension]],
    NDimensionalGrid[NDimension],
    typing.Optional[NDimensionalGrid[NDimension]],
]:
    """
    Seed the (injected) phase saturation at injector cells to break the cold-start
    deadlock that occurs when the (injected) phase has zero relative permeability
    at initial conditions.

    In black-oil simulation the well productivity index is proportional to
    the (injected) phase mobility. When the (injected) phase saturation is zero at
    t=0, kr=0 exactly, PI=0, and the injector contributes nothing to the pressure
    equation regardless of BHP. No flux enters the cell, saturation stays zero,
    and the deadlock persists indefinitely. Setting a small nonzero saturation
    just above the kr=0 plateau at the injector perforations breaks this cycle
    on the first timestep.

    Oil saturation is reduced by the seed delta at every affected cell to preserve
    the saturation constraint Sw + So + Sg = 1. Oil is always chosen as the
    displaced phase because gas injection displaces oil (not connate water) and
    water injection displaces oil (not residual gas).

    When `relperm_table` is provided the seed is auto-detected by evaluating
    kr at a representative initial saturation state and scanning for the minimum
    (injected) phase saturation that yields kr > 0, stepped one table entry further
    into the nonzero region for numerical margin. When `seed_saturation` is
    provided it is used directly and `relperm_table` is ignored. At least one
    of the two must be supplied.

    :param oil_saturation_grid: Oil saturation grid (fraction, 0-1). Reduced by
        the seed delta at each injector cell to preserve the saturation constraint.
        Shape: (nx, ny, nz).
    :param cells: List of (i, j, k) grid indices identifying all
        perforated cells of the injection well.
    :param phase: Phase being targeted/injected. Must be `FluidPhase.GAS` or
        `FluidPhase.WATER`. Determines which saturation grid is seeded and
        which relative permeability curve is scanned for auto-detection.
    :param gas_saturation_grid: Gas saturation grid (fraction, 0-1). Required
        when `phase == FluidPhase.GAS`. Shape: (nx, ny, nz).
    :param water_saturation_grid: Water saturation grid (fraction, 0-1). Required
        when `phase == FluidPhase.WATER`. Shape: (nx, ny, nz).
    :param seed_saturation: Explicit seed value (fraction). When provided,
        `relperm_table` is ignored and this value is applied directly. The
        caller is responsible for ensuring kr_phase(seed) > 0.
    :param relperm_table: Optional relative permeability table used to auto-detect
        the minimum targeted/injected phase saturation that yields kr > 0. The table is
        evaluated at a representative initial state (Sw=min(water_saturation_grid),
        So=1-Sw, Sg=0 for gas; Sw scanning from 0 upward for water) and the
        first saturation entry with kr > 0 is selected, stepped one entry further
        for margin. Ignored if `seed_saturation` is provided. If neither
        `seed_saturation` nor `relperm_table` is provided a ValidationError
        is raised.
    :param inplace: If True, modifies the input grids in place and returns them.
        If False (default), operates on copies and leaves originals unchanged.

    :returns: Tuple of (water_saturation_grid, oil_saturation_grid,
        gas_saturation_grid). Grids not relevant to the targeted/injected phase are
        returned unchanged (and may be None if not provided). If `inplace=True`
        the returned objects are the same as the inputs.

    :raises ValidationError: If `phase` is not GAS or WATER.
    :raises ValidationError: If the saturation grid for the targeted/injected phase is
        not provided.
    :raises ValidationError: If neither `seed_saturation` nor `relperm_table`
        is provided.
    :raises ValidationError: If `relperm_table` is provided but all kr values
        for the targeted/injected phase are zero.
    :raises ValidationError: If applying the seed delta would reduce oil
        saturation below zero at any injector cell.
    """
    if isinstance(phase, str):
        phase = FluidPhase(phase)

    if phase not in (FluidPhase.GAS, FluidPhase.WATER):
        raise ValidationError(f"`phase` must be 'gas' or 'water', got {phase!r}.")
    if phase == FluidPhase.GAS and gas_saturation_grid is None:
        raise ValidationError("`gas_saturation_grid` is required when `phase` is gas.")
    if phase == FluidPhase.WATER and water_saturation_grid is None:
        raise ValidationError(
            "`water_saturation_grid` is required when `phase` is water."
        )
    if seed_saturation is None and relperm_table is None:
        raise ValidationError(
            "Either `seed_saturation` or `relperm_table` must be provided."
        )

    if not inplace:
        oil_saturation_grid = oil_saturation_grid.copy()
        if gas_saturation_grid is not None:
            gas_saturation_grid = gas_saturation_grid.copy()
        if water_saturation_grid is not None:
            water_saturation_grid = water_saturation_grid.copy()

    # Determine seed value
    if seed_saturation is not None:
        seed = float(seed_saturation)
    else:
        assert relperm_table is not None  # guarded above
        # Build a representative initial saturation state to scan kr values.
        # Use minimum water saturation (connate) as the base Sw.
        base_water_sat = (
            float(np.min(water_saturation_grid))
            if water_saturation_grid is not None
            else 0.0
        )

        if phase == FluidPhase.GAS:
            # Scan increasing Sg from 0 upward at connate Sw.
            # Step through candidate Sg values from the table by decrementing So.
            # Use 100 evenly spaced Sg candidates from 0 to (1 - base_water_sat).
            gas_sat_candidates = np.linspace(0.0, 1.0 - base_water_sat, 200)
            oil_saturation_candidates = 1.0 - base_water_sat - gas_sat_candidates
            kr_values = relperm_table(
                water_saturation=np.full_like(gas_sat_candidates, base_water_sat),
                oil_saturation=oil_saturation_candidates,
                gas_saturation=gas_sat_candidates,
            )
            krg_values = np.asarray(kr_values["gas"])
            nonzero_mask = krg_values > 0.0
            if not np.any(nonzero_mask):
                raise ValidationError(
                    "All kr_g values evaluated by `relperm_table` are zero across "
                    "the full gas saturation range. Cannot auto-detect seed saturation."
                )
            first_nonzero_idx = int(np.argmax(nonzero_mask))
            # Step one candidate further into nonzero region for margin
            target_idx = min(first_nonzero_idx + 1, len(gas_sat_candidates) - 1)
            seed = float(gas_sat_candidates[target_idx])

        else:  # FluidPhase.WATER
            # Scan increasing Sw from base_water_sat upward.
            water_saturation_candidates = np.linspace(base_water_sat, 1.0, 200)
            oil_saturation_candidates = 1.0 - water_saturation_candidates
            kr_values = relperm_table(
                water_saturation=water_saturation_candidates,
                oil_saturation=oil_saturation_candidates,
                gas_saturation=np.zeros_like(water_saturation_candidates),
            )
            krw_values = np.asarray(kr_values["water"])
            nonzero_mask = krw_values > 0.0
            if not np.any(nonzero_mask):
                raise ValidationError(
                    "All kr_w values evaluated by `relperm_table` are zero across "
                    "the full water saturation range. Cannot auto-detect seed saturation."
                )
            first_nonzero_idx = int(np.argmax(nonzero_mask))
            target_idx = min(
                first_nonzero_idx + 1, len(water_saturation_candidates) - 1
            )
            seed = float(water_saturation_candidates[target_idx])

    # Apply seed at each cell
    target_phase_grid: NDimensionalGrid[NDimension] = (  # type: ignore[assignment]
        gas_saturation_grid if phase == FluidPhase.GAS else water_saturation_grid
    )

    for cell in cells:
        i, j, k = cell
        current = float(target_phase_grid[i, j, k])
        if current >= seed:
            continue

        delta = seed - current
        current_oil_sat = float(oil_saturation_grid[i, j, k])
        if current_oil_sat - delta < 0.0:
            raise ValidationError(
                f"Cannot seed cell ({i}, {j}, {k}): oil saturation So={current_oil_sat:.6f} "
                f"is insufficient to absorb seed delta={delta:.6f}. "
                "Reduce `seed_saturation` or check initial conditions."
            )
        target_phase_grid[i, j, k] = seed
        oil_saturation_grid[i, j, k] = current_oil_sat - delta

    return water_saturation_grid, oil_saturation_grid, gas_saturation_grid
