# Well Controls

## Overview

Well controls define the operating strategy for each well: how the simulator determines the flow rate and bottom-hole pressure at every timestep. The choice of control strategy has a major impact on simulation behavior because it determines the boundary condition at each well location.

In real field operations, wells are typically controlled by one of two modes: constant rate (the operator sets a target production or injection rate, and the well delivers that rate as long as it physically can) or constant pressure (the operator sets a bottom-hole pressure, and the well produces or injects whatever rate the reservoir can deliver at that drawdown). Most production wells start under rate control and switch to pressure control when the reservoir can no longer sustain the target rate. BORES models this with the `AdaptiveRateControl`, which automatically switches between rate and pressure modes.

For production wells in three-phase simulation, a single-phase control is rarely sufficient because oil, water, and gas all flow simultaneously. The standard approach is to control the rate of one "primary" phase (usually oil) and let the other phases produce at whatever rate corresponds to the resulting bottom-hole pressure. BORES provides `CoupledRateControl` for this purpose, and `MultiPhaseControl` for cases where you need separate controls for each phase.

---

## Sign Convention

Throughout BORES, flow rates follow a consistent sign convention:

- **Positive rates = injection** (fluid entering the reservoir)
- **Negative rates = production** (fluid leaving the reservoir)

This convention applies to all control types, all phase rates, and all flow rate outputs. When you specify `target_rate=-500.0`, you are requesting 500 STB/day of production. When you specify `target_rate=800.0`, you are requesting 800 STB/day (or SCF/day for gas) of injection.

---

## `BHPControl`

The simplest control strategy: fix the bottom-hole pressure and let the flow rate be determined by Darcy's law. The rate depends on the pressure difference between the reservoir and the wellbore, the well index, and the phase mobility.

```python
import bores

control = bores.BHPControl(
    bhp=1500.0,  # psi
)
```

At each timestep, the flow rate is computed as:

$$q = WI \cdot \frac{k_{r\alpha}}{\mu_\alpha B_\alpha} \cdot (P_{res} - P_{wf})$$

If $P_{res} > P_{wf}$, the rate is negative (production). If $P_{res} < P_{wf}$, the rate is positive (injection). BHP control is most commonly used for injection wells where you want to maintain a constant injection pressure, or for production wells on artificial lift where the BHP is determined by the lift equipment.

| Parameter | Default | Description |
| --- | --- | --- |
| `bhp` | required | Bottom-hole pressure in psi (must be positive) |
| `target_phase` | `None` | If set, this control only applies to the specified phase |
| `clamp` | `None` | Optional rate clamp to prevent backflow |

!!! warning "BHP Control for Production"

    When using BHP control for production, the production rate depends entirely on the reservoir pressure. As the reservoir depletes, the rate naturally declines. If reservoir pressure drops below the specified BHP, the well will stop producing (or even start injecting if no clamp is set). Always pair BHP production controls with a `ProductionClamp()` to prevent unintended injection.

---

## `RateControl`

Maintains a target flow rate regardless of reservoir conditions, as long as a minimum BHP constraint is satisfied.

```python
import bores

# Production at 500 STB/day with 1000 psi minimum BHP
control = bores.RateControl(
    target_rate=-500.0,    # STB/day (negative = production)
    bhp_limit=1000.0,      # psi (minimum BHP for producers)
)

# Injection at 800 STB/day with 5000 psi maximum BHP
control = bores.RateControl(
    target_rate=800.0,     # STB/day (positive = injection)
    bhp_limit=5000.0,      # psi (maximum BHP for injectors)
)
```

The `bhp_limit` has different meanings depending on the flow direction: for production wells, it is the minimum allowable BHP (the well cannot produce if the required BHP to achieve the target rate drops below this limit); for injection wells, it is the maximum allowable BHP (the well cannot inject if the required BHP exceeds this limit, which could indicate fracture risk).

| Parameter | Default | Description |
| --- | --- | --- |
| `target_rate` | required | Target rate in STB/day or SCF/day (positive = injection, negative = production) |
| `bhp_limit` | `None` | Minimum BHP for production, maximum BHP for injection (psi) |
| `target_phase` | `None` | If set, this control only applies to the specified phase |
| `clamp` | `None` | Optional rate clamp to prevent backflow |

---

## `AdaptiveRateControl`

The most commonly used control for production wells. It operates at a constant target rate as long as the BHP stays above the limit, then automatically switches to BHP control when the rate is no longer achievable. This mimics the real behavior of production wells as reservoir pressure declines.

```python
import bores

control = bores.AdaptiveRateControl(
    target_rate=-500.0,    # STB/day (production)
    bhp_limit=1000.0,      # psi
    target_phase="oil",    # Only controls oil phase rate
)
```

The adaptive behavior works as follows:

1. At each timestep, the simulator computes the BHP required to deliver the target rate.
2. If the required BHP is above the `bhp_limit`, the well operates in **rate mode**: it delivers the full target rate.
3. If the required BHP would fall below the `bhp_limit`, the well switches to **BHP mode**: it produces at the BHP limit, and the rate declines below target.

This is the industry-standard approach for modeling depletion-drive reservoirs where production rate is maintained as long as possible, then allowed to decline as pressure support diminishes.

| Parameter | Default | Description |
| --- | --- | --- |
| `target_rate` | required | Target rate in STB/day or SCF/day |
| `bhp_limit` | required | Switching pressure in psi |
| `target_phase` | `None` | Phase this control applies to |
| `clamp` | `None` | Optional rate clamp |

---

## `CoupledRateControl`

The recommended control for production wells in three-phase simulation. It fixes the rate of one "primary" phase (typically oil) and computes the BHP required to deliver that rate. All other phases (water and gas) then produce at whatever their natural Darcy rates are at the resulting BHP.

This is the standard approach in reservoir simulation because it reflects how production wells actually operate: the operator controls oil production rate (or sometimes gas rate for gas wells), and water and gas are produced as byproducts. The water cut and gas-oil ratio are determined by the reservoir conditions, not by the well control.

```python
import bores

control = bores.CoupledRateControl(
    primary_phase=bores.FluidPhase.OIL,
    primary_control=bores.AdaptiveRateControl(
        target_rate=-500.0,
        target_phase="oil",
        bhp_limit=1000.0,
    ),
    secondary_clamp=bores.ProductionClamp(),
)
```

The `primary_phase` parameter specifies which phase drives the BHP calculation. The `primary_control` is any single-phase control (`RateControl` or `AdaptiveRateControl`) applied to that phase. The `secondary_clamp` is an optional safety clamp applied to the non-primary phases to prevent unphysical behavior (like injection of water through a production well when reservoir pressure exceeds wellbore pressure in the water zone).

| Parameter | Default | Description |
| --- | --- | --- |
| `primary_phase` | required | Phase whose rate is fixed (`OIL`, `GAS`, or `WATER`) |
| `primary_control` | required | Rate control for the primary phase |
| `secondary_clamp` | `None` | Optional clamp for non-primary phases |

### How It Works

At each timestep, for each grid cell the well perforates:

1. The primary phase control computes the BHP required to deliver the target oil rate (for example).
2. That BHP is used for all phases. Water and gas rates are computed from Darcy's law at that BHP.
3. If a `secondary_clamp` is set (recommended), any secondary phase that would flow in the wrong direction (e.g., water injection through a production well) is clamped to zero.

This approach ensures physical consistency: all phases share the same wellbore pressure, and the total production rate is the sum of the individual phase rates at that pressure.

!!! tip "Always Use CoupledRateControl for Producers"

    For production wells in three-phase simulations, always use `CoupledRateControl` rather than a bare `AdaptiveRateControl`. A bare `AdaptiveRateControl` applies the same target rate to ALL phases independently, which is physically incorrect. With `CoupledRateControl`, you control only the phase you care about and let the others flow naturally.

---

## `MultiPhaseControl`

For cases where you need explicit, independent control over each phase. This provides separate control objects for oil, gas, and water, each with their own target rates and BHP limits.

```python
import bores

control = bores.MultiPhaseControl(
    oil_control=bores.AdaptiveRateControl(
        target_rate=-500.0,
        target_phase="oil",
        bhp_limit=1000.0,
    ),
    gas_control=bores.AdaptiveRateControl(
        target_rate=-200000.0,
        target_phase="gas",
        bhp_limit=800.0,
    ),
    water_control=bores.BHPControl(
        bhp=1000.0,
        target_phase="water",
    ),
)
```

Each phase operates under its own control independently. This is less physically realistic than `CoupledRateControl` (because in reality, all phases share the same wellbore pressure), but it is useful for certain history-matching scenarios where you want to prescribe known production rates for each phase separately.

| Parameter | Default | Description |
| --- | --- | --- |
| `oil_control` | `None` | Control for oil phase (set to `None` to exclude oil) |
| `gas_control` | `None` | Control for gas phase (set to `None` to exclude gas) |
| `water_control` | `None` | Control for water phase (set to `None` to exclude water) |

!!! info "When to Use MultiPhaseControl"

    Use `MultiPhaseControl` when you have measured production data for each phase separately and want to replay those rates in a history-matching study. For predictive simulations, `CoupledRateControl` is more physically meaningful because it lets the reservoir determine the water cut and GOR based on mobility and relative permeability.

---

## Rate Clamps

Rate clamps are safety mechanisms that prevent unphysical flow. They check the computed flow rate or BHP and clamp it if a condition is met.

### `ProductionClamp`

Prevents injection through a production well. If the computed rate is positive (injection), it clamps the rate to zero. If the computed BHP exceeds reservoir pressure (which would drive injection), it clamps BHP to reservoir pressure.

```python
import bores

clamp = bores.ProductionClamp()

# Use with any control
control = bores.BHPControl(bhp=1500.0, clamp=clamp)
```

### `InjectionClamp`

Prevents production through an injection well. If the computed rate is negative (production), it clamps the rate to zero.

```python
clamp = bores.InjectionClamp()

control = bores.RateControl(
    target_rate=800.0,
    bhp_limit=5000.0,
    clamp=clamp,
)
```

!!! warning "Always Use Clamps"

    Rate clamps are strongly recommended for all wells. Without them, a production well can start injecting if reservoir pressure drops below the wellbore pressure, and an injection well can start producing if wellbore pressure drops below reservoir pressure. These scenarios are physically possible but rarely intended, and they can cause numerical instability.

---

## Choosing the Right Control

| Scenario | Recommended Control |
| --- | --- |
| Production well (general case) | `CoupledRateControl` with `AdaptiveRateControl` |
| Gas production well | `CoupledRateControl` with `primary_phase=GAS` |
| Water injection | `RateControl` or `AdaptiveRateControl` |
| Gas injection | `RateControl` or `AdaptiveRateControl` |
| Constant-pressure production | `BHPControl` with `ProductionClamp` |
| History matching (known rates per phase) | `MultiPhaseControl` |
| Artificial lift | `BHPControl` (BHP set by lift equipment) |
