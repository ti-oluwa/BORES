# Time Step Control

## Overview

Time step control determines how much simulated time the solver advances in each iteration. Choosing the right time step size is critical because steps that are too large cause numerical instability, oscillations, or outright divergence, while steps that are too small waste computational effort without improving accuracy.

In explicit and IMPES schemes, time step size is constrained by the CFL (Courant-Friedrichs-Lewy) condition, which sets a physical limit on how far information can travel through the grid in a single step. If the time step allows a saturation or pressure front to cross more than one grid cell, the solution becomes unstable. Even in implicit schemes where unconditional stability is theoretically guaranteed, excessively large steps can cause the Newton solver to struggle or produce inaccurate solutions.

BORES uses an adaptive time stepping system through the `Timer` class. Rather than requiring you to guess a fixed step size, the timer monitors CFL numbers, saturation changes, pressure changes, and solver performance to automatically grow the step size when conditions are smooth and shrink it when conditions become challenging. This adaptive approach typically uses 3 to 10 times fewer total steps than a fixed step size that is conservative enough to handle the worst case, while maintaining the same level of accuracy and stability.

The adaptive system is particularly important in reservoir simulation because the physics changes dramatically during a run. Early time steps near wells and injection fronts require very small steps, while late-time depletion far from wells can use much larger steps. A fixed step size must accommodate the most demanding period, wasting effort during the easier periods.

---

## The `Time()` Helper

All timer parameters are specified in seconds, but thinking in seconds is inconvenient for reservoir simulation where durations range from hours to years. The `Time()` function converts human-readable time components into seconds:

```python
import bores

# Single component
one_day = bores.Time(days=1)           # 86400.0
six_hours = bores.Time(hours=6)        # 21600.0
three_years = bores.Time(years=3)      # ~94608000.0

# Combined components
mixed = bores.Time(days=1, hours=6)    # 108000.0
```

`Time()` accepts any combination of `milliseconds`, `seconds`, `minutes`, `hours`, `days`, `weeks`, `months`, and `years`. Components are additive. One year is defined as 365.25 days (by default), and one month is one-twelfth of a year (approximately 30.4 days). These are engineering approximations, not calendar-accurate durations, and they are standard conventions in reservoir simulation.

You can use `Time()` anywhere a duration in seconds is expected, including all `Timer` parameters and simulation analysis functions.

---

## Configuring the Timer

The `Timer` class controls all aspects of time stepping. At minimum, you must specify four parameters: the initial step size, the maximum and minimum allowed step sizes, and the total simulation time.

```python
import bores

timer = bores.Timer(
    initial_step_size=bores.Time(days=1),
    maximum_step_size=bores.Time(days=10),
    minimum_step_size=bores.Time(hours=1),
    simulation_time=bores.Time(years=3),
)
```

The initial step size is the starting point for the adaptive algorithm. It should be conservative enough that the first few steps succeed without rejection. A good starting point is 0.5 to 2 days for most problems. If you start too large, the timer will reject the first few steps and reduce the size automatically, but each rejection wastes a solver call.

The maximum step size caps how large the timer can grow. Even when conditions are very smooth, you generally do not want steps larger than 10 to 30 days because the linearization errors in PVT property updates accumulate. The minimum step size sets a floor below which the timer will not go. If the timer hits this floor repeatedly, it raises a `TimingError` after `maximum_rejections` consecutive rejections (default 10), indicating that the problem may be poorly configured.

### Full Timer Parameters

| Parameter | Default | Description |
|---|---|---|
| `initial_step_size` | (required) | Starting step size in seconds |
| `maximum_step_size` | (required) | Upper bound on step size |
| `minimum_step_size` | (required) | Lower bound on step size |
| `simulation_time` | (required) | Total simulation duration |
| `maximum_cfl` | 0.9 | Default CFL limit for adaptive adjustments |
| `backoff_factor` | 0.5 | Multiplier when a step is rejected |
| `aggressive_backoff_factor` | 0.25 | Multiplier for severe rejections |
| `ramp_up_factor` | `None` | Optional growth multiplier after cooldown |
| `maximum_steps` | `None` | Optional hard limit on total step count |
| `maximum_rejections` | 10 | Maximum consecutive rejections before error |
| `maximum_growth_per_step` | 1.3 | Maximum multiplicative growth per step (30%) |
| `growth_cooldown_steps` | 5 | Successful steps required before ramp-up |
| `cfl_safety_margin` | 0.85 | Safety factor applied to CFL targets |
| `step_size_smoothing` | 0.2 | EMA smoothing factor (0 = none, 1 = max) |
| `metrics_history_size` | 10 | Number of recent steps tracked for trends |
| `failure_memory_window` | 5 | Number of recent failures remembered |

---

## Adaptive Time Stepping

The adaptive algorithm works by monitoring multiple criteria at each step and adjusting the next step size accordingly. The process has two phases: acceptance (where the step succeeded and we decide whether to grow) and rejection (where the step failed and we must shrink).

### Step Acceptance

When a step succeeds, the timer evaluates several adjustment factors:

1. **CFL-based adjustment**: If the CFL number is well below the threshold (utilization < 70%), the timer allows growth. If the CFL is above 90% of the threshold, growth is suppressed. The `cfl_safety_margin` (default 0.85) targets a CFL below the absolute limit.

2. **Saturation change adjustment**: The timer compares the maximum saturation change against the allowed limit (from `Config`). Low utilization (< 30%) allows up to 30% growth, while high utilization (> 95%) triggers a 15% reduction even though the step was accepted.

3. **Pressure change adjustment**: Same logic as saturation, comparing actual pressure change against the allowed limit.

4. **Newton iteration adjustment**: For implicit schemes, if the solver needed more than 10 iterations, the step size is reduced by 30%. If it converged in fewer than 4 iterations after several stable steps, a 20% growth is allowed.

5. **Performance trend analysis**: The timer tracks the last 10 steps and detects concerning trends. If CFL numbers are consistently high or increasing, or if Newton iterations are consistently above 8, a performance factor below 1.0 further dampens growth.

All factors are multiplied together, then the result is capped by `maximum_growth_per_step` (default 1.3, meaning no more than 30% growth per step). The final value is smoothed through an exponential moving average controlled by `step_size_smoothing`, which prevents erratic oscillations in step size.

### Step Rejection

When a step fails (CFL exceeded, saturation change too large, solver diverged), the timer uses intelligent backoff based on the specific failure cause:

| Failure Cause | Overshoot Ratio | Backoff Factor |
|---|---|---|
| Mild CFL violation (1.0 to 1.5x) | < 1.5 | 0.6 to 0.9 (proportional) |
| Moderate CFL violation (1.5 to 2.0x) | 1.5 to 2.0 | 0.5 |
| Severe CFL violation (> 2.0x) | > 2.0 | 0.3 |
| Moderate saturation overshoot | < 2.0 | 0.5 to proportional |
| Large saturation overshoot | 2.0 to 3.0 | 0.4 |
| Severe saturation overshoot | > 3.0 | 0.25 |
| Newton solver struggling (> 15 iterations) | n/a | 0.5 |
| Newton solver failing (> 20 iterations) | n/a | 0.3 |

When multiple criteria are violated simultaneously, the timer uses the most conservative (smallest) backoff factor. If no specific failure information is available, it falls back to `backoff_factor` (0.5) or `aggressive_backoff_factor` (0.25).

The timer also remembers recently failed step sizes and avoids proposing sizes near those values. If a proposed size is within 15% of a recently failed size, it is reduced by an additional 20%.

### Constant Step Size Mode

If you set `initial_step_size`, `maximum_step_size`, and `minimum_step_size` to the same value, the timer automatically enters constant step size mode. In this mode, all adaptive logic is bypassed and every step uses the specified size:

```python
# Fixed 1-day steps
timer = bores.Timer(
    initial_step_size=bores.Time(days=1),
    maximum_step_size=bores.Time(days=1),
    minimum_step_size=bores.Time(days=1),
    simulation_time=bores.Time(years=1),
)
```

Constant step size is rarely optimal, but it can be useful for debugging (to isolate whether time stepping is contributing to an issue) or for comparing against analytical solutions at specific time points.

---

## Saturation and Pressure Change Limits

The `Config` class provides per-phase saturation change limits and a pressure change limit that the timer uses to judge whether a step is acceptable. These limits are the primary mechanism for controlling accuracy in IMPES and explicit schemes.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    maximum_oil_saturation_change=0.5,      # Default
    maximum_water_saturation_change=0.4,    # Default
    maximum_gas_saturation_change=0.85,     # Default
    maximum_pressure_change=100.0,          # Default, psi
)
```

Each limit specifies the maximum absolute change allowed in a single time step. If any cell in the grid exceeds the limit, the step is rejected and the timer reduces the step size.

These defaults are deliberately permissive to allow large time steps in the common case. Most reservoir simulations involve gradual, smooth changes where large steps are perfectly accurate. The defaults are designed so that the adaptive timer can take big steps during quiet periods and only restrict step size when the physics genuinely demands it. You should feel free to leave them at their defaults for most work and only tighten them if you observe specific issues.

The gas saturation limit (0.85) is the most permissive because gas saturation fronts are inherently sharp and restricting them too aggressively forces very small time steps without proportional accuracy improvement. The water limit (0.4) is more conservative because water fronts tend to be smoother and more amenable to accurate large-step tracking. The oil limit (0.5) sits in the middle. The pressure change limit of 100 psi works well for typical reservoir pressures of 1,000 to 5,000 psi, representing roughly a 2 to 10% relative change.

If you want tighter accuracy (for example, for validation studies or detailed front tracking), you can reduce these limits. If you want faster runs for screening purposes, you can relax them further. The simulation remains stable either way because the timer enforces the limits strictly; looser limits simply allow larger steps.

### Adjusting Limits by Scenario

| Scenario | Saturation Limits | Pressure Limit |
|---|---|---|
| Standard waterflood | Defaults are fine | 100 psi |
| Gas injection | Reduce water to 0.3 | 75 psi |
| Low-pressure reservoir (< 1000 psi) | Defaults are fine | 25 to 50 psi |
| High-pressure reservoir (> 5000 psi) | Defaults are fine | 150 to 200 psi |
| Tight convergence / validation | Reduce all by 30 to 50% | 50 psi |
| Fast screening runs | Relax all by 50% or more | 200 to 300 psi |

Tighter limits improve accuracy but require more time steps. Looser limits improve performance at the cost of some accuracy in tracking sharp fronts. In most cases, the defaults provide a good balance between speed and accuracy. If you observe pressure oscillations or material balance errors, tighten the pressure limit first. If saturation fronts show non-physical overshoots, tighten the relevant saturation limit.

---

## CFL Thresholds

The CFL (Courant-Friedrichs-Lewy) number measures how far a wave front moves relative to the grid cell size in a single time step. A CFL number above 1.0 means the front crosses more than one cell per step, which causes instability in explicit methods.

The `Config` provides separate CFL thresholds for pressure and saturation:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    cfl_threshold=0.6,   # Default
    pressure_cfl_threshold=0.9,     # Default
)
```

The saturation CFL threshold (0.6) is more conservative than the pressure threshold (0.9) because saturation transport is hyperbolic and more sensitive to CFL violations. The pressure equation is parabolic (or elliptic in incompressible limits) and tolerates higher CFL numbers.

These thresholds interact with the timer's `maximum_cfl` (default 0.9) and `cfl_safety_margin` (default 0.85). The timer targets a CFL of `threshold * safety_margin`, so with defaults the effective target for saturation is approximately 0.51 and for pressure approximately 0.77.

!!! tip "CFL and Scheme Selection"

    CFL thresholds only affect the explicit and IMPES schemes. In the IMPES scheme, only the saturation CFL threshold applies (pressure is implicit). In the fully implicit scheme, CFL is not a stability constraint, but the saturation and pressure change limits still apply as accuracy controls.

---

## Ramp-Up Factor

The `ramp_up_factor` provides an additional multiplicative growth on top of the adaptive adjustments. It only activates after `growth_cooldown_steps` consecutive successful steps and only when all monitoring criteria (CFL, saturation change, pressure change) are below 70% of their limits.

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=0.5),
    maximum_step_size=bores.Time(days=15),
    minimum_step_size=bores.Time(hours=1),
    simulation_time=bores.Time(years=5),
    ramp_up_factor=1.2,              # 20% extra growth
    growth_cooldown_steps=5,          # After 5 stable steps
)
```

The ramp-up factor is useful when you want the simulation to reach large step sizes quickly after an initial transient period (for example, after well startup or injection rate changes). Without it, the timer grows conservatively based solely on the monitoring criteria. With a ramp-up factor of 1.2, the timer can grow up to 56% per step (1.3 max growth * 1.2 ramp-up), though this is still capped by `maximum_step_size`.

Setting `ramp_up_factor` to `None` (the default) disables this feature entirely, relying solely on the adaptive criteria for growth.

---

## Step Size Smoothing

The `step_size_smoothing` parameter controls an exponential moving average (EMA) filter applied to the proposed step size. Without smoothing, the step size can oscillate between large and small values when the adaptive criteria give conflicting signals.

The smoothing factor ranges from 0.0 (no smoothing, the proposed step is used directly) to 1.0 (maximum smoothing, the step size barely changes). The default of 0.2 provides light smoothing that prevents erratic oscillations while still allowing the step size to respond quickly to changing conditions.

```python
# No smoothing (responsive but potentially oscillatory)
timer = bores.Timer(
    ...,
    step_size_smoothing=0.0,
)

# Heavy smoothing (very stable but slow to adapt)
timer = bores.Timer(
    ...,
    step_size_smoothing=0.5,
)
```

The EMA formula is: `ema = smoothing * ema_previous + (1 - smoothing) * proposed`. With the default of 0.2, each new step size is 80% the proposed value and 20% the previous EMA, giving the timer a short memory that damps fluctuations without introducing significant lag.

---

## Timer State and Checkpointing

The `Timer` tracks its complete internal state, including step count, elapsed time, recent performance metrics, and failed step size history. You can save and restore this state for simulation checkpointing:

```python
# Save timer state
timer_state = timer.dump_state()

# Later, restore from saved state
restored_timer = bores.Timer.load_state(timer_state)
```

The `dump_state()` method returns a `TimerState` dictionary containing all configuration parameters and runtime state. The `load_state()` class method reconstructs a timer with the exact same state, allowing a simulation to resume from a checkpoint without losing the adaptive algorithm's learned behavior.

This is particularly valuable for long-running simulations where you want to resume after a system interruption without restarting from the beginning. The restored timer will propose the same step size and use the same performance history as the original.

---

## Recommended Configurations

### Standard Waterflood

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=1),
    maximum_step_size=bores.Time(days=10),
    minimum_step_size=bores.Time(hours=1),
    simulation_time=bores.Time(years=3),
)
```

Waterfloods have moderate dynamics with a fairly smooth saturation front. The default parameters work well for most waterflood scenarios.

### Gas Injection

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=0.5),
    maximum_step_size=bores.Time(days=5),
    minimum_step_size=bores.Time(hours=0.5),
    simulation_time=bores.Time(years=3),
)
```

Gas injection creates sharper fronts and higher velocities. Smaller initial and maximum step sizes prevent early rejections. The smaller minimum step size accommodates the sharp gas front arrival.

### Primary Depletion

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=2),
    maximum_step_size=bores.Time(days=30),
    minimum_step_size=bores.Time(days=1),
    simulation_time=bores.Time(years=10),
)
```

Depletion has very smooth, gradual pressure decline with no injection fronts. Larger step sizes are appropriate because the physics changes slowly and uniformly across the grid.

### Well Startup / Rate Change

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=6),
    maximum_step_size=bores.Time(days=10),
    minimum_step_size=bores.Time(minutes=30),
    simulation_time=bores.Time(years=5),
    ramp_up_factor=1.15,
    growth_cooldown_steps=3,
)
```

Well events cause rapid near-wellbore transients that need very small initial steps. The ramp-up factor with a short cooldown allows the timer to grow quickly once the transient passes.

---

## Troubleshooting

!!! warning "Maximum Rejections Exceeded"

    If you see `TimingError: Maximum number of consecutive time step rejections exceeded`, the timer hit the minimum step size and still could not produce an acceptable step. Common causes:

    - The minimum step size is too large for the problem dynamics. Try reducing it.
    - Permeability contrast is extreme. Try a stronger preconditioner (AMG or CPR).
    - Well rates are too high for the grid resolution. Reduce the rate or refine the grid near the well.
    - Initial conditions are inconsistent (saturations do not sum to 1.0, pressure below bubble point without free gas).

!!! warning "Step Size Stuck at Minimum"

    If the timer repeatedly reports step sizes at or near the minimum, the adaptive algorithm is struggling. This usually indicates that the physics demands small steps throughout, not just during a transient. Consider relaxing the saturation or pressure change limits, using the implicit scheme (which allows larger steps), or coarsening the grid.
