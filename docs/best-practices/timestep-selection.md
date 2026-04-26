# Timestep Selection

## Overview

The timestep is the discrete time interval over which the simulator advances the solution. Choosing the right timestep size is critical for balancing accuracy, stability, and performance. A timestep that is too large can cause the simulation to become unstable or produce inaccurate results. A timestep that is too small wastes computation on unnecessarily fine time resolution that does not improve the answer.

In BORES, the `Timer` class manages timestep control. It supports both constant and adaptive timestep strategies. Adaptive control is almost always preferred for production simulations because it automatically adjusts the timestep based on how the solution is evolving, taking larger steps when conditions change slowly and smaller steps during rapid transients like well startups or saturation front arrivals.

The behavior of the timestep controller depends heavily on the evolution scheme you are using. Explicit and IMPES schemes have a hard stability limit governed by the CFL (Courant-Friedrichs-Lewy) condition. If the timestep exceeds this limit, the simulation will produce non-physical oscillations or diverge entirely. Fully implicit schemes do not have a strict CFL limit but can lose accuracy or fail to converge at large timesteps. Understanding these constraints is essential for choosing good timestep parameters.

---

## The CFL Condition

The CFL number is a dimensionless ratio that compares the physical wave speed (how fast information travels through the reservoir) to the numerical grid speed (how far information can travel in one grid cell per timestep). For an explicit or IMPES scheme, the CFL number must stay below 1.0 for stability. In practice, you want it comfortably below 1.0 to account for local variations that are not captured by a single global CFL estimate.

The CFL number depends on the flow velocity, the cell size, and the timestep:

$$
\text{CFL} = \frac{v \cdot \Delta t}{\Delta x}
$$

where $v$ is the flow velocity, $\Delta t$ is the timestep, and $\Delta x$ is the cell size. Higher velocities (near wells, in high-permeability zones) and smaller cells both push the CFL number higher for a given timestep.

In BORES, the `maximum_cfl` parameter on the `Timer` controls the stability limit. The default is 0.9, which provides a 10% safety margin below the theoretical limit of 1.0. If the simulation detects that the CFL number has been exceeded during a step, it rejects the step and retries with a smaller timestep.

```python
import bores

timer = bores.Timer(
    initial_step_size=bores.Time(days=1),
    maximum_step_size=bores.Time(days=30),
    minimum_step_size=bores.Time(seconds=1),
    simulation_time=bores.Time(years=5),
    maximum_cfl=0.9,   # Default, good for most problems
)
```

!!! tip "When to Lower maximum_cfl"

    If you see oscillations in saturation or pressure even though the simulation is not rejecting steps, try lowering `maximum_cfl` to 0.7 or 0.5. Some problems with sharp permeability contrasts or complex well patterns have local CFL numbers that are not perfectly captured by the global estimate, and a lower target provides extra margin.

---

## Initial Timestep

The initial timestep sets the size of the very first step in the simulation. Choosing it well saves time because the adaptive controller does not have to spend many steps ramping up from a tiny value, or backing off from a value that was too large.

A good rule of thumb for the initial timestep:

| Scenario | Suggested Initial Timestep |
| --- | --- |
| Wells starting production/injection | 0.1 to 1.0 days |
| Pressure depletion (no wells near boundaries) | 1.0 to 10.0 days |
| High-rate gas injection | 0.01 to 0.1 days |
| Miscible flooding | 0.1 to 0.5 days |

If you are unsure, start with 1 day. The adaptive controller will reduce it if needed on the first step, and you will lose at most a few seconds of computation.

---

## Adaptive Timestep Control

The `Timer` in BORES uses a sophisticated adaptive algorithm that adjusts the timestep based on multiple criteria. Understanding these parameters helps you tune the controller for your specific problem.

### Growth Parameters

After a successful timestep, the controller tries to increase the timestep for the next step. The rate of increase is governed by several parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `maximum_growth_per_step` | 1.3 | Maximum multiplicative growth per step (1.3 = 30% increase). |
| `ramp_up_factor` | `None` | Optional additional growth factor applied when the simulation has been stable for several steps. Set to 1.1 or 1.2 for faster ramp-up. |
| `growth_cooldown_steps` | 5 | Number of consecutive successful steps required before `ramp_up_factor` is applied. |
| `step_size_smoothing` | 0.7 | Smoothing factor (0 to 1) that blends the new timestep with the previous one to avoid jumpy behavior. Higher values give smoother transitions. |

The `maximum_growth_per_step` is the most important parameter. The default of 1.3 means the timestep can grow by at most 30% from one step to the next. This prevents wild jumps after a period of small timesteps. For very stable problems (simple depletion), you can increase this to 1.5 or even 2.0. For problems with sudden events (well shutins, rate changes), keep it at 1.3 or lower.

### Backoff Parameters

When a timestep is rejected (CFL violation, solver non-convergence, or excessive saturation change), the controller reduces the timestep and retries. The reduction factor depends on the severity of the failure:

| Parameter | Default | Description |
| --- | --- | --- |
| `backoff_factor` | 0.5 | Timestep reduction for mild failures (halves the timestep). |
| `aggressive_backoff_factor` | 0.25 | Timestep reduction for severe failures (quarters the timestep). |

The controller automatically chooses between these based on how badly the step failed. A mild CFL violation (CFL slightly above the limit) uses `backoff_factor`. A severe CFL violation (CFL more than 2x the limit) or solver divergence uses `aggressive_backoff_factor`.

### Minimum and Maximum Timestep

The `minimum_step_size` and `maximum_step_size` parameters set absolute bounds on the timestep. The minimum prevents the controller from taking infinitesimally small steps that would make the simulation crawl. The maximum prevents individual steps from being so large that accuracy is lost even when the CFL condition would allow it.

For most simulations, set `minimum_step_size` to a value small enough that the solver can always converge (typically 0.1 to 10 seconds), and set `maximum_step_size` to a value that ensures you capture the time-scale of interest (typically 30 to 90 days for multi-year simulations).

!!! warning "Minimum Timestep and Simulation Failure"

    If the adaptive controller reduces the timestep to `minimum_step_size` and the step still fails, the simulation will raise a `SimulationError`. This usually means there is a fundamental problem with the model (extreme permeability contrasts, unphysical fluid properties, or a well operating outside its capacity). See the [Error Handling](errors.md) page for guidance on diagnosing these failures.

---

## Timestep Strategies by Evolution Scheme

Different evolution schemes interact with the timestep in different ways. Here is a summary of what to expect and how to configure the timer for each scheme.

### IMPES (Implicit Pressure, Explicit Saturation)

IMPES is the default scheme in BORES and is the most commonly used for black-oil simulation. The pressure equation is solved implicitly (unconditionally stable for pressure), but the saturation update is explicit and subject to the CFL condition. This means the timestep is limited by how fast saturation fronts move through the grid.

Recommended timer settings for IMPES:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=1),
    maximum_step_size=bores.Time(days=30),
    minimum_step_size=bores.Time(seconds=1),
    simulation_time=bores.Time(years=10),
    maximum_cfl=0.9,
    maximum_growth_per_step=1.3,
    backoff_factor=0.5,
)
```

### Explicit

Fully explicit schemes solve both pressure and saturation explicitly. This makes each step very cheap but imposes a stricter CFL condition (both pressure and saturation waves must satisfy the CFL limit). You will need smaller timesteps than IMPES, typically by a factor of 5 to 10.

Recommended adjustments for explicit schemes:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=1),
    maximum_step_size=bores.Time(days=5),
    minimum_step_size=bores.Time(milliseconds=100),
    simulation_time=bores.Time(years=5),
    maximum_cfl=0.5,        # More conservative for fully explicit
    maximum_growth_per_step=1.2,   # Slower growth
)
```

### Implicit

Fully implicit schemes solve both pressure and saturation together using Newton iteration. There is no CFL stability limit, so timesteps can be much larger. However, Newton convergence can fail at very large timesteps, and accuracy degrades as the timestep grows. The adaptive controller monitors Newton iteration counts and adjusts accordingly.

Recommended settings for implicit schemes:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=5),
    maximum_step_size=bores.Time(days=90),
    minimum_step_size=bores.Time(days=0.1),
    simulation_time=bores.Time(years=20),
    maximum_cfl=5.0,         # Much higher, not a hard limit
    maximum_growth_per_step=1.5,    # Can grow faster
    ramp_up_factor=1.2,         # Additional growth when stable
)
```

---

## Common Problems and Solutions

### Simulation Takes Too Long

If your simulation is running slowly, the most common cause is timesteps that are too small. Check the timer's step history to see if the adaptive controller is keeping the timestep well below `maximum_step_size`. If so, the CFL condition is the bottleneck, and you have two options: coarsen your grid (see [Grid Design](grid-design.md)) or switch to a fully implicit scheme.

### Oscillating Timestep

If the timestep rapidly increases and decreases from one step to the next, the smoothing factor may be too low. Increase `step_size_smoothing` to 0.8 or 0.9 to stabilize the timestep trajectory. Also check whether wells are cycling between rate and BHP control, which creates sudden changes in flow velocities that confuse the adaptive controller.

### Simulation Crashes at Well Startup

Well startups create sudden, localized velocity spikes that can violate the CFL condition on the very first step. Use a small initial timestep (0.01 to 0.1 days) and let the controller ramp up naturally. If the problem persists, check that the well's initial rate or BHP is physically reasonable.

### Constant Timestep Mode

For benchmarking or debugging, you may want to disable adaptive control and use a fixed timestep. Simply set `maximum_step_size` equal to `initial_step_size` and `minimum_step_size` to the same value. The controller will use that fixed value for every step.

---

## Quick Reference

| Parameter | When to Adjust | Direction |
| --- | --- | --- |
| `maximum_cfl` | Oscillations, instability | Lower (0.5 to 0.7) |
| `maximum_growth_per_step` | Timestep jumps too fast | Lower (1.1 to 1.2) |
| `backoff_factor` | Too many rejected steps | Lower (0.3) for faster recovery |
| `minimum_step_size` | Simulation stalls at minimum | Lower, but investigate root cause |
| `maximum_step_size` | Missing transient events | Lower to capture short-lived phenomena |
| `step_size_smoothing` | Erratic timestep behavior | Higher (0.8 to 0.9) |
| `ramp_up_factor` | Slow recovery after transients | Set to 1.1 to 1.2 |
