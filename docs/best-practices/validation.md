# Validation

## Overview

A simulation result is only useful if you can trust it. Validation is the process of verifying that your simulation produces physically meaningful results that are consistent with known solutions, material balance constraints, and expected reservoir behavior. Every model should be validated before its results are used for engineering decisions, regardless of how carefully the input data was prepared.

Validation serves two purposes. First, it catches errors in model construction: wrong units, transposed grids, incorrect well placements, or invalid fluid properties. These errors are surprisingly common and can produce results that look plausible at first glance but are quantitatively wrong. Second, it establishes confidence in the numerical solution: verifying that the grid is fine enough, the timestep is small enough, and the solver is converging to the right answer.

This page covers three levels of validation: internal consistency checks (material balance), comparison with analytical solutions, and benchmarking against published test cases. You should perform at least the first two for every model, and the third whenever a suitable benchmark exists for your problem type.

---

## Material Balance Checks

Material balance is the most fundamental validation check. The total amount of each fluid phase in the reservoir (oil, water, gas) must be conserved: what enters through injection wells and boundaries must either accumulate in the reservoir or leave through production wells and boundaries. Any discrepancy indicates a numerical error.

### Computing Material Balance

The material balance error for a phase is defined as:

$$
\text{MB Error} = \frac{|\Delta V_{\text{pore}} - (Q_{\text{in}} - Q_{\text{out}}) \cdot \Delta t|}{V_{\text{pore,initial}}}
$$

where $\Delta V_{\text{pore}}$ is the change in pore volume occupied by that phase, $Q_{\text{in}}$ and $Q_{\text{out}}$ are the volumetric injection and production rates at reservoir conditions, and $V_{\text{pore,initial}}$ is the initial pore volume of that phase.

### Acceptable Error Levels

For engineering purposes, material balance errors should be:

| Error Level | Interpretation |
| --- | --- |
| < 0.1% | Excellent. Results are reliable. |
| 0.1% to 1% | Acceptable. Monitor for drift over time. |
| 1% to 5% | Marginal. Refine grid or reduce timestep. |
| > 5% | Unacceptable. Investigate cause before using results. |

Material balance error tends to accumulate over time. A simulation with 0.01% error per step can reach 1% total error after 100 steps. Check material balance at the end of the simulation, not just at individual timesteps.

### Common Causes of Material Balance Error

**Timestep too large.** The explicit saturation update in IMPES introduces truncation error proportional to the timestep size. Reducing `maximum_step_size` or lowering `maximum_cfl` reduces material balance error.

**Grid too coarse.** Coarse grids cannot resolve sharp saturation fronts accurately, leading to numerical dispersion that manifests as material balance error. Refine the grid and compare.

**Boundary condition leakage.** If boundary conditions are not properly configured (for example, a constant-pressure boundary where a no-flow boundary was intended), fluid can leak in or out of the model domain, causing apparent material balance violations.

---

## Comparison with Analytical Solutions

Analytical solutions exist for several simple reservoir configurations. Comparing your numerical results with these solutions verifies that the simulator is solving the governing equations correctly. If the numerical and analytical solutions agree to within the expected discretization error, you can be confident that the simulator is working properly.

### Buckley-Leverett (Waterflood Front)

The Buckley-Leverett solution gives the position of the water front as a function of time in a 1D linear waterflood. It is the classic test for saturation transport accuracy.

To compare with Buckley-Leverett:

1. Set up a 1D model (nx cells by 1 by 1) with uniform properties
2. Inject water at one end, produce oil at the other
3. Use straight-line relative permeability curves (Corey exponent = 1)
4. Compare the saturation profile at specific times with the analytical front position

The front position is:

$$
x_f = \frac{q \cdot t}{\phi \cdot A} \cdot f'_w(S_{wf})
$$

where $q$ is the injection rate, $\phi$ is porosity, $A$ is the cross-sectional area, and $f'_w(S_{wf})$ is the derivative of the fractional flow curve evaluated at the front saturation.

The numerical solution should show a sharp front at approximately the correct position, with some numerical smearing that decreases as the grid is refined. If the front position is wrong or the front does not sharpen with refinement, there is a problem with the saturation transport calculation.

### Radial Flow (Steady State)

For a single well in an infinite-acting reservoir, the steady-state pressure distribution follows Darcy's law in radial coordinates:

$$
P(r) = P_{wf} + \frac{q \cdot \mu \cdot B}{2\pi \cdot k \cdot h} \cdot \ln\left(\frac{r}{r_w}\right)
$$

where $P_{wf}$ is the well bottom-hole pressure, $q$ is the flow rate, $\mu$ is viscosity, $B$ is the formation volume factor, $k$ is permeability, $h$ is the net pay, and $r_w$ is the wellbore radius.

To test this in BORES, set up a 2D model with a single well at the center and constant-pressure boundaries at the edges. Run to steady state and compare the pressure profile along a radial line from the well to the boundary with the analytical solution.

### Material Balance Time

For depletion-drive reservoirs (no injection, no aquifer), the pressure decline follows the material balance equation:

$$
P = P_i - \frac{N_p \cdot B_o}{c_t \cdot V_p}
$$

where $P_i$ is the initial pressure, $N_p$ is cumulative oil production at surface conditions, $B_o$ is the oil formation volume factor, $c_t$ is total compressibility, and $V_p$ is pore volume.

This is a good first check for any depletion model. Plot pressure versus cumulative production and compare with the straight line predicted by the material balance equation.

---

## Grid Convergence Study

A grid convergence study verifies that your results are not significantly affected by the grid resolution. The procedure is straightforward:

1. Run your model at the current resolution
2. Refine the grid by a factor of 2 in each direction (using more cells, not changing cell values)
3. Run the refined model with the same configuration
4. Compare key outputs (recovery factor, breakthrough time, pressure at a monitoring point)

If the results change by less than 5%, your original resolution is adequate. If they change by more than 5%, refine further until convergence is achieved.

```python
import bores
import numpy as np

# Coarse model
porosity_coarse = bores.build_uniform_grid((10, 10, 5), 0.20)
# Fine model (2x refinement)
porosity_fine = bores.build_uniform_grid((20, 20, 10), 0.20)
```

!!! info "Cost of Grid Refinement"

    Doubling the grid in all three dimensions creates 8x as many cells and typically requires 2x smaller timesteps (for IMPES), resulting in 16x more total computation. Plan your convergence study accordingly and use coarsening (`bores.coarsen_grid`) to quickly test multiple resolutions.

---

## SPE Benchmark Cases

The Society of Petroleum Engineers (SPE) has published several benchmark comparison cases that are widely used to validate reservoir simulators. BORES includes setups for some of these cases in the `benchmarks/` directory.

### SPE 1: Gas Depletion

The SPE 1 problem is a simple gas reservoir under depletion. It tests basic single-phase gas flow, compressibility effects, and gas PVT calculations. This is the simplest benchmark and a good starting point for validating a new simulator installation.

The key validation targets are:

- Pressure decline curve over the production period
- Gas production rate over time
- Cumulative gas production at the end of the simulation

### SPE 9: Three-Phase Flow

The SPE 9 problem involves three-phase flow with gas injection. It tests the interaction between oil, water, and gas phases, including gravity segregation and three-phase relative permeability. This is a more demanding benchmark that exercises most of the simulator's capabilities.

### SPE 10: Heterogeneous Model

The SPE 10 problem uses a highly heterogeneous permeability field with extreme contrasts (up to 7 orders of magnitude). It tests the solver's ability to handle ill-conditioned systems and the grid's ability to capture channelized flow. This is the most challenging standard benchmark.

---

## Validation Checklist

Use this checklist when validating a new model:

1. **Material balance**: Compute and verify that total material balance error is below 1% at the end of the simulation.

2. **Pressure reasonableness**: Check that pressures stay positive, are below the initial pressure for depletion, and are above the initial pressure for injection. Pressures outside these bounds indicate a model error.

3. **Saturation bounds**: Verify that saturations remain between their residual values and 1.0 in every cell at every timestep. Saturations outside these bounds indicate a numerical stability issue.

4. **Production rates**: Check that production rates are positive for production wells and negative for injection wells (or vice versa, depending on sign convention). A production well that switches to injection indicates a pressure configuration error.

5. **Grid convergence**: Run at least one coarsened and one refined version to verify that results are grid-independent.

6. **Analytical comparison**: If an analytical solution exists for your problem configuration, compare against it.

7. **Conservation of volume**: The total pore volume times the sum of saturations should equal the total pore volume in every cell (saturations sum to 1.0).

8. **Physical plausibility**: Do the results make physical sense? Does water move downward under gravity? Does gas rise? Does the displacement front move at roughly the expected velocity? Does pressure stabilize at the expected value?
