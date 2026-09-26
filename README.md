<p align="center">
  <img src="docs/images/logo.svg" alt="BORES Logo" width="200">
</p>

<h1 align="center">BORES</h1>

<p align="center">
  <strong>3-Phase Black-Oil Reservoir Simulation Framework</strong>
</p>

[![Documentation](https://img.shields.io/badge/docs-ti--oluwa.github.io%2Fbores-blue)](https://ti-oluwa.github.io/bores)
[![PyPI](https://img.shields.io/pypi/v/bores-framework)](https://pypi.org/project/bores-framework/)
[![License](https://img.shields.io/github/license/ti-oluwa/bores)](LICENSE)

BORES is a Python framework for 2D/3D block grid black-oil reservoir simulation of three-phase (oil, water, gas) flow in porous media. You can build a simulation case by hand through its Python API or load one straight from an Eclipse/GRDECL-style deck, then run and analyze the simulation.

> [!IMPORTANT]
> **Disclaimer**: BORES is designed for **educational, research, and prototyping purposes**. It is not production-grade software and should not be used for critical business decisions or regulatory compliance. Results should be validated against established commercial simulators before any real-world application.

**Full documentation @** [https://ti-oluwa.github.io/BORES](https://ti-oluwa.github.io/BORES)

## Work in progress

BORES is under active development and going through a fairly major migration right now, from a purely Cartesian grid model to a fully free-form one (corner-point, unstructured, NNCs, the whole deal), so a good chunk of the framework is being rebuilt as part of that. The API on `main` does not match the published docs or the latest PyPI release at the moment, and the quick example below is closer to what's actually usable today than the old high-level model-builder API used to be.

Rough status, best of my knowledge as of right now:

- **Working**: Eclipse/GRDECL-style deck parsing (grid, PVT, saturation functions, regions, operators like `BOX`/`EQUALS`/`ADD`/`MULTIPLY`/`COPY`), grid construction and viewing (Cartesian, corner-point, and polyhedral), PVT (correlations and table-based), relative permeability and capillary pressure models, reservoir initialization/equilibration, multiple linear solvers with preconditioner support, and storage backends with serialization.
- **Built, but needs to be wired into the compiled solver path**: boundary conditions (including a Carter-Tracy analytical aquifer) exist, but still need compilation, and that may take a bit of a refactor to fit the JIT-compiled architecture the solver hot path needs.
- **In progress**: well models (BHP/rate control, schedules, event-driven actions) are being reworked for free-form grids, and I'm currently writing the compilation piece for them, this is where most of the current effort is going.
- **Not started yet**: schedule/event API, and the main solver kernel(s).
- **Planned order of work**: finish wells, then schedule/event API, then boundary condition compilation, then the main solver kernel(s), then testing and finalizing.

On the solver side, the plan is to get a fully implicit kernel working first, and possibly circle back to add an IMPES scheme later once that's stable. Neither is wired up on `main` yet, so don't expect `bores.monitor(...)`-style simulation runs to work right now, everything up to and including initialization does though.

## Installation

```bash
pip install bores-framework
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add bores-framework
```

The PyPI release lags behind `main` quite a bit right now given the migration above, so if you want to try the current deck-based API, installing from `main` directly is the better bet:

```bash
uv add "git+https://github.com/ti-oluwa/bores.git@main"
```

## Quick Example

This is roughly what the current, deck-driven workflow looks like: load an Eclipse/GRDECL-style `.DATA` file, build a grid, PVT, saturation functions and rock model off it, and get an initialized reservoir state out. Well loading is included since it mostly works, but that module is still being rewritten so treat it as unstable for now.

```python
from bores.blackoil.fluids.model import BlackOil
from bores.blackoil.pvt import PVT
from bores.blackoil.satfunc import SatFunc
from bores.deck import DeckFile
from bores.grids import Grid
from bores.initialization import initialize_reservoir_state
from bores.reservoir import Regions, Reservoir, Temperature
from bores.reservoir.rock import Rock
from bores.reservoir.state import Equilibrium
from bores.typing import UnitSystem
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore
from bores.wells.model import WellSystem

df = DeckFile(
    "path/to/model.DATA",
    encoding="utf-8",
    unit_system=UnitSystem.METRIC,
)

# Grid, regions, saturation functions and rock, all read straight off the deck
grid = Grid.from_deck(df)
regions = Regions.from_deck(df, n_cells=grid.n_cells, use_default=True)
satfunc = SatFunc.from_deck(df, mixing_rule="eclipse_rule")
rock = Rock.from_deck(
    df,
    grid=grid,
    rock_region=regions.rock_region,
    satfunc=satfunc,
    saturation_region=regions.saturation_region,
)
reservoir = Reservoir(grid=grid, rock=rock, regions=regions)

# Fluid model
temperature = Temperature(200)
pvt = PVT.from_deck(df, temperature=temperature)
blackoil = BlackOil(pvt=pvt, satfunc=satfunc)

# Equilibrium and initial reservoir state
equilibrium = Equilibrium.from_deck(df)
initial_state = initialize_reservoir_state(
    reservoir=reservoir,
    pvt=pvt,
    deck_file=df,
    equilibrium=equilibrium,
    satfunc=satfunc,
    temperature=temperature,
)

# Wells (still being reworked as part of the free-form migration)
wells = WellSystem.from_deck(
    df,
    grid=grid,
    default_wellbore=homogeneous_wellbore(tubing_inner_diameter=0.5),
)

print(f"cells: {grid.n_cells}, faces: {grid.n_faces}, bbox: {grid.bounding_box}")
print(f"mean initial pressure: {initial_state.pressure.mean():.1f}")
```

See `examples/` for a fuller version of this, including grid visualization with PyVista.

## Features

What's actually working right now:

- Eclipse/GRDECL-style deck parsing, including grid, PVT, saturation function, and region keywords, plus `BOX`/`EQUALS`/`ADD`/`MULTIPLY`/`COPY`/`MAXVALUE`/`MINVALUE` operators
- Cartesian and corner-point (free-form) grid construction, with faults, NNCs, and transmissibility multipliers
- Three-phase (oil, water, gas) black-oil PVT, both correlation-based (Standing, Vazquez-Beggs, Hall-Yarborough, and more) and table-based
- Relative permeability models (Brooks-Corey, LET, tabular) with 15+ three-phase mixing rules
- Capillary pressure models (Brooks-Corey, Leverett J-function, Van Genuchten, tabular)
- Reservoir initialization and equilibration (including capillary transition zones, wet-gas EQUIL support)
- Multiple linear solvers (BiCGSTAB, GMRES, CG, direct) with preconditioner support (ILU, AMG, CPR)
- HDF5, Zarr, JSON, and YAML storage backends with serialization

Built, but not yet plugged into the compiled solver path:

- Boundary conditions, including a Carter-Tracy analytical aquifer, still need to go through compilation, which will likely need a bit of a refactor to fit

In progress:

- Well models with BHP/rate control, schedules, and event-driven actions, being rebuilt for free-form grids, currently writing the compilation piece for these

Planned, not started yet:

- Schedule and event API for time-varying well/group behavior
- A fully implicit solver kernel, with an IMPES scheme possibly following once that's stable
- Todd-Longstaff miscible flooding with pressure-dependent miscibility
- Plotly-based visualization (1D time series, 2D maps, 3D volume rendering)
- Post-simulation analysis (recovery factors, sweep efficiency, front tracking)

## Citing BORES

If you use BORES in academic work, please cite it as:

```bibtex
@software{bores,
  author = {Daniel Toluwalase Afolayan},
  title = {BORES: 3D 3-Phase Black-Oil Reservoir Simulation Framework},
  year = {2026},
  url = {https://github.com/ti-oluwa/bores},
}
```

## Contributing

BORES is being developed by a graduate petroleum engineer with just theoretical knowledge and little research experience. The project does not have the benefit of decades of field experience backing its implementations, so contributions from practitioners and researchers are welcome.

**Reporting issues**: If you find bugs, inaccuracies in the physics, or unexpected behavior, please [open an issue](https://github.com/ti-oluwa/bores/issues) on GitHub with a clear description and, if possible, a minimal example that reproduces the problem.

**Improvements**: Pull requests for bug fixes, documentation improvements, and enhancements that fall within the scope of a black-oil reservoir simulation framework are welcome. Please keep changes focused and well-tested. Given the migration in progress, it's worth opening an issue first to check a change still fits before putting work into a PR.

**Out of scope**: Changes that go beyond the black-oil formulation (compositional simulation, thermal recovery, etc.) are outside the current scope of the project.

## License

See [LICENSE](LICENSE) for details.
