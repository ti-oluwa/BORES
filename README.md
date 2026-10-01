<p align="center">
  <img src="docs/images/logo.svg" alt="BORES Logo" width="150">
</p>

<h2 align="center">BORES</h2>

<p align="center">
  <strong>Black-Oil Reservoir Simulation Framework</strong>
</p>

[![Documentation](https://img.shields.io/badge/docs-ti--oluwa.github.io%2Fbores-blue)](https://ti-oluwa.github.io/bores)
[![PyPI](https://img.shields.io/pypi/v/bores-framework)](https://pypi.org/project/bores-framework/)
[![License](https://img.shields.io/github/license/ti-oluwa/bores)](LICENSE)

BORES is a Python framework for 2D/3D block grid black-oil reservoir simulation of three-phase (oil, water, gas) flow in porous media. You can build a simulation case by hand through easy-to-use APIs or load one straight from an Eclipse/GRDECL-style deck, then run and analyze the simulation.

> [!IMPORTANT]
> **Disclaimer**: BORES' goal is to support **educational, research, and prototyping** work. It is not production-grade code yet and should not be used for critical business decisions or regulatory compliance. Results should be validated against established simulators before any real-world application.

**Full documentation @** [https://ti-oluwa.github.io/BORES](https://ti-oluwa.github.io/BORES)

## Work in progress

BORES is under active development and going through a fairly major migration right now, from a purely Cartesian grid model to a fully free-form one (corner-point, unstructured, NNCs, the whole deal), so a good chunk of the framework is being rebuilt as part of that. The API on `main` does not match the published docs or the latest PyPI release at the moment, and the quick example below is closer to what's actually usable today than the old high-level model-builder API used to be.

Rough status, best of my knowledge as of right now:

- **Working**: Eclipse/GRDECL-style deck parsing (grid, PVT, saturation functions, regions, boundary conditions, operators like `BOX`/`EQUALS`/`ADD`/`MULTIPLY`/`COPY`), grid construction and viewing (Cartesian, corner-point, and polyhedral), PVT (correlations and table-based), relative permeability and capillary pressure models, reservoir initialization/equilibration, boundary conditions (Carter-Tracy and Fetkovich analytic aquifers, flux aquifers, attached to the grid through `AQUANCON`), well models with BHP/rate control and a schedule/event API for time-varying well and group behavior, multiple linear solvers with preconditioner support, and storage backends with serialization.
- **Working, one call from deck to a ready-to-run model**: `SimulationCase.from_deck` compiles the model, resolves the initial reservoir state, and reads the schedule, all from one deck. The run's workspace (reservoir, wells, boundary conditions) builds lazily off that on first access via `case.workspace`.
- **In progress**: wrapping up the well module, specifically wellbore hydraulics (five correlations now: homogeneous, Beggs and Brill, Hagedorn and Brown, Gray, and Woldesemayat and Ghajar) and VFP table support. This is the last piece of wells left, so it's close.
- **Not started yet**: the main solver kernel(s).
- **Planned order of work**: finish wells (hydraulics and VFP), consolidate everything onto a single units module and retire the overlapping bits still living in `constants`, then the main solver kernel(s), then testing and finalizing.

On the solver side, the plan is to get a fully implicit kernel working first, and possibly circle back to add an IMPES scheme later once that's stable. Neither is wired up on `main` yet, so don't expect `bores.monitor(...)`-style simulation runs to work right now, everything up to and including building a compiled, ready-to-run case does though.

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

`SimulationCase.from_deck` is the recommended way to load a model: point it at an Eclipse/GRDECL-style `.DATA` file and it builds the grid, PVT, saturation functions, rock model, boundary conditions, wells, and schedule off it, then resolves an initial reservoir state, all in one call.

```python
from bores.deck import DeckFile
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

df = DeckFile(
    "path/to/model.DATA",
    encoding="utf-8",
    unit_system=UnitSystem.FIELD,
)

# One call loads the compiled model, initial reservoir state, and schedule, all read from the deck.
wellbore = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)
case = SimulationCase.from_deck(df, default_wellbore=wellbore, temperature=200.0)

grid = case.model.reservoir.grid
print(f"cells: {grid.n_cells}, faces: {grid.n_faces}, bbox: {grid.bounding_box}")
print(f"mean initial pressure: {case.initial_state.pressure.mean():.1f}")
print(f"wells: {list(case.model.wells.names) if case.model.wells else []}")
print(f"scheduled items: {len(case.schedule)}")
```

If you want more control than a deck gives you, or you're building a model by hand rather than loading one, every piece `SimulationCase` builds internally (`Grid`, `PVT`, `SatFunc`, `Rock`, `Reservoir`, `BlackOil`, `WellSystem`, `BoundaryConditions`) is still usable on its own, `load_case`/`SimulationCase.from_deck` just wires them together for you.

See `examples/` for a fuller version of this, including grid visualization with PyVista.

## Features

What's sort-off working right now (not fully tested yet):

- Eclipse/GRDECL-style deck parsing, including grid, PVT, saturation function, and region keywords, plus `BOX`/`EQUALS`/`ADD`/`MULTIPLY`/`COPY`/`MAXVALUE`/`MINVALUE` operators
- Cartesian and corner-point (free-form) grid construction, with faults, NNCs, and transmissibility multipliers
- Three-phase (oil, water, gas) black-oil PVT, both correlation-based (Standing, Vazquez-Beggs, Hall-Yarborough, and more) and table-based
- Relative permeability models (Brooks-Corey, LET, tabular) with 15+ three-phase mixing rules
- Capillary pressure models (Brooks-Corey, Leverett J-function, Van Genuchten, tabular)
- Reservoir initialization and equilibration (including capillary transition zones, wet-gas EQUIL support)
- Boundary conditions, including Carter-Tracy and Fetkovich analytic aquifers and flux-specified aquifers, attached to grid faces through `AQUANCON`, fully compiled into the solver's data structures
- Well models with BHP/rate control, and a schedule/event API for time-varying well and group behavior
- `SimulationCase`, loading a full compiled model, initial state, and schedule from a deck in one call, with a run workspace (reservoir, wells, boundary conditions) built lazily off that
- Multiple linear solvers (BiCGSTAB, GMRES, CG, direct, etc.) with preconditioner support (from SciPy) (ILU, AMG, CPR)
- HDF5, Zarr, JSON, and YAML storage backends with serialization

In progress:

- Wellbore hydraulics (homogeneous, Beggs and Brill, Hagedorn and Brown, Gray, and Woldesemayat and Ghajar correlations) and VFP table support, the last piece of the well module being wrapped up

Planned, not started yet:

- Consolidating unit handling onto the `units` module and retiring the overlapping bits currently in `constants`
- A fully implicit solver kernel, with an IMPES scheme possibly following once that's stable
- Todd-Longstaff miscible flooding with pressure-dependent miscibility (Solvent support)
- Custom visualization API for model state, simulation or analyses results (1D time series, 2D maps, 3D volume/section rendering)
- Post-simulation analysis (recovery factors, sweep efficiency, front tracking)

## Citing BORES

If you use BORES in academic work, please cite it as:

```bibtex
@software{bores,
  author = {Daniel Toluwalase Afolayan},
  title = {BORES: Black-Oil Reservoir Simulation Framework},
  year = {2026},
  url = {https://github.com/ti-oluwa/bores},
}
```

## Contributing

BORES is being developed by a graduate petroleum engineer with just theoretical/research knowledge and little experience. The project does not have the benefit of decades of field/research experience backing its implementation (atleast in code), so contributions, issues, bug report and fixes, from students,researchers, and domain experts are very welcome and appreciated.

**Reporting issues**: If you find bugs, inaccuracies in the physics, or unexpected behavior, please [open an issue](https://github.com/ti-oluwa/bores/issues) on GitHub with a clear description and, if possible, a minimal example that reproduces the problem. Even better, draft a fix and submit a pull request for review.

**Improvements**: Pull requests for bug fixes, documentation improvements, and enhancements that fall within the scope of a black-oil reservoir simulation framework are welcome. Please keep changes focused and well-tested. Given the migration in progress, it's worth opening an issue first to check a change still fits before putting work into a PR.

**Out of scope**: Changes that go beyond the black-oil formulation (compositional simulation, thermal recovery, etc.) are outside the current scope of the project.

## License

See [LICENSE](LICENSE) for details.
