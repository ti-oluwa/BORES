"""
Quick start: load a full simulation case from a deck in one call.

This is the same example as the README's Quick Example. See main.py for a
fuller walkthrough that also plots the grid.
"""

from bores.deck import DeckFile
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

df = DeckFile(
    "data/SPE1CASE1.DATA",
    encoding="utf-8",
    unit_system=UnitSystem.FIELD,
)

default_wellbore = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)

# One call gets you a compiled model, initial reservoir state, and schedule,
# all read straight off the deck.
case = SimulationCase.from_deck(df, default_wellbore=default_wellbore, temperature=200.0)

grid = case.model.reservoir.grid
print(f"cells: {grid.n_cells}, faces: {grid.n_faces}, bbox: {grid.bounding_box}")
print(f"mean initial pressure: {case.initial_state.pressure.mean():.1f}")
print(f"wells: {list(case.model.wells.names) if case.model.wells else []}")
print(f"scheduled items: {len(case.schedule)}")
