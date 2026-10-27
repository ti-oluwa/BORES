from bores.deck import DeckFile
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

df = DeckFile(
    "data/SPE1CASE1.DATA",
    encoding="utf-8",
    unit_system=UnitSystem.FIELD,
)
wellbore = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=df.unit_system)

# One call loads a compiled model, initial reservoir state, and schedule, read from the deck.
case = SimulationCase.from_deck(df, tubing_performance=wellbore, temperature=200.0)

grid = case.model.reservoir.grid
print(f"Cells: {grid.n_cells}, Faces: {grid.n_faces}, Bounding box: {grid.bounding_box}")
print(f"Mean initial pressure: {case.initial_state.pressure.mean():.1f}")
print(f"Wells: {list(case.model.wells.names) if case.model.wells else []}")
print(f"Scheduled items: {len(case.schedule)}")

for item in case.schedule:
    print(item, "\n")

print(f"{len(case.schedule)} scheduled item(s)")
relperm_table = case.model.fluid.satfunc.region(1).relative_permeability
print(relperm_table.evaluate(0.25, 0.15, 0.55))
