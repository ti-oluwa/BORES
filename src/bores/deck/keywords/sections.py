"""
Section keywords.

An Eclipse deck is split into sections, each opened by a keyword of its own that takes no data. The
parser does not need them to find anything (every keyword is read wherever it is), but they are
keywords of the deck like any other, so `DeckFile.get` answers whether a section is present.

The sections come in this order in a deck: `RUNSPEC`, `GRID`, `EDIT`, `PROPS`, `REGIONS`,
`SOLUTION`, `SUMMARY`, `SCHEDULE`.
"""

from bores.deck.keywords.base import FlagKeyword

__all__ = [
    "EDIT",
    "GRID",
    "PROPS",
    "REGIONS",
    "RUNSPEC",
    "SCHEDULE",
    "SOLUTION",
    "SUMMARY",
]

RUNSPEC = FlagKeyword("RUNSPEC")
"""`RUNSPEC` - opens the run specification section: dimensions, phases, units, start date."""

GRID = FlagKeyword("GRID")
"""`GRID` - opens the grid section: geometry, permeability, porosity, transmissibility edits."""

EDIT = FlagKeyword("EDIT")
"""`EDIT` - opens the optional edit section: changes to the calculated grid properties."""

PROPS = FlagKeyword("PROPS")
"""`PROPS` - opens the properties section: PVT, saturation functions and end-point scaling."""

REGIONS = FlagKeyword("REGIONS")
"""`REGIONS` - opens the optional regions section: saturation, PVT, equilibration, FIP regions."""

SOLUTION = FlagKeyword("SOLUTION")
"""`SOLUTION` - opens the solution section: the initial state of the reservoir."""

SUMMARY = FlagKeyword("SUMMARY")
"""`SUMMARY` - opens the summary section: which result vectors are written for plotting."""

SCHEDULE = FlagKeyword("SCHEDULE")
"""`SCHEDULE` - opens the schedule section: wells, controls and the time steps to run."""
