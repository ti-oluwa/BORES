"""Loading the reservoir summary vectors a deck requests."""

import numpy as np

from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
from bores.reservoir import summary as rs
from bores.schedule.summary import Summary

__all__ = [
    "BLOCK_SUMMARY_VECTORS",
    "FIELD_SUMMARY_VECTORS",
    "REGION_SUMMARY_VECTORS",
    "get_fluid_in_place_regions",
    "load_summary_quantities",
]

FIELD_SUMMARY_VECTORS: dict[str, type[rs.FieldVector]] = {
    "FPR": rs.FieldAveragePressure,
    "FPRH": rs.FieldHydrocarbonAveragePressure,
    "FPORV": rs.FieldPoreVolume,
    "FOIP": rs.FieldOilInPlace,
    "FOIPL": rs.FieldOilInPlaceInOil,
    "FOIPG": rs.FieldOilInPlaceInGas,
    "FGIP": rs.FieldGasInPlace,
    "FGIPL": rs.FieldGasInPlaceDissolved,
    "FGIPG": rs.FieldGasInPlaceFree,
    "FWIP": rs.FieldWaterInPlace,
}
"""Deck mnemonic to the `bores.reservoir.summary` class it requests. Field-wide, no object list."""

REGION_SUMMARY_VECTORS: dict[str, type[rs.RegionVector]] = {
    "RPR": rs.RegionAveragePressure,
    "RPRH": rs.RegionHydrocarbonAveragePressure,
    "RPORV": rs.RegionPoreVolume,
    "ROIP": rs.RegionOilInPlace,
    "ROIPL": rs.RegionOilInPlaceInOil,
    "ROIPG": rs.RegionOilInPlaceInGas,
    "RGIP": rs.RegionGasInPlace,
    "RGIPL": rs.RegionGasInPlaceDissolved,
    "RGIPG": rs.RegionGasInPlaceFree,
    "RWIP": rs.RegionWaterInPlace,
}
"""Deck mnemonic to the `bores.reservoir.summary` class it requests. One instance per `FIPNUM` region."""

BLOCK_SUMMARY_VECTORS: dict[str, type[rs.BlockVector]] = {
    "BPORV": rs.BlockPoreVolume,
    "BPR": rs.BlockPressure,
    "BPRESSUR": rs.BlockPressure,
    "BGSAT": rs.BlockGasSaturation,
    "BSGAS": rs.BlockGasSaturation,
    "BOSAT": rs.BlockOilSaturation,
    "BSOIL": rs.BlockOilSaturation,
    "BWSAT": rs.BlockWaterSaturation,
    "BSWAT": rs.BlockWaterSaturation,
    "BRS": rs.BlockGasOilRatio,
    "BRV": rs.BlockVaporizedOilGasRatio,
    "BOKR": rs.BlockOilRelativePermeability,
    "BWKR": rs.BlockWaterRelativePermeability,
    "BGKR": rs.BlockGasRelativePermeability,
}
"""
Deck mnemonic to the `bores.reservoir.summary` class it requests. One instance per listed cell.
Mnemonics that name the same vector (`BPR` and `BPRESSUR`) map to the same class.
"""


def get_fluid_in_place_regions(deck_file: DeckFile) -> list[int]:
    """
    Return every fluid-in-place region number a deck defines.

    :param deck_file: The deck to read `FIPNUM` from.
    :returns: The distinct `FIPNUM` values in ascending order, or `[1]` if the deck has none.
    """
    fip_numbers = deck_file.get("FIPNUM")
    if fip_numbers is None:
        return [1]
    return sorted(int(region) for region in np.unique(np.asarray(fip_numbers)))


def load_summary_quantities(deck_file: DeckFile) -> list[Summary]:
    """
    Load every reservoir summary vector the deck's `SUMMARY` section requests.

    - Field vectors (`FOIP`, `FPR`, ...) take no object list and request one vector for the field.
    - Region vectors (`ROIP`, `RPR`, ...) request one vector per listed fluid-in-place region. With
      no region list they request every `FIPNUM` region in the deck, or region 1 if the deck has
      none.
    - Block vectors (`BPR`, `BPORV`, `BGSAT`, `BOSAT`, `BWSAT`, `BRS`, `BRV`, `BOKR`, `BWKR`, `BGKR` and
      their alternative spellings `BPRESSUR`, `BSGAS`, `BSOIL`, `BSWAT`) request one vector per
      listed cell. A cell listed more than once for the same vector is recorded once.

    :param deck_file: The deck to read the `SUMMARY` section from.
    :returns: The requested vectors, in the order field, region, block. Empty if the deck
        requests none.
    :raises DeckParseError: If a block vector lists a cell outside the deck's grid, or the deck
        has no grid dimensions to place it in.
    """
    quantities: list[Summary] = []
    for mnemonic, field_summary_cls in FIELD_SUMMARY_VECTORS.items():
        if deck_file.has(mnemonic):
            quantities.append(field_summary_cls())

    for mnemonic, region_summary_cls in REGION_SUMMARY_VECTORS.items():
        requested = deck_file.get(mnemonic)
        if requested is None:
            continue
        regions = [int(region) for region in requested] or get_fluid_in_place_regions(deck_file)
        quantities.extend(region_summary_cls(region=region) for region in regions)

    dimensions = deck_file.dimensions
    block_keys: set[str] = set()
    for mnemonic, block_summary_cls in BLOCK_SUMMARY_VECTORS.items():
        requested = deck_file.get(mnemonic)
        if requested is None:
            continue
        if dimensions is None:
            raise DeckParseError(
                f"{mnemonic}: cannot place block vectors without grid dimensions "
                "(`SPECGRID`/`DIMENS` not found)."
            )
        for cell in requested:
            i, j, k = cell["i"], cell["j"], cell["k"]
            if not (
                1 <= i <= dimensions.nx and 1 <= j <= dimensions.ny and 1 <= k <= dimensions.nz
            ):
                raise DeckParseError(
                    f"{mnemonic}: cell ({i}, {j}, {k}) is outside the {dimensions.nx} x "
                    f"{dimensions.ny} x {dimensions.nz} grid."
                )
            quantity = block_summary_cls(i=i - 1, j=j - 1, k=k - 1)
            # A cell listed twice, or under two names for the same vector, is recorded once.
            if quantity.key in block_keys:
                continue
            block_keys.add(quantity.key)
            quantities.append(quantity)
    return quantities
