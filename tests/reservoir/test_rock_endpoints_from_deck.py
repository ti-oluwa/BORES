import numpy as np
import pytest

from bores.deck import DeckFile
from bores.reservoir.rock.model import Rock

HEADER = "METRIC\nSPECGRID\n 2 1 1 1 F /\nPORO\n 0.2 0.2 /\nPERMX\n 100 100 /\n"


class Grid:
    n_cells = 2


def rock_from(extra: str) -> Rock:
    return Rock.from_deck(DeckFile(HEADER + extra), grid=Grid())  # type: ignore[arg-type]


def test_eclipse_names_are_read():
    rock = rock_from("SWL\n 0.04 0.16 /\nSWCR\n 0.12 0.24 /\n")
    np.testing.assert_allclose(rock.connate_water_saturation, [0.04, 0.16])
    np.testing.assert_allclose(rock.irreducible_water_saturation, [0.12, 0.24])


def test_bores_aliases_are_read_when_the_eclipse_name_is_absent():
    rock = rock_from("SWCON\n 0.05 0.06 /\nSWCRIT\n 0.1 0.11 /\n")
    np.testing.assert_allclose(rock.connate_water_saturation, [0.05, 0.06])
    np.testing.assert_allclose(rock.irreducible_water_saturation, [0.1, 0.11])


def test_eclipse_name_wins_over_its_alias():
    rock = rock_from(
        "SWL\n 0.04 0.16 /\nSWCON\n 0.5 0.5 /\nSWCR\n 0.12 0.24 /\nSWCRIT\n 0.9 0.9 /\n"
    )
    np.testing.assert_allclose(rock.connate_water_saturation, [0.04, 0.16])
    np.testing.assert_allclose(rock.irreducible_water_saturation, [0.12, 0.24])


@pytest.mark.parametrize("name", ["SWL", "SWCON"])
def test_absent_critical_saturation_defaults_to_connate_under_either_name(name):
    rock = rock_from(f"{name}\n 0.04 0.16 /\n")
    np.testing.assert_allclose(rock.irreducible_water_saturation, [0.04, 0.16])


def test_other_endpoints_keep_their_eclipse_names():
    rock = rock_from("SOWCR\n 0.13 0.13 /\nSOGCR\n 0.07 0.07 /\nSGCR\n 0.03 0.03 /\n")
    np.testing.assert_allclose(rock.residual_oil_saturation_water, 0.13)
    np.testing.assert_allclose(rock.residual_oil_saturation_gas, 0.07)
    np.testing.assert_allclose(rock.residual_gas_saturation, 0.03)


def test_cells_no_record_assigns_fall_back_to_zero_without_satfunc():
    rock = rock_from("EQUALS\n SWL 0.04 1 1 1 1 1 1 /\n SWCR 0.12 1 1 1 1 1 1 /\n/\n")
    np.testing.assert_allclose(rock.connate_water_saturation, [0.04, 0.0])
    np.testing.assert_allclose(rock.irreducible_water_saturation, [0.12, 0.0])
