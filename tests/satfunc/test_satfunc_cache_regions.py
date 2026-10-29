"""Tests for evaluating the saturation function cache over precomputed region groups."""

import numpy as np
import pytest

from bores.blackoil.caches.satfunc import compute_satfunc_cache
from bores.blackoil.satfunc.regions import SatFunc
from bores.blackoil.satfunc.tables import SatFuncTables
from bores.reservoir.regions import Regions, group_cells_by_region
from tests.satfunc.test_capillary_hysteresis import _tabular as _capillary_table
from tests.satfunc.test_relperm_hysteresis import _tabular as _relperm_table


def _satfunc():
    first = _relperm_table("baker_linear_rule")
    second = _relperm_table("baker_linear_rule", oil_water_reference_phase="wetting")
    return SatFunc({
        1: SatFuncTables(relative_permeability=first, capillary_pressure=_capillary_table()),
        2: SatFuncTables(relative_permeability=second, capillary_pressure=None),
        3: SatFuncTables(relative_permeability=first, capillary_pressure=None),
    })


def _saturations(size, seed=4):
    rng = np.random.default_rng(seed)
    sw = rng.uniform(0.25, 0.6, size)
    sg = rng.uniform(0.07, 0.3, size)
    return sw, 1.0 - sw - sg, sg


def test_groups_hold_ascending_cells_of_each_region():
    groups = group_cells_by_region(saturation_region=np.array([2, 1, 2, 1, 3], dtype=np.int32))
    assert groups.saturation_regions == (1, 2, 3)
    assert groups.imbibition_regions == (0, 0, 0)
    assert [cells.tolist() for cells in groups.cell_indices] == [[1, 3], [0, 2], [4]]


def test_one_region_is_a_single_group_over_every_cell():
    groups = group_cells_by_region(saturation_region=np.ones(5, dtype=np.int32))
    assert len(groups) == 1
    assert groups.cell_indices == (None,)


def test_groups_follow_the_pair_of_saturation_and_imbibition_regions():
    groups = group_cells_by_region(
        saturation_region=np.array([1, 1, 2, 1], dtype=np.int32),
        imbibition_region=np.array([2, 1, 1, 2], dtype=np.int32),
    )
    assert list(zip(groups.saturation_regions, groups.imbibition_regions, strict=True)) == [
        (1, 1),
        (1, 2),
        (2, 1),
    ]
    assert [cells.tolist() for cells in groups.cell_indices] == [[1], [0, 3], [2]]


def test_regions_without_a_saturation_region_cannot_be_grouped():
    from bores.errors import ValidationError

    with pytest.raises(ValidationError):
        Regions().group_saturation_cells()


@pytest.mark.parametrize("precomputed", [True, False])
def test_each_cell_gets_the_value_of_its_own_regions_tables(precomputed):
    satfunc = _satfunc()
    size = 60
    region = (np.arange(size) % 3 + 1).astype(np.int32)
    sw, so, sg = _saturations(size)
    groups = group_cells_by_region(saturation_region=region) if precomputed else None
    cache = compute_satfunc_cache(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        saturation_region=region,
        satfunc=satfunc,
        region_groups=groups,
        dtype=np.float64,
    )
    for number in (1, 2, 3):
        cells = region == number
        tables = satfunc[number]
        relperm, derivatives = tables.relative_permeability.evaluate_with_derivatives(
            water_saturation=sw[cells], oil_saturation=so[cells], gas_saturation=sg[cells]
        )
        np.testing.assert_array_equal(cache.water_relative_permeability[cells], relperm["water"])
        np.testing.assert_array_equal(cache.oil_relative_permeability[cells], relperm["oil"])
        np.testing.assert_array_equal(cache.dkrg_dsg[cells], derivatives["dkrg_dsg"])
        if tables.capillary_pressure is None:
            assert np.isnan(cache.oil_water_capillary_pressure[cells]).all()
        else:
            capillary, _ = tables.capillary_pressure.evaluate_with_derivatives(
                water_saturation=sw[cells], oil_saturation=so[cells], gas_saturation=sg[cells]
            )
            np.testing.assert_array_equal(
                cache.oil_water_capillary_pressure[cells], capillary["oil_water"]
            )
    np.testing.assert_array_equal(cache.saturation_region, region)


def test_reusing_the_cache_leaves_no_value_from_the_previous_regions():
    satfunc = _satfunc()
    size = 30
    sw, so, sg = _saturations(size)
    with_capillary = np.ones(size, dtype=np.int32)
    without_capillary = np.full(size, 2, dtype=np.int32)

    cache = compute_satfunc_cache(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        saturation_region=with_capillary,
        satfunc=satfunc,
        region_groups=group_cells_by_region(saturation_region=with_capillary),
        dtype=np.float64,
    )
    assert not np.isnan(cache.oil_water_capillary_pressure).any()

    reused = compute_satfunc_cache(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        saturation_region=without_capillary,
        satfunc=satfunc,
        region_groups=group_cells_by_region(saturation_region=without_capillary),
        out=cache,
    )
    assert reused is cache
    assert np.isnan(reused.oil_water_capillary_pressure).all()
    assert np.isnan(reused.dpcgo_dsg).all()
    assert not np.isnan(reused.water_relative_permeability).any()
