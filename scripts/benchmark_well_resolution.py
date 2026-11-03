"""
Times `resolve_wells` resolving wells one at a time against resolving them in one compiled pass
across threads, and checks the two agree.

Run from the repository root:

    python scripts/benchmark_well_resolution.py
    python scripts/benchmark_well_resolution.py --wells 2000 --connections 20 --threads 1,2,4,8

Each well is a producer with its own Beggs and Brill tubing. Half are held to a bottom-hole
pressure and half to an oil rate, so both the single solve and the rate search are timed.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attrs
import numba
import numpy as np

from bores.grids.factories.cartesian import make_cartesian_grid
from bores.precision import with_precision
from bores.types import Orientation
from bores.wells.compile import ProducerControlModeTag, compile_well_system
from bores.wells.factories import make_producer, make_well_controls, make_wells
from bores.wells.hydraulics.base import compile_connection_fluids
from bores.wells.hydraulics.beggs_and_brill import beggs_and_brill_wellbore
from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
from bores.wells.resolution.system import resolve_wells
from bores.wells.workspace import build_wells_workspace
from tests.wells.test_well_derivatives import make_samples

FIELDS = (
    "bhps",
    "oil_rates",
    "water_rates",
    "gas_rates",
    "surface_oil_rates",
    "connection_pressures",
    "connection_oil_rates",
)


def build_system(*, n_wells, n_connections):
    grid = make_cartesian_grid(nx=n_wells, ny=1, nz=n_connections, dx=50.0, dy=50.0, dz=50.0)
    wells, controls = [], {}
    for index in range(n_wells):
        well, control = make_producer(
            f"W{index}",
            surface_location=(50.0 * index + 25.0, 25.0),
            perforation_depths=(0.0, 50.0 * n_connections),
            reference_depth=25.0 * n_connections,
            target_bhp=1000.0,
        )
        wells.append(well)
        controls[f"W{index}"] = control
    system = compile_well_system(
        make_wells(*wells),
        make_well_controls(**controls),
        grid,
        permeabilities={
            orientation: np.full(n_wells * n_connections, 100.0) for orientation in Orientation
        },
    )
    workspace = build_wells_workspace(
        n_wells=n_wells,
        n_connections=len(system.perforations.well_indices),
        well_offsets=system.perforations.well_offsets,
    )
    return system, workspace


def build_fluids(*, n_connections):
    pressures = np.linspace(2900.0, 3100.0, n_connections)
    water = np.linspace(0.15, 0.35, n_connections)
    gas = np.linspace(0.02, 0.12, n_connections)
    return compile_connection_fluids(connection_samples=make_samples(pressures, water, gas))


def run(system, workspace, spec, tubing, fluids):
    resolve_wells(
        compiled_system=system,
        workspace=workspace,
        spec=spec,
        get_tubing_performance=lambda row: tubing,
        get_connection_fluids=lambda row: fluids,
    )


def time_pass(system, workspace, spec, tubing, fluids, *, repeat):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        run(system, workspace, spec, tubing, fluids)
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wells", type=int, default=400)
    parser.add_argument("--connections", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--threads", default=None, help="Comma separated, e.g. 1,2,4,8")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    maximum = numba.config.NUMBA_NUM_THREADS
    thread_counts = (
        [int(value) for value in arguments.threads.split(",")]
        if arguments.threads
        else sorted({1, 2, 4, maximum} | ({8} if maximum >= 8 else set()))
    )
    thread_counts = [count for count in thread_counts if count <= maximum]
    print(
        f"{arguments.wells} wells x {arguments.connections} connections, numba threads available: {maximum}"
    )

    with with_precision(np.dtype(arguments.precision).type):
        system, workspace = build_system(
            n_wells=arguments.wells, n_connections=arguments.connections
        )
        tubing = beggs_and_brill_wellbore(tubing_inner_diameter=0.25)
        fluids = build_fluids(n_connections=arguments.connections)
        spec = WellControlSpec(connection_pressure_mode=ConnectionPressureMode.HYDRAULIC)
        one_at_a_time = attrs.evolve(spec, use_compiled_resolution=False)
        compiled = attrs.evolve(spec, use_compiled_resolution=True)

        run(system, workspace, one_at_a_time, tubing, fluids)
        rate = float(workspace.surface_oil_rates[0])
        half = arguments.wells // 2
        system.controls.control_modes[half:] = ProducerControlModeTag.OIL_RATE
        system.controls.target_rates[half:] = 0.7 * rate

        print("warming up (the first compiled call compiles, which takes a minute or two)...")
        numba.set_num_threads(maximum)
        run(system, workspace, compiled, tubing, fluids)

        reference = build_system(n_wells=arguments.wells, n_connections=arguments.connections)
        reference[0].controls.control_modes[half:] = ProducerControlModeTag.OIL_RATE
        reference[0].controls.target_rates[half:] = 0.7 * rate
        run(reference[0], reference[1], one_at_a_time, tubing, fluids)
        worst = 0.0
        for field in FIELDS:
            expected, actual = getattr(reference[1], field), getattr(workspace, field)
            scale = np.maximum(np.abs(expected), 1e-12)
            worst = max(worst, float(np.nanmax(np.abs(actual - expected) / scale)))
        print(f"largest relative difference, compiled against one at a time: {worst:.2e}\n")

        best_serial, median_serial = time_pass(
            system, workspace, one_at_a_time, tubing, fluids, repeat=max(3, arguments.repeat // 5)
        )
        print(f"{'path':<28}{'best (ms)':>12}{'median (ms)':>14}{'speedup':>10}{'us/well':>10}")
        print(
            f"{'one at a time':<28}{best_serial * 1e3:>12.2f}{median_serial * 1e3:>14.2f}"
            f"{1.0:>10.2f}{best_serial * 1e6 / arguments.wells:>10.1f}"
        )
        for count in thread_counts:
            numba.set_num_threads(count)
            run(system, workspace, compiled, tubing, fluids)
            best, median = time_pass(
                system, workspace, compiled, tubing, fluids, repeat=arguments.repeat
            )
            print(
                f"{f'compiled, {count} thread(s)':<28}{best * 1e3:>12.2f}{median * 1e3:>14.2f}"
                f"{best_serial / best:>10.2f}{best * 1e6 / arguments.wells:>10.1f}"
            )


if __name__ == "__main__":
    main()
