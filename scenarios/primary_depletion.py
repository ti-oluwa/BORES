import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", app_title="bores")


@app.cell
def setup_run():
    import logging
    from pathlib import Path

    import numpy as np

    import bores

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_32bit_precision()

    preconditioner_factory = bores.CachedPreconditionerFactory(
        factory="ilu",
        name="cached_ilu",
        update_frequency=10,
        recompute_threshold=0.3,
    )
    preconditioner_factory.register(override=True)

    # Load the new run with the resulting model state from the stabilization run
    run = bores.Run.from_files(
        model_path=Path("./scenarios/runs/stabilization/results/model.h5"),
        config_path=Path("./scenarios/runs/setup/config.yaml"),
        pvt_data_path=Path("./scenarios/runs/setup/pvt.h5"),
    )

    # Production well
    control = bores.CoupledRateControl(
        primary_phase="oil",
        primary_control=bores.AdaptiveRateControl(
            target_rate=-2000,
            bhp_limit=1000,
            clamp=bores.ProductionClamp(),
        ),
        secondary_clamp=bores.ProductionClamp(),
    )
    producer = bores.production_well(
        well_name="P-1",
        perforating_intervals=[
            ((14, 10, 6), (14, 10, 7)),
        ],
        radius=0.3542,
        control=control,
        produced_fluids=(
            bores.ProducedFluid(
                name="Oil",
                phase=bores.FluidPhase.OIL,
                specific_gravity=0.845,
                molecular_weight=180.0,
            ),
            bores.ProducedFluid(
                name="Gas",
                phase=bores.FluidPhase.GAS,
                specific_gravity=0.65,
                molecular_weight=bores.c.MOLECULAR_WEIGHT_CH4,
            ),
            bores.ProducedFluid(
                name="Water",
                phase=bores.FluidPhase.WATER,
                specific_gravity=1.05,
                molecular_weight=bores.c.MOLECULAR_WEIGHT_WATER,
            ),
        ),
        skin_factor=-2.5,
        is_active=True,
    )

    wells = bores.wells_(injectors=None, producers=[producer])
    timer = bores.Timer(
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=15),
        minimum_step_size=bores.Time(minutes=10.0),
        simulation_time=bores.Time(years=15),
        maximum_cfl=0.9,
        ramp_up_factor=1.2,
        backoff_factor=0.5,
        aggressive_backoff_factor=0.25,
        maximum_rejections=20,
    )
    run.config = run.config.with_updates(
        wells=wells,
        timer=timer,
        maximum_gas_saturation_change=0.05,
        maximum_oil_saturation_change=0.05,
        maximum_water_saturation_change=0.05,
        maximum_pressure_change=200.0,
    )
    return Path, bores, run


@app.cell
def save_run(Path, run):
    run.save(Path("./scenarios/runs/primary_depletion/run.h5"))
    return


@app.cell
def make_store(Path, bores):
    store = bores.ZarrStore(
        store=Path("./scenarios/runs/primary_depletion/results/primary_depletion.zarr"),
    )
    return (store,)


@app.cell
def run_simulation(bores, run, store):
    with bores.StateStream(
        run,
        store=store,
        batch_size=20,
        background_io=True,
    ) as stream:
        last_state = stream.last()
    return (last_state,)


@app.cell
def capture_last_state(Path, last_state):
    last_state.model.save(Path("./scenarios/runs/primary_depletion/results/model.h5"))
    return


if __name__ == "__main__":
    app.run()
