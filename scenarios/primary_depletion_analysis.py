import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools
    from pathlib import Path

    import numpy as np

    import bores

    bores.image_config(scale=3)

    store = bores.ZarrStore(
        store=Path("./scenarios/runs/primary_depletion/results/primary_depletion.zarr")
    )
    stream = bores.StateStream(store=store)

    states = list(stream.replay(steps=lambda s: s % 30 == 0))
    return bores, itertools, np, states


@app.cell
def _(bores, itertools, np, states):
    analyst = bores.ModelAnalyst(states)

    sweep_efficiency_history = analyst.sweep_efficiency_history(
        interval=1,
        from_step=1,
        displacing_phase="water",
    )
    production_rate_history = analyst.instantaneous_rates_history(
        interval=1,
        from_step=1,
        rate_type="production",
    )

    oil_saturation_history = []
    water_saturation_history = []
    gas_saturation_history = []
    avg_pressure_history = []

    oil_water_capillary_pressure_history = []
    gas_oil_capillary_pressure_history = []
    krw_history = []
    kro_history = []
    krg_history = []
    krw_saturation_history = []
    kro_saturation_history = []
    krg_saturation_history = []
    oil_relative_mobility_history = []
    oil_effective_viscosity_history = []
    oil_effective_density_history = []

    volumetric_sweep_efficiency_history = []
    displacement_efficiency_history = []
    recovery_efficiency_history = []

    water_cut_history = []
    gor_history = []

    for state in itertools.islice(states, 0, None, 1):
        model = state.model
        time_step = state.step
        fluid_properties = model.fluid_properties
        avg_oil_sat = np.mean(fluid_properties.oil_saturation_grid)
        avg_water_sat = np.mean(fluid_properties.water_saturation_grid)
        avg_gas_sat = np.mean(fluid_properties.gas_saturation_grid)
        avg_pressure = np.mean(fluid_properties.pressure_grid)
        avg_viscosity = np.mean(fluid_properties.oil_effective_viscosity_grid)
        avg_density = np.mean(fluid_properties.oil_effective_density_grid)
        avg_oil_rel_mobility = np.mean(state.relative_mobilities.oil_relative_mobility)

        avg_pcow = np.mean(state.capillary_pressures.oil_water_capillary_pressure)
        avg_pcgo = np.mean(state.capillary_pressures.gas_oil_capillary_pressure)
        avg_krw = np.mean(state.relative_permeabilities.krw)
        avg_kro = np.mean(state.relative_permeabilities.kro)
        avg_krg = np.mean(state.relative_permeabilities.krg)

        oil_saturation_history.append((time_step, avg_oil_sat))
        water_saturation_history.append((time_step, avg_water_sat))
        gas_saturation_history.append((time_step, avg_gas_sat))
        avg_pressure_history.append((time_step, avg_pressure))

        oil_water_capillary_pressure_history.append((time_step, avg_pcow))
        gas_oil_capillary_pressure_history.append((time_step, avg_pcgo))
        krw_history.append((time_step, avg_krw))
        kro_history.append((time_step, avg_kro))
        krg_history.append((time_step, avg_krg))
        krg_saturation_history.append((avg_krg, avg_gas_sat))
        kro_saturation_history.append((avg_kro, avg_oil_sat))
        krw_saturation_history.append((avg_krw, avg_water_sat))
        oil_effective_viscosity_history.append((time_step, avg_viscosity))
        oil_effective_density_history.append((time_step, avg_density))
        oil_relative_mobility_history.append((time_step, avg_oil_rel_mobility))

    for time_step, result in sweep_efficiency_history:
        volumetric_sweep_efficiency_history.append(
            (time_step, result.volumetric_sweep_efficiency)
        )
        displacement_efficiency_history.append(
            (time_step, result.displacement_efficiency)
        )
        recovery_efficiency_history.append((time_step, result.recovery_efficiency))

    for time_step, result in production_rate_history:
        water_cut_history.append((time_step, result.water_cut))
        gor_history.append((time_step, result.gas_oil_ratio))
    return (
        analyst,
        avg_pressure_history,
        displacement_efficiency_history,
        gas_oil_capillary_pressure_history,
        gas_saturation_history,
        gor_history,
        krg_history,
        kro_history,
        krw_history,
        oil_effective_density_history,
        oil_effective_viscosity_history,
        oil_relative_mobility_history,
        oil_saturation_history,
        oil_water_capillary_pressure_history,
        recovery_efficiency_history,
        volumetric_sweep_efficiency_history,
        water_cut_history,
        water_saturation_history,
    )


@app.cell
def _(avg_pressure_history, bores, np):
    # Pressure
    pressure_fig = bores.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history)},
        title="Pressure Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Avg. Pressure (psia)",
        marker_sizes=6,
        show_markers=True,
        width=720,
        height=460,
    )
    pressure_fig.show()
    return


@app.cell
def _(
    bores,
    gas_saturation_history,
    np,
    oil_saturation_history,
    water_saturation_history,
):
    # Saturation
    saturation_fig = bores.make_series_plot(
        data={
            "Avg. Water Saturation": np.array(water_saturation_history),
            "Avg. Oil Saturation": np.array(oil_saturation_history),
            "Avg. Gas Saturation": np.array(gas_saturation_history),
        },
        title="Saturation Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Saturation",
        marker_sizes=6,
        width=720,
        height=460,
    )
    saturation_fig.show()
    return


@app.cell
def _(
    bores,
    gas_oil_capillary_pressure_history,
    np,
    oil_water_capillary_pressure_history,
):
    # Capillary Pressure
    capillary_pressure_fig = bores.make_series_plot(
        data={
            "Pcow": np.array(oil_water_capillary_pressure_history),
            "Pcgo": np.array(gas_oil_capillary_pressure_history),
        },
        title="Capillary Pressure Analysis (Case 2)",
        x_label="Time Step",
        y_label="Avg. Capillary Pressure",
        marker_sizes=6,
        width=720,
        height=460,
    )
    capillary_pressure_fig.show()
    return


@app.cell
def _(bores, krg_history, kro_history, krw_history, np):
    # Rel Perm
    relperm_fig = bores.make_series_plot(
        data={
            "Krw": np.array(krw_history),
            "Kro": np.array(kro_history),
            "Krg": np.array(krg_history),
        },
        title="RelPerm Analysis (Case 2)",
        x_label="Time Step",
        y_label="Avg. Relative Permeability",
        marker_sizes=6,
        width=720,
        height=460,
    )
    relperm_fig.show()
    return


@app.cell
def _(
    bores,
    np,
    oil_effective_density_history,
    oil_effective_viscosity_history,
):
    # Oil Effective Density
    oil_effective_density_fig = bores.make_series_plot(
        data={
            "Oil Effective Density": np.array(oil_effective_density_history),
        },
        title="Oil Density Analysis",
        x_label="Time Step",
        y_label="Avg. Oil Effective Density (lbm/ft³)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    # Oil Effective Viscosity
    oil_effective_viscosity_fig = bores.make_series_plot(
        data={
            "Oil Effective Viscosity": np.array(oil_effective_viscosity_history),
        },
        title="Oil Viscosity Analysis",
        x_label="Time Step",
        y_label="Avg. Oil Effective Viscosity (cP)",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="brown",
    )

    effective_density_viscosity_fig = bores.merge_plots(
        oil_effective_density_fig,
        oil_effective_viscosity_fig,
        cols=2,
        title="Oil Effective Density & Viscosity Analysis  (CASE 2)",
    )
    effective_density_viscosity_fig.show()
    return


@app.cell
def _(bores, np, oil_relative_mobility_history):
    # Oil Relative Mobility
    oil_relative_mobility_fig = bores.make_series_plot(
        data={
            "Oil Relative Mobility": np.array(oil_relative_mobility_history),
        },
        title="Oil Mobility Analysis (Case 2)",
        x_label="Time Step",
        y_label="Avg. Oil Relative Mobility (cP⁻¹)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    oil_relative_mobility_fig.show()
    return


@app.cell
def _(analyst, bores, np):
    # Production & Injection
    oil_production_history = analyst.oil_production_history(
        interval=1, cumulative=False, from_step=1
    )
    oil_production_fig = bores.make_series_plot(
        data={
            "Oil Production": np.array(list(oil_production_history)),
        },
        title="Oil Production Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    oil_production_fig.show()
    return


@app.cell
def _(analyst, bores, np):
    water_production_history = analyst.water_production_history(
        interval=1, cumulative=False, from_step=1
    )
    water_production_fig = bores.make_series_plot(
        data={
            "Water Production": np.array(list(water_production_history)),
        },
        title="Water Production Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    water_production_fig.show()
    return


@app.cell
def _(analyst, bores, np):
    gas_production_history = analyst.gas_production_history(
        interval=1, cumulative=False, from_step=1
    )
    gas_production_fig = bores.make_series_plot(
        data={
            "Gas Production": np.array(list(gas_production_history)),
        },
        title="Gas Production Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Production (SCF)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_production_fig.show()
    return


@app.cell
def _(analyst, bores, np):
    # Cumulative production & injection
    cumulative_oil_production_history = analyst.oil_production_history(
        interval=1, cumulative=True, from_step=1
    )

    cumulative_oil_production_fig = bores.make_series_plot(
        data={
            "Cumulative Oil Production": np.array(
                list(cumulative_oil_production_history)
            ),
        },
        title="Cumulative Oil Production Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="orange",
    )
    cumulative_oil_production_fig.show()
    return


@app.cell
def _(analyst, bores, np):
    cumulative_water_production_history = analyst.water_production_history(
        interval=1, cumulative=True, from_step=1
    )

    cumulative_water_production_fig = bores.make_series_plot(
        data={
            "Cumulative Water Production": np.array(
                list(cumulative_water_production_history)
            ),
        },
        title="Cumulative Water Production Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="blue",
    )

    cumulative_water_production_fig.show()
    return


@app.cell
def _(analyst, bores, np):
    cumulative_gas_production_history = analyst.gas_production_history(
        interval=1, cumulative=True, from_step=1
    )

    cumulative_gas_production_fig = bores.make_series_plot(
        data={
            "Cumulative Gas Production": np.array(
                list(cumulative_gas_production_history)
            ),
        },
        title="Cumulative Gas Production Analysis (CASE 2)",
        x_label="Time Step",
        y_label="Production (SCF)",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="green",
    )

    cumulative_gas_production_fig.show()
    return


@app.cell
def _(bores, gor_history, np, water_cut_history):
    water_cut_fig = bores.make_series_plot(
        data={
            "Water Cut (WOR)": np.array(water_cut_history),
        },
        title="Water Cut Analysis",
        x_label="Time Step",
        y_label="Water Cut",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gor_fig = bores.make_series_plot(
        data={
            "Gas-Oil Ratio (GOR)": np.array(gor_history),
        },
        title="Gas-Oil Ratio Analysis",
        x_label="Time Step",
        y_label="Gas-Oil Ratio",
        marker_sizes=6,
        width=720,
        height=460,
    )

    fluid_cut_plots = bores.merge_plots(
        water_cut_fig, gor_fig, cols=2, title="Fluid Cut Analysis (CASE 2)"
    )
    fluid_cut_plots.show()
    return


@app.cell
def _(analyst, bores, np):
    # Reserves
    oil_in_place_history = analyst.oil_in_place_history(interval=1, from_step=1)
    gas_in_place_history = analyst.gas_in_place_history(interval=1, from_step=1)
    water_in_place_history = analyst.water_in_place_history(interval=1, from_step=1)

    oil_water_reserves_fig = bores.make_series_plot(
        data={
            "Oil In Place": np.array(list(oil_in_place_history)),
            "Water In Place": np.array(list(water_in_place_history)),
        },
        title="Oil & Water Reserves Analysis",
        x_label="Time Step",
        y_label="OIP/WIP (STB)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_reserve_fig = bores.make_series_plot(
        data={
            "Gas In Place": np.array(list(gas_in_place_history)),
        },
        title="Gas Reserve Analysis",
        x_label="Time Step",
        y_label="GIP (SCF)",
        marker_sizes=6,
        width=720,
        height=460,
    )

    reserves_plots = bores.merge_plots(
        oil_water_reserves_fig,
        gas_reserve_fig,
        cols=2,
        title="Reserves Analysis (CASE 2)",
    )
    reserves_plots.show()
    return


@app.cell
def _(
    bores,
    displacement_efficiency_history,
    np,
    volumetric_sweep_efficiency_history,
):
    # Recovery
    displacement_efficiency_fig = bores.make_series_plot(
        data={
            "Displacement Efficiency": np.array(displacement_efficiency_history),
        },
        title="Displacement Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="green",
    )
    vol_sweep_efficiency_fig = bores.make_series_plot(
        data={
            "Vol. Sweep Efficiency": np.array(volumetric_sweep_efficiency_history),
        },
        title="Volumetric Sweep Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
    )

    sweep_efficiency_plots = bores.merge_plots(
        displacement_efficiency_fig,
        vol_sweep_efficiency_fig,
        cols=2,
        title="Sweep Efficiency Analysis (CASE 2)",
    )
    sweep_efficiency_plots.show()
    return


@app.cell
def _(analyst, bores, np, recovery_efficiency_history):
    recovery_efficiency_fig = bores.make_series_plot(
        data={
            "Recovery Efficiency": np.array(recovery_efficiency_history),
        },
        title="Recovery Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="orange",
    )

    oil_recovery_factor_history = analyst.oil_recovery_factor_history(
        interval=1, from_step=1
    )
    recovery_factor_fig = bores.make_series_plot(
        data={
            "Oil Recovery Factor": np.array(list(oil_recovery_factor_history)),
        },
        title="Recovery Factor Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
    )

    recovery_plots = bores.merge_plots(
        recovery_efficiency_fig,
        recovery_factor_fig,
        cols=2,
        title="Recovery Analysis (CASE 2)",
    )
    recovery_plots.show()
    return


@app.cell
def _(analyst, bores, np):
    recommended_model, results = analyst.recommend_decline_model(phase="oil")
    print("Recommended Decline Model: ", recommended_model)
    decline_curve = results[recommended_model]
    if error := decline_curve.error:
        print(error)
    else:
        production_forecast = analyst.forecast_production(
            decline_result=decline_curve, steps=500
        )
        production_forecast_fig = bores.make_series_plot(
            data={
                "Production Rate": np.array(production_forecast),
            },
            title="Production Forecast (CASE 2)",
            x_label="Time Step",
            y_label="Production Rate (STB/day)",
            marker_sizes=4,
        )
        production_forecast_fig.show()
    return


@app.cell
def _(analyst):
    mbe = analyst.material_balance_error()
    print(mbe.total_mbe)
    return


@app.cell
def _(bores):
    viz = bores.pyvista3d.DataVisualizer(bores.pyvista3d.PlotConfig(off_screen=True))
    return (viz,)


@app.cell
def _(bores, states, viz):
    wells = states[0].wells
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [*injector_locations, *producer_locations]
    well_names = [*injector_names, *producer_names]
    labels = bores.plotly3d.Labels()
    labels.add_well_labels(well_positions, well_names)

    shared_kwargs = dict(
        plot_type="cell_blocks",
        width=960,
        height=600,
        opacity=1,
        labels=labels,
        aspect_mode="data",
        # z_scale=3.0,
        # marker_size=3,
        show_wells=True,
        surface_count=20,
        show_surface_marker=True,
        show_perforations=True,
        # show_cell_outlines=True,
        # show_edges=True,
        # x_slice=(10, 20),
        # z_slice=(0, 5),
        # isomin=0.6,
        # cmin=0,
        # cmax=1,
    )

    property = "pressure"
    figures = []
    timesteps = [14]
    for timestep in timesteps:
        figure = viz.make_plot(
            states[timestep],
            property=property,
            # title=f"{property.strip('-').title()} Profile at Timestep {timestep}",
            **shared_kwargs,
        )
        figures.append(figure)

    if len(figures) > 1:
        plots = bores.merge_plots(*figures, cols=2, height=620)
        plots.show()
        # for figure in figures:
        #     figure.show()
    else:
        figures[0].show()
    return


@app.cell
def _(states, viz):
    viz.animate(
        states,
        property="oil-saturation",
        plot_type="cell_blocks",
        output_gif="oil_saturation.gif",
        frame_duration=250,
        # step_size=5,  # Every 5th timestep
        cmin=0.0,
        cmax=1.0,  # Fixed range for consistent coloring
        opacity=1.0,
    )
    return


if __name__ == "__main__":
    app.run()
