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
        store=Path.cwd() / "scenarios/runs/stabilization/results/stabilization.zarr"
    )
    stream = bores.StateStream(store=store)

    states = list(stream.replay())
    return bores, itertools, np, states


@app.cell
def _(bores, itertools, np, states):
    analyst = bores.ModelAnalyst(states)

    oil_saturation_history = []
    water_saturation_history = []
    gas_saturation_history = []
    avg_pressure_history = []
    oil_relative_mobility_history = []
    oil_effective_viscosity_history = []
    oil_effective_density_history = []

    oil_water_capillary_pressure_history = []
    gas_oil_capillary_pressure_history = []
    krw_history = []
    kro_history = []
    krg_history = []
    krw_saturation_history = []
    kro_saturation_history = []
    krg_saturation_history = []

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
    return (
        analyst,
        avg_pressure_history,
        gas_oil_capillary_pressure_history,
        gas_saturation_history,
        krg_history,
        krg_saturation_history,
        kro_history,
        kro_saturation_history,
        krw_history,
        krw_saturation_history,
        oil_effective_density_history,
        oil_effective_viscosity_history,
        oil_relative_mobility_history,
        oil_saturation_history,
        oil_water_capillary_pressure_history,
        water_saturation_history,
    )


@app.cell
def _(analyst):
    mbe = analyst.material_balance_error()
    print(mbe.total_mbe)
    return


@app.cell
def _(avg_pressure_history, bores, np):
    # Pressure
    pressure_fig = bores.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history)},
        title="Pressure Stability Analysis (Case 1)",
        x_label="Time Step",
        y_label="Avg. Pressure (psia)",
        marker_sizes=6,
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
            "Avg. Oil Saturation": np.array(oil_saturation_history),
            "Avg. Water Saturation": np.array(water_saturation_history),
            "Avg. Gas Saturation": np.array(gas_saturation_history),
        },
        title="Saturation Stability Analysis (Case 1)",
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
        title="Capillary Pressure Stability Analysis (Case 1)",
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
        title="RelPerm Stability Analysis (Case 1)",
        x_label="Time Step",
        y_label="Avg. Relative Permeability",
        marker_sizes=6,
        width=720,
        height=460,
    )
    relperm_fig.show()
    return


@app.cell
def _(bores, krw_saturation_history, np):
    # RelPerm-Saturation
    water_relperm_saturation_fig = bores.make_series_plot(
        data={
            "Krw/Sw": np.array(krw_saturation_history),
        },
        title="Water RelPerm-Saturation Stability Analysis (Case 1)",
        x_label="Avg. Water Saturation",
        y_label="Avg. Relative Permeability",
        marker_sizes=6,
        width=720,
        height=460,
    )
    water_relperm_saturation_fig.show()
    return


@app.cell
def _(bores, kro_saturation_history, np):
    oil_relperm_saturation_fig = bores.make_series_plot(
        data={
            "Kro/So": np.array(kro_saturation_history),
        },
        title="Oil RelPerm-Saturation Stability Analysis (Case 1)",
        x_label="Avg. Oil Saturation",
        y_label="Avg. Relative Permeability",
        marker_sizes=6,
        width=720,
        height=460,
    )
    oil_relperm_saturation_fig.show()
    return


@app.cell
def _(bores, krg_saturation_history, np):
    gas_relperm_saturation_fig = bores.make_series_plot(
        data={
            "Krg/Sg": np.array(krg_saturation_history),
        },
        title="Gas RelPerm-Saturation Stability Analysis (Case 1)",
        x_label="Avg. Gas Saturation",
        y_label="Avg. Relative Permeability",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_relperm_saturation_fig.show()
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
        title="Oil Effective Density & Viscosity Analysis  (CASE 3)",
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
        title="Oil Mobility Analysis (CASE 3)",
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
    oil_in_place_history = analyst.oil_in_place_history(interval=1, from_step=1)
    gas_in_place_history = analyst.gas_in_place_history(interval=1, from_step=1)
    water_in_place_history = analyst.water_in_place_history(interval=1, from_step=1)

    # Reserves
    oil_water_reserves_fig = bores.make_series_plot(
        data={
            "Water In Place": np.array(list(water_in_place_history)),
            "Oil In Place": np.array(list(oil_in_place_history)),
        },
        title="Oil & Water Reserves Stability Analysis (Case 1)",
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
        title="Gas Reserve Stability Analysis (Case 1)",
        x_label="Time Step",
        y_label="GIP (SCF)",
        marker_sizes=6,
        line_colors="green",
        width=720,
        height=460,
    )
    reserves_plot = bores.merge_plots(
        oil_water_reserves_fig,
        gas_reserve_fig,
        cols=2,
        title="Reserves Stability Analysis (Case 1)",
    )
    reserves_plot.show()
    return


@app.cell
def _(bores):
    viz = bores.plotly3d.DataVisualizer()
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
        plot_type="isosurface",
        width=960,
        height=460,
        # opacity=0.67,
        labels=labels,
        aspect_mode="data",
        z_scale=3,
        marker_size=4,
        show_wells=True,
        show_surface_marker=True,
        show_perforations=True,
        # isomin=0.05
    )

    property = "z"
    figures = []
    timesteps = [10]
    for timestep in timesteps:
        figure = viz.make_plot(
            states[timestep],
            property=property,
            title=f"{property.strip('-').title()} Profile at Timestep {timestep}",
            **shared_kwargs,
        )
        figures.append(figure)

    if len(figures) > 1:
        # plots = bores.merge_plots(*figures, cols=2)
        # plots.show()
        for figure in figures:
            figure.show()
    else:
        figures[0].show()
    return


if __name__ == "__main__":
    app.run()
