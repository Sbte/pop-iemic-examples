import numpy

import iemic
import pop
import utils

from omuse.units import units

from amuse.ext.grid_remappers import bilinear_2D_remapper

from functools import partial

bilinear_2D_remapper = partial(bilinear_2D_remapper, check_inside=False, x_periodic=True)
bilinear_2D_remapper_3D = partial(bilinear_2D_remapper, check_inside=False, do_slices=True, x_periodic=True)


# state_name = 'global_state'
state_name = 'idealized_120x54x12'


def reset_pop_state_from_iemic_state(pop_instance, iemic_state):

    # quick fix
    v_state = iemic_state.v_grid[:, :, ::-1].copy()
    v_state.z = -v_state.z

    t_state = iemic_state.t_grid[:, :, ::-1].copy()
    t_state.z = -t_state.z

    ssh = iemic.get_ssh(iemic_state)
    channel1 = ssh.new_remapping_channel_to(pop_instance.elements, bilinear_2D_remapper)
    channel1.copy_attributes(["ssh"])
    channel1.copy_attributes(["ssh"], target_names=["ssh_old"])
    channel1.copy_attributes(["ssh"], target_names=["ssh_guess"])

    barotropic_velocities = iemic.get_barotropic_velocities(iemic_state)
    channel2 = barotropic_velocities.new_remapping_channel_to(pop_instance.nodes, bilinear_2D_remapper)
    channel2.copy_attributes(["uvel_barotropic", "vvel_barotropic"], target_names=["vx_barotropic", "vy_barotropic"])
    channel2.copy_attributes(["uvel_barotropic", "vvel_barotropic"], target_names=["vx_barotropic_old", "vy_barotropic_old"])

    channel3 = v_state.new_remapping_channel_to(pop_instance.nodes3d, bilinear_2D_remapper_3D)
    channel3.copy_attributes(["u_velocity", "v_velocity"], target_names=["xvel", "yvel"])
    channel3.copy_attributes(["u_velocity", "v_velocity"], target_names=["xvel_old", "yvel_old"])

    channel4 = t_state.new_remapping_channel_to(pop_instance.elements3d, bilinear_2D_remapper_3D)
    channel4.copy_attributes(["salinity", "temperature"])
    channel4.copy_attributes(["salinity", "temperature"], target_names=["salinity_old", "temperature_old"])


def reset_pop_forcing_from_iemic_state(pop_instance, iemic_state):
    channel = iemic_state.surface_v_grid.new_remapping_channel_to(pop_instance.forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tau_x", "tau_y"], target_names=["tau_x", "tau_y"])

    channel = iemic_state.surface_t_grid.new_remapping_channel_to(pop_instance.element_forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tatm", "emip"], target_names=["restoring_temp", "restoring_salt"])


def compute_depth_index(iemic_state, Nx, Ny, number_of_workers=4):
    mask = iemic_state.t_grid.mask

    # We get the levels directly instead because of the way i-emic calculates them
    z = iemic_state.v_grid.z[0, 0, :]
    z = iemic.z_from_center(z)
    levels = -z[::-1]

    depth = numpy.zeros((Nx, Ny))

    pop_instance = pop.initialize_pop(
        levels, depth, mode=f"{Nx}x{Ny}x12", number_of_workers=number_of_workers
    )

    iemic_surface = iemic_state.t_grid[:, :, -1]
    source_depth = iemic_surface.empty_copy()
    channel = iemic_surface.new_channel_to(source_depth)
    channel.copy_attributes(["lon", "lat"])

    source_depth.set_values_in_store(None, ["depth_index"], [iemic.depth_array_from_mask(mask)])

    pop_surface = pop_instance.elements
    target_depth = pop_surface.empty_copy()
    channel = pop_surface.new_channel_to(target_depth)
    channel.copy_attributes(["lon", "lat"])

    channel = source_depth.new_remapping_channel_to(target_depth, bilinear_2D_remapper)
    channel.copy_attributes(["depth_index"])

    depth = numpy.round(target_depth.depth_index)
    depth[:, 0] = 0
    depth[:, -1] = 0

    pop_instance.stop()

    return levels, depth


def initialize_pop(number_of_workers=6, iemic_state=None, iemic_mask=None):
    if not iemic_state:
        iemic_state = iemic.read_iemic_state_with_units(state_name)

    if iemic_mask is not None:
        levels, depth = utils.compute_depth_index_from_mask(iemic_mask)
    else:
        levels, depth = compute_depth_index(iemic_state, 120, 56)

    Nx = depth.shape[0]
    Ny = depth.shape[1]

    pop_instance = pop.initialize_pop(
        levels, depth, mode=f"{Nx}x{Ny}x12", number_of_workers=number_of_workers)

    reset_pop_forcing_from_iemic_state(pop_instance, iemic_state)

    pop.plot_forcings_and_depth(pop_instance)

    lat_diff = numpy.min(abs(pop_instance.elements.lat - iemic_state.t_grid.lat[0, 0, 0]).value_in(units.deg))
    print('Latitude difference', lat_diff)
    print(pop_instance.elements.lat[0, :].value_in(units.deg))
    print(iemic_state.t_grid.lat[0, :, 0].value_in(units.deg))
    assert lat_diff < 1e-2

    lon_diff = numpy.min(abs(pop_instance.elements.lon - iemic_state.t_grid.lon[0, 0, 0]).value_in(units.deg))
    print('Longitude difference', lon_diff)
    print(pop_instance.elements.lon[:, 0].value_in(units.deg))
    print(iemic_state.t_grid.lon[:, 0, 0].value_in(units.deg))
    assert lon_diff < 1e-2

    return pop_instance


def initialize_pop_with_iemic_setup(number_of_workers=6, state_name=state_name, iemic_mask=None):
    iemic_state = iemic.read_iemic_state_with_units(state_name)

    # iemic.plot_barotropic_streamfunction(iemic_state, "iemic_bstream.eps")
    # iemic.plot_u_velocity(iemic_state.v_grid, "iemic_u_velocity.eps")
    # iemic.plot_v_velocity(iemic_state.v_grid, "iemic_v_velocity.eps")
    # iemic.plot_surface_pressure(iemic_state.t_grid, "iemic_pressure.eps")
    # iemic.plot_surface_salinity(iemic_state.t_grid, "iemic_salinity.eps")
    # iemic.plot_surface_temperature(iemic_state.t_grid, "iemic_temperature.eps")
    # iemic.plot_streamplot(iemic_state, "iemic_streamplot.eps")

    pop_instance = initialize_pop(number_of_workers, iemic_state, iemic_mask)

    print("before reset")

    reset_pop_state_from_iemic_state(pop_instance, iemic_state)

    print("after reset")

    pop_instance.parameters.pressure_correction = True
    pop_instance.parameters.reinit_gradp = True
    pop_instance.parameters.reinit_rho = True
    pop_instance.parameters.ts_option = "amuse"

    return pop_instance


def initialize_pop_with_pop_setup(number_of_workers=6, label="latest",
                                  snapdir="snapshots", iemic_mask=None):
    pop_instance = initialize_pop(number_of_workers, iemic_mask=iemic_mask)

    print("before reset")

    pop.reset_pop_state_from_pop_state(pop_instance, label, snapdir)

    print("after reset")

    return pop_instance
