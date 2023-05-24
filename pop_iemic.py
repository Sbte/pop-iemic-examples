import numpy
from matplotlib import pyplot

import bstream
import iemic
import pop

from omuse.units import units, constants
from amuse.io.base import IoException

from amuse.ext.grid_remappers import bilinear_2D_remapper, nearest_2D_remapper

from functools import partial

bilinear_2D_remapper = partial(bilinear_2D_remapper, check_inside=False)
nearest_2D_remapper = partial(nearest_2D_remapper, check_inside=False)
bilinear_2D_remapper_3D = partial(bilinear_2D_remapper, check_inside=False, do_slices=True)


def simple_upscale(x, fx, fy):
    # scale 2d x with integer factors
    return numpy.kron(x, numpy.ones((fx, fy)))


def reset_pop_state_from_iemic_state(pop_instance, iemic_state):

    # quick fix
    v_state = iemic_state.v_grid[:, :, ::-1].copy()
    v_state.z = -v_state.z

    t_state = iemic_state.t_grid[:, :, ::-1].copy()
    t_state.z = -t_state.z

    ssh = iemic.get_ssh(iemic_state)
    channel1 = ssh.new_remapping_channel_to(pop_instance.elements, bilinear_2D_remapper)
    channel1.copy_attributes(["ssh"])

    barotropic_velocities = iemic.get_barotropic_velocities(iemic_state)
    channel2 = barotropic_velocities.new_remapping_channel_to(pop_instance.nodes, bilinear_2D_remapper)
    channel2.copy_attributes(["uvel_barotropic", "vvel_barotropic"], target_names=["vx_barotropic", "vy_barotropic"])

    channel3 = v_state.new_remapping_channel_to(pop_instance.nodes3d, bilinear_2D_remapper_3D)
    channel3.copy_attributes(["u_velocity", "v_velocity"], target_names=["xvel", "yvel"])

    channel4 = t_state.new_remapping_channel_to(pop_instance.elements3d, bilinear_2D_remapper_3D)
    channel4.copy_attributes(["salinity", "temperature"])


def reset_pop_forcing_from_iemic_state(pop_instance, iemic_state):
    channel = iemic_state.surface_v_grid.new_remapping_channel_to(pop_instance.forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tau_x", "tau_y"], target_names=["tau_x", "tau_y"])

    channel = iemic_state.surface_t_grid.new_remapping_channel_to(pop_instance.element_forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tatm", "emip"], target_names=["restoring_temp", "restoring_salt"])


def compute_depth_index(iemic_state, number_of_workers=8):
    mask = iemic_state.t_grid.mask

    Nx = iemic_state.t_grid.shape[0]
    Ny = iemic_state.t_grid.shape[1] + 2
    depth = numpy.zeros((Nx, Ny))
    depth[:, 1:-1] = iemic.depth_array_from_mask(mask)  # convert to (index) depth array

    # levels = depth_levels(Nz + 1, stretch_factor=stretch_factor) * Hdim

    # We get the levels directly instead because of the way i-emic calculates them
    z = iemic_state.w_grid.z[0, 0, :]
    levels = numpy.zeros(len(z) + 1) * z[0]
    levels[1:] = -z[::-1]

    # no interpolation for bathymetry
    upscaled_depth = simple_upscale(depth, 1, 3)

    # return levels, upscaled_depth

    # import sys
    # numpy.set_printoptions(threshold=sys.maxsize, linewidth=500)
    # print(upscaled_depth.astype(int))

    pop_instance = pop.initialize_pop(levels, depth, mode=f"{pop.Nx}x{pop.Ny}x12", latmin=pop.latmin, latmax=pop.latmax, number_of_workers=number_of_workers)

    iemic_surface = iemic_state.t_grid[:, :, -1]
    source_depth = iemic_surface.empty_copy()
    channel = iemic_surface.new_channel_to(source_depth)
    channel.copy_attributes(["lon", "lat"])

    source_depth.set_values_in_store(None, ["depth_index"], [iemic.depth_array_from_mask(mask)])

    depth = numpy.round(source_depth.depth_index)

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


def initialize_pop(number_of_workers=8, iemic_state=None):
    if not iemic_state:
        iemic_state = iemic.read_iemic_state_with_units("idealized_120x54x12")

    levels, depth = compute_depth_index(iemic_state)

    pop_instance = pop.initialize_pop(
        levels, depth, mode=f"{pop.Nx}x{pop.Ny}x12", number_of_workers=number_of_workers, latmin=pop.latmin, latmax=pop.latmax
    )

    reset_pop_forcing_from_iemic_state(pop_instance, iemic_state)

    pop.plot_forcings_and_depth(pop_instance)

    return pop_instance


def initialize_pop_with_iemic_setup(number_of_workers=6):
    iemic_state = iemic.read_iemic_state_with_units("idealized_120x54x12")

    # iemic.plot_barotropic_streamfunction(iemic_state, "iemic_bstream.eps")
    # iemic.plot_u_velocity(iemic_state.v_grid, "iemic_u_velocity.eps")
    # iemic.plot_v_velocity(iemic_state.v_grid, "iemic_v_velocity.eps")
    # iemic.plot_surface_pressure(iemic_state.t_grid, "iemic_pressure.eps")
    # iemic.plot_surface_salinity(iemic_state.t_grid, "iemic_salinity.eps")
    # iemic.plot_surface_temperature(iemic_state.t_grid, "iemic_temperature.eps")
    # iemic.plot_streamplot(iemic_state, "iemic_streamplot.eps")

    levels, depth = compute_depth_index(iemic_state)

    pop_instance = pop.initialize_pop(
        levels, depth, mode=f"{pop.Nx}x{pop.Ny}x12", number_of_workers=number_of_workers, latmin=pop.latmin, latmax=pop.latmax
    )

    reset_pop_forcing_from_iemic_state(pop_instance, iemic_state)

    pop.plot_forcings_and_depth(pop_instance)

    print("before reset")

    reset_pop_state_from_iemic_state(pop_instance, iemic_state)

    print("after reset")

    pop_instance.parameters.reinit_gradp = True
    pop_instance.parameters.reinit_rho = True

    return pop_instance


def amoc(pop_instance):
    try:
        iemic_state = iemic.read_iemic_state_with_units("amoc_state_" + pop_instance.mode)
    except FileNotFoundError:
        iemic_instance = iemic.initialize_global_iemic()

        Nx = iemic_instance.parameters.Ocean__THCM__Global_Grid_Size_n
        Ny = iemic_instance.parameters.Ocean__THCM__Global_Grid_Size_m
        Nz = iemic_instance.parameters.Ocean__THCM__Global_Grid_Size_l

        iemic_instance.parameters.Ocean__THCM__Land_Mask = f"amoc_{Nx}x{Ny}x{Nz}.mask"

        iemic.save_iemic_state(iemic_instance, "amoc_state_" + pop_instance.mode)

        iemic_instance.stop()

        iemic_state = iemic.read_iemic_state_with_units("amoc_state_" + pop_instance.mode)

    try:
        pop_amoc_state = pop.read_pop_state("amoc_state_" + pop_instance.mode)
    except IoException:
        amoc_pop_instance = initialize_pop(iemic_state=iemic_state)
        pop.save_pop_state(amoc_pop_instance, "amoc_state_" + pop_instance.mode)
        amoc_pop_instance.stop()

        pop_amoc_state = pop.read_pop_state("amoc_state_" + pop_instance.mode)

    depth = pop_amoc_state.nodes.depth
    mask = depth.value_in(units.km) == 0

    yvel = pop_instance.nodes3d.yvel.copy()
    for i in range(yvel.shape[0]):
        for j in range(yvel.shape[1]):
            if mask[i, j]:
                yvel[i, j, :] = 0 | units.m / units.s

    pop_amoc_state.nodes3d.yvel = yvel

    y = pop_instance.nodes3d.lat[0, :, 0]
    yi = [i for i, v in enumerate(y) if v.value_in(units.deg) > -30]

    psim = pop.overturning_streamfunction(pop_amoc_state)
    return psim[yi, :]


def plot_amoc(pop_instance, name="amoc.eps"):
    psim = amoc(pop_instance)

    pop_amoc_state = pop.read_pop_state("amoc_state_" + pop_instance.mode)

    y = pop_instance.nodes3d.lat[0, :, 0]
    yi = [i for i, v in enumerate(y) if v.value_in(units.deg) > -30]
    y = y[yi]

    z = pop_amoc_state.nodes3d.z[0, 0, :]
    z = pop.z_from_center(z)

    mask = [numpy.max(pop_amoc_state.nodes.depth.value_in(units.km), axis=0) < zi for zi in z.value_in(units.km)]
    mask = numpy.array(mask).T
    mask = mask[yi, :]

    val = numpy.ma.array(psim, mask=mask)

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), -z.value_in(units.m), val.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()
