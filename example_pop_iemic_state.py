import os
import numpy

from omuse.units import units

from omuse.io import read_set_from_file

from iemic import (initialize_global_iemic, depth_array_from_mask,
                   depth_levels, get_surface_forcings, get_equilibrium_state,
                   z_from_center, get_surface_grid, get_grid_with_units)

from pop import initialize_pop, long_evolve, plot_forcings_and_depth, evolve_test
from pop import plot_ssh, plot_sst
from pop import plot_barotropic_streamfunction, plot_overturning_streamfunction
from pop import plot_salinity, plot_temperature

# from iemic import plot_velocity
from pop import plot_velocity

from amuse.ext.grid_remappers import bilinear_2D_remapper, nearest_2D_remapper

from functools import partial

bilinear_2D_remapper = partial(bilinear_2D_remapper, check_inside=False, ignore_zero=False)
nearest_2D_remapper = partial(nearest_2D_remapper, check_inside=False)
bilinear_2D_remapper_3D = partial(bilinear_2D_remapper,
                                  check_inside=False,
                                  do_slices=True)
bilinear_2D_remapper_3D_2 = partial(bilinear_2D_remapper,
                                    check_inside=False,
                                    do_slices=True, ignore_zero=True)


def simple_upscale(x, fx, fy):
    # scale 2d x with integer factors
    return numpy.kron(x, numpy.ones((fx, fy)))


def bilinear_upscale(depth, fx, fy):
    repeat = numpy.zeros((depth.shape[0]+1, depth.shape[1]+1))
    repeat[:-1, :-1] = depth
    # repeat[-1, :-1] = depth[0, :]
    # repeat[:-1, -1] = depth[:, 0]
    out = numpy.kron(depth, numpy.ones((fx, fy)))

    import sys
    numpy.set_printoptions(threshold=sys.maxsize)
    print(out)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            wx = i / fx - i // fx
            wy = j / fy - j // fy
            out[i, j] = (1 - wx) * (1 - wy) * repeat[i // fx, j // fy] \
                + wx * (1 - wy) * repeat[i // fx + 1, j // fy] \
                + (1 - wx) * wy * repeat[i // fx, j // fy + 1] \
                + wx * wy * repeat[i // fx + 1, j // fy + 1]

            total_weights = 0
            if repeat[i // fx, j // fy] != 0:
                total_weights += (1 - wx) * (1 - wy)
            if repeat[i // fx + 1, j // fy] != 0:
                total_weights += wx * (1 - wy)
            if repeat[i // fx, j // fy + 1] != 0:
                total_weights += (1 - wx) * wy
            if repeat[i // fx + 1, j // fy + 1] != 0:
                total_weights += wx * wy
            if total_weights == 0:
                total_weights = 1000

            out[i, j] /= total_weights

        # 1 / (exp(dts * j / tau) + (y - x)^2)

    out = numpy.round(out)

    print(out)

    return out


def reset_pop_state_from_iemic_state(pop, surface_state, iemic_state):

    # quick fix
    state = iemic_state[:, :, ::-1].copy()
    state.z = -state.z
    # state.collection_attributes.axes_names = ["lon", "lat", "z"]

    # note formally there is an offset between node and element vars on the iemic grid too
    # this not accounted for: solve by making a copy and offsetting lat and lon

    channel1 = surface_state.new_remapping_channel_to(pop.elements,
                                                      bilinear_2D_remapper)
    channel1.copy_attributes(["ssh"], target_names=["ssh"])

    channel2 = surface_state.new_remapping_channel_to(pop.nodes,
                                                      bilinear_2D_remapper)
    channel2.copy_attributes(["uvel_barotropic", "vvel_barotropic"],
                             target_names=["vx_barotropic", "vy_barotropic"])

    pop.nodes3d.set_axes_names(state.get_axes_names())

    channel3 = state.new_remapping_channel_to(pop.nodes3d,
                                              bilinear_2D_remapper_3D_2)
    channel3.copy_attributes(["u_velocity", "v_velocity"],
                             target_names=["xvel", "yvel"])

    channel4 = state.new_remapping_channel_to(pop.elements3d,
                                              bilinear_2D_remapper_3D_2)
    channel4.copy_attributes(["salinity", "temperature"])


def compute_depth_index(iemic_state):
    mask = iemic_state.mask

    Nx = iemic_state.shape[0]
    Ny = iemic_state.shape[1] + 2
    depth = numpy.zeros((Nx, Ny))
    depth[:, 1:-1] = depth_array_from_mask(mask)  # convert to (index) depth array

    # levels = depth_levels(Nz + 1, stretch_factor=stretch_factor) * Hdim

    # no interpolation for bathymetry
    upscaled_depth = simple_upscale(depth, 1, 3)

    return upscaled_depth

    import sys
    numpy.set_printoptions(threshold=sys.maxsize, linewidth=500)
    print(upscaled_depth.astype(int))

    # We get the levels directly instead because of the way i-emic calculates them
    levels = z_from_center(iemic_state.z[0, 0, :])
    levels = -levels[::-1]

    pop = initialize_pop(levels,
                         upscaled_depth,
                         mode="96x120x12",
                         )  # , latmin=latmin, latmax=latmax)

    iemic_surface = iemic_state[:, :, -1]
    source_depth = iemic_surface.empty_copy()
    channel = iemic_surface.new_channel_to(source_depth)
    channel.copy_attributes(['lon', 'lat'])

    source_depth.set_values_in_store(None, ['depth_index'], [depth_array_from_mask(mask)])

    depth = numpy.round(source_depth.depth_index)
    print(depth.astype(int))

    pop_surface = pop.elements
    target_depth = pop_surface.empty_copy()
    channel = pop_surface.new_channel_to(target_depth)
    channel.copy_attributes(['lon', 'lat'])

    channel = source_depth.new_remapping_channel_to(target_depth, bilinear_2D_remapper)
    channel.copy_attributes(['depth_index'])

    depth = numpy.round(target_depth.depth_index)
    print(depth.astype(int))

    print(upscaled_depth.astype(int) - depth.astype(int))

    depth[:, 0] = 0
    depth[:, -1] = 0

    return depth


def initialize_pop_with_iemic_setup(pop_number_of_workers=8):

    iemic_forcings_file = "asdf.amuse"
    iemic_state_file = "global_state_96x38x12.amuse"

    surface_forcings = None
    iemic_grid = None

    if os.path.isfile(iemic_forcings_file):
        surface_forcings = read_set_from_file(iemic_forcings_file)
        surface_forcings.set_axes_names(['lon', 'lat'])
        print(surface_forcings._derived_attributes['position'].attribute_names)
    if os.path.isfile(iemic_state_file):
        iemic_grid = read_set_from_file(iemic_state_file)
        iemic_grid.set_axes_names(['lon', 'lat', 'z'])

    if surface_forcings is None or iemic_grid is None:
        iemic = initialize_global_iemic(number_of_workers=1)

        # stretch_factor = iemic.parameters.Ocean__THCM__Grid_Stretching_qz
        # Hdim = iemic.parameters.Ocean__THCM__Depth_hdim | units.m

        if surface_forcings is None:
            surface_forcings = get_surface_forcings(iemic, iemic_forcings_file)
        if iemic_grid is None:
            iemic_grid = get_equilibrium_state(iemic, iemic_state_file)

        iemic.stop()

    iemic_state = get_grid_with_units(iemic_grid)
    iemic_state.set_axes_names(['lon', 'lat', 'z'])
    surface_state = get_surface_grid(iemic_state)
    surface_state.set_axes_names(['lon', 'lat'])

    # plot_velocity(iemic_state)
    # return

    # We get the levels directly instead because of the way i-emic calculates them
    levels = z_from_center(iemic_state.z[0, 0, :])
    levels = -levels[::-1]

    depth = compute_depth_index(iemic_state)

    # # temporary fixes for AMUSE issue #856
    # surface_forcings.collection_attributes.axes_names = ["lon", "lat"]
    # surface_state.collection_attributes.axes_names = ["lon", "lat"]
    # iemic_state.collection_attributes.axes_names = ["lon", "lat", "z"]

    pop = initialize_pop(levels,
                         depth,
                         mode="96x120x12",
                         number_of_workers=pop_number_of_workers
                         )  # , latmin=latmin, latmax=latmax)

    # print(iemic_state.lat[0, :, 0])
    # print(pop.nodes3d.lat[0, :, 0])

    # return

    pop.parameters.reinit_gradp = True
    pop.parameters.reinit_rho = True

    print(dir(surface_forcings))
    print(type(surface_forcings))

    channel = surface_forcings.new_remapping_channel_to(
        pop.forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tau_x", "tau_y"])

    channel = surface_forcings.new_remapping_channel_to(
        pop.element_forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tatm", "emip"],
                            target_names=["restoring_temp", "restoring_salt"])

    plot_forcings_and_depth(pop)

    print('before reset')

    reset_pop_state_from_iemic_state(pop, surface_state, iemic_state)

    print('after reset')

    plot_velocity(pop)
    return

    return pop


if __name__ == "__main__":
    pop = initialize_pop_with_iemic_setup()

    # plot_barotropic_streamfunction(pop, "bstream-pop-iemic.eps")
    # plot_overturning_streamfunction(pop, "mstream-pop-iemic.eps")
    # plot_salinity(pop, "salinity-pop-iemic.eps")
    # plot_temperature(pop, "temperature-pop-iemic.eps")

    days = 10
    evolve_test(pop, days)
    plot_ssh(pop.elements)
    plot_sst(pop)
    # long_evolve(pop, tend=5000 | units.yr, dt=1000. | units.day)
    plot_forcings_and_depth(pop)

    # days = 0
    plot_barotropic_streamfunction(pop, "bstream-pop-iemic-" + str(days) + ".eps")
    plot_overturning_streamfunction(pop, "mstream-pop-iemic-" + str(days) + ".eps")
    plot_salinity(pop, "salinity-pop-iemic-" + str(days) + ".eps")
    plot_temperature(pop, "temperature-pop-iemic-" + str(days) + ".eps")
