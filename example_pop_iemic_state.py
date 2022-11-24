import os
import numpy

from omuse.units import units

from omuse.io import read_set_from_file

from iemic import (initialize_global_iemic, depth_array_from_mask,
                   depth_levels, get_surface_forcings, get_equilibrium_state,
                   z_from_cellcenterz, get_surface_grid, get_grid_with_units)

from pop import initialize_pop, long_evolve, plot_forcings_and_depth, evolve_test

from amuse.ext.grid_remappers import bilinear_2D_remapper, nearest_2D_remapper

from functools import partial

bilinear_2D_remapper = partial(bilinear_2D_remapper, check_inside=False)
bilinear_2D_remapper_3D = partial(bilinear_2D_remapper,
                                  check_inside=False,
                                  do_slices=True)
nearest_2D_remapper = partial(nearest_2D_remapper, check_inside=False)


def simple_upscale(x, fx, fy):
    # scale 2d x with integer factors
    return numpy.kron(x, numpy.ones((fx, fy)))


def reset_pop_state_from_iemic_state(pop, surface_state, iemic_state):

    # quick fix
    state = iemic_state[:, :, ::-1].copy()
    state.z = -state.z
    state.collection_attributes.axes_names = ["lon", "lat", "z"]

    # note formally there is an offset between node and element vars on the iemic grid too
    # this not accounted for: solve by making a copy and offsetting lat and lon

    channel1 = surface_state.new_remapping_channel_to(pop.elements,
                                                      bilinear_2D_remapper)
    channel1.copy_attributes(["ssh"], target_names=["ssh"])

    channel2 = surface_state.new_remapping_channel_to(pop.nodes,
                                                      bilinear_2D_remapper)
    channel2.copy_attributes(["uvel_barotropic", "vvel_barotropic"],
                             target_names=["vx_barotropic", "vy_barotropic"])

    channel3 = state.new_remapping_channel_to(pop.nodes3d,
                                              bilinear_2D_remapper_3D)
    channel3.copy_attributes(["u_velocity", "v_velocity"],
                             target_names=["xvel", "yvel"])

    channel4 = state.new_remapping_channel_to(pop.elements3d,
                                              bilinear_2D_remapper_3D)
    channel4.copy_attributes(["salinity", "temperature"])


def initialize_pop_with_iemic_setup(pop_number_of_workers=8):

    Nx = 96
    Ny = 40
    Nz = 12

    iemic_forcings_file = "global_forcings_96x38x12.amuse"
    iemic_state_file = "global_state_96x38x12.amuse"

    surface_forcings = None
    iemic_grid = None
    if os.path.isfile(iemic_forcings_file):
        surface_forcings = read_set_from_file(iemic_forcings_file)
    if os.path.isfile(iemic_state_file):
        iemic_grid = read_set_from_file(iemic_state_file)

    iemic = initialize_global_iemic(number_of_workers=1)

    stretch_factor = iemic.parameters.Ocean__THCM__Grid_Stretching_qz
    Hdim = iemic.parameters.Ocean__THCM__Depth_hdim | units.m
    # We get the levels directly instead because of the way i-emic calculates them
    levels = z_from_cellcenterz(iemic.grid[0, 0, :].z)
    levels = -levels[::-1]

    if surface_forcings is None or iemic_grid is None:

        if surface_forcings is None:
            surface_forcings = get_surface_forcings(iemic, iemic_forcings_file)
        if iemic_grid is None:
            iemic_grid = get_equilibrium_state(iemic, iemic_state_file)

    iemic.stop()

    iemic_state = get_grid_with_units(iemic_grid)
    surface_state = get_surface_grid(iemic_state)

    # temporary fixes for AMUSE issue #856
    surface_forcings.collection_attributes.axes_names = ["lon", "lat"]
    surface_state.collection_attributes.axes_names = ["lon", "lat"]
    iemic_state.collection_attributes.axes_names = ["lon", "lat", "z"]

    mask = iemic_grid.mask

    depth = numpy.zeros((Nx, Ny))
    depth[:,
          1:-1] = depth_array_from_mask(mask)  # convert to (index) depth array

    levels = depth_levels(Nz + 1, stretch_factor=stretch_factor) * Hdim

    # no interpolation for bathymetry
    depth = simple_upscale(depth, 1, 3)

    pop = initialize_pop(levels,
                         depth,
                         mode="96x120x12",
                         number_of_workers=pop_number_of_workers
                         )  # , latmin=latmin, latmax=latmax)

    pop.parameters.reinit_gradp = True
    pop.parameters.reinit_rho = True

    channel = surface_forcings.new_remapping_channel_to(
        pop.forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tau_x", "tau_y"])

    channel = surface_forcings.new_remapping_channel_to(
        pop.element_forcings, bilinear_2D_remapper)
    channel.copy_attributes(["tatm", "emip"],
                            target_names=["restoring_temp", "restoring_salt"])

    plot_forcings_and_depth(pop)

    reset_pop_state_from_iemic_state(pop, surface_state, iemic_state)

    return pop


if __name__ == "__main__":
    pop = initialize_pop_with_iemic_setup()
    evolve_test(pop)
    # long_evolve(pop, tend=5000 | units.yr, dt=1000. | units.day)
    plot_forcings_and_depth(pop)
