import numpy
from matplotlib import pyplot

import iemic
import pop

from omuse.units import units
from amuse.io.base import IoException

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


def compute_depth_index(iemic_state, number_of_workers=4):
    mask = iemic_state.t_grid.mask

    # We get the levels directly instead because of the way i-emic calculates them
    z = iemic_state.v_grid.z[0, 0, :]
    z = iemic.z_from_center(z)
    levels = -z[::-1]

    depth = numpy.zeros((pop.Nx, pop.Ny))

    pop_instance = pop.initialize_pop(
        levels, depth, mode=f"{pop.Nx}x{pop.Ny}x12", latmin=pop.latmin, latmax=pop.latmax, number_of_workers=number_of_workers
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


def compute_depth_index_from_mask(mask):
    levels = iemic.depth_levels(13) * 5000 | units.m

    depth = numpy.zeros((pop.Nx, pop.Ny), dtype=int)
    for k in range(mask.shape[2]):
        for j in range(mask.shape[1]):
            for i in range(mask.shape[0]):
                if not depth[i, j+1]:
                    if mask[i, j, k] == 0:
                        depth[i, j+1] = 12 - k

    return levels, depth


def initialize_pop(number_of_workers=6, iemic_state=None, iemic_mask=None):
    if not iemic_state:
        iemic_state = iemic.read_iemic_state_with_units(state_name)

    if iemic_mask is not None:
        levels, depth = compute_depth_index_from_mask(iemic_mask)
    else:
        levels, depth = compute_depth_index(iemic_state)

    pop_instance = pop.initialize_pop(
        levels, depth, mode=f"{pop.Nx}x{pop.Ny}x12", number_of_workers=number_of_workers, latmin=pop.latmin, latmax=pop.latmax
    )

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

    pop_instance.parameters.reinit_gradp = True
    pop_instance.parameters.reinit_rho = True

    return pop_instance


def amoc(pop_instance):
    try:
        pop_amoc_state = pop.read_pop_state("amoc_state_" + pop_instance.mode)
    except IoException:
        Nx, Ny, Nz = pop_instance.mode.split('x')
        Ny = str(int(Ny) - 2)

        mask = iemic.read_global_mask(f"mkmask/amoc_{Nx}x{Ny}x{Nz}.mask")

        amoc_pop_instance = initialize_pop(iemic_mask=mask)

        assert amoc_pop_instance.mode == pop_instance.mode

        pop.save_pop_state(amoc_pop_instance, "amoc_state_" + pop_instance.mode)
        amoc_pop_instance.stop()

        pop_amoc_state = pop.read_pop_state("amoc_state_" + pop_instance.mode)

    amoc_depth = pop_amoc_state.elements.depth.value_in(units.km)
    depth = pop_instance.elements.depth.value_in(units.km)

    yvel = pop_instance.nodes3d.yvel.copy()
    for i in range(yvel.shape[0]):
        for j in range(yvel.shape[1]):
            if depth[i, j] > amoc_depth[i, j]:
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

    mask = [numpy.max(pop_amoc_state.elements.depth.value_in(units.km), axis=0) < zi for zi in z.value_in(units.km)]
    mask = numpy.array(mask).T
    mask = mask[yi, :]

    val = numpy.ma.array(psim, mask=mask)

    pyplot.figure(figsize=(7, 3.5))
    pyplot.contourf(y.value_in(units.deg), -z.value_in(units.m), val.T)
    pyplot.xticks([-20, 0, 20, 40, 60], ['20°S', '0°', '20°N', '40°N', '60°N'])
    pyplot.xlim(-30, 70)
    yticks = [0, -1000, -2000, -3000, -4000, -5000]
    pyplot.yticks(yticks, [str(int(abs(i))) for i in yticks])
    pyplot.ylabel('Depth (m)')
    pyplot.colorbar(label='Sv')
    pyplot.savefig(name)
    pyplot.close()
