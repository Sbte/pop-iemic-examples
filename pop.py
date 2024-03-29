import os
import shutil
import numpy
import time
from matplotlib import colors, pyplot, colormaps, ticker

from omuse.community.pop.interface import POP
from omuse.units import units, constants

from amuse.io import write_set_to_file, read_set_from_file
from amuse.ext.grid_remappers import bilinear_2D_remapper

from functools import partial

import bstream

numpy.random.seed(123451)

bilinear_2D_remapper = partial(bilinear_2D_remapper, check_inside=False, x_periodic=True)
bilinear_2D_remapper_3D = partial(bilinear_2D_remapper, check_inside=False, do_slices=True, x_periodic=True)


def z_from_center(zc):
    z = numpy.zeros(len(zc) + 1) * zc[0]

    direction = 1
    if zc[0] <= zc[0] * 0:
        direction = -1

    for i, _zc in enumerate(zc[::direction]):
        half = _zc - z[i]
        z[i + 1] = z[i] + 2 * half

    return z[::direction]


def plot_forcings_and_depth(p, label="pop"):
    pyplot.figure()
    val = p.nodes.depth.value_in(units.km).T
    mask = val == 0
    val = numpy.ma.array(val, mask=mask)
    pyplot.imshow(val, origin="lower")
    cbar = pyplot.colorbar()
    cbar.set_label("depth (km)")
    pyplot.savefig(label + "_depth.png")
    pyplot.close()

    pyplot.figure()
    val = p.forcings.tau_x.value_in(units.Pa).T
    val = numpy.ma.array(val, mask=mask)
    pyplot.imshow(val, origin="lower")
    cbar = pyplot.colorbar()
    cbar.set_label("wind stress (Pa)")
    pyplot.savefig(label + "_taux.png")
    pyplot.close()

    pyplot.figure()
    val = p.element_forcings.restoring_temp.value_in(units.Celsius).T
    val = numpy.ma.array(val, mask=mask)
    pyplot.imshow(val, origin="lower")
    cbar = pyplot.colorbar()
    cbar.set_label("restoring T (C)")
    pyplot.savefig(label + "_restoring_temp.png")
    pyplot.close()

    pyplot.figure()
    val = p.element_forcings.restoring_salt.value_in(units.g / units.kg).T
    val = numpy.ma.array(val, mask=mask)
    pyplot.imshow(val, origin="lower")
    cbar = pyplot.colorbar()
    cbar.set_label("restoring salt (psu)")
    pyplot.savefig(label + "_restoring_salt.png")
    pyplot.close()


def initialize_pop(depth_levels, depth_array, mode=None, number_of_workers=6):
    print(f"initializing POP mode {mode} with {number_of_workers} workers")

    Nx = depth_array.shape[0]
    Ny = depth_array.shape[1]

    if Nx == 120:
        latmin = -84 | units.deg
        latmax = 84 | units.deg

        lonmin = 0 | units.deg
        lonmax = 360 | units.deg
    elif Nx == 240:
        latmin = -81.75 | units.deg
        latmax = 83.25 | units.deg

        lonmin = 0.75 | units.deg
        lonmax = 360.75 | units.deg

    if mode is None:
        mode = f"{Nx}x{Ny}x12"

    p = POP(
        number_of_workers=number_of_workers, mode=mode, redirection="none"
    )  # , job_scheduler="slurm") #, channel_type="sockets"

    dz = depth_levels[1:] - depth_levels[:-1]
    # print(f"dz: {dz}")

    p.parameters.topography_option = "amuse"
    p.parameters.depth_index = depth_array
    p.parameters.horiz_grid_option = "amuse"
    p.parameters.lonmin = lonmin
    p.parameters.lonmax = lonmax
    p.parameters.latmin = latmin
    p.parameters.latmax = latmax
    p.parameters.vert_grid_option = "amuse"
    p.parameters.vertical_layer_thicknesses = dz
    p.parameters.surface_heat_flux_forcing = "amuse"
    p.parameters.surface_freshwater_flux_forcing = "amuse"
    p.parameters.ts_option = "amuse"

    # print(p.nodes[0,0].lon.in_(units.deg))
    # print(p.nodes[0,0].lat.in_(units.deg))
    # print(p.nodes[-1,-1].lon.in_(units.deg))
    # print(p.nodes[-1,-1].lat.in_(units.deg))
    # print()
    # print(p.nodes[0,:].lat.in_(units.deg))

    # print()
    # print(p.elements[0,0].lon.in_(units.deg))
    # print(p.elements[0,0].lat.in_(units.deg))
    # print(p.elements[-1,-1].lon.in_(units.deg))
    # print(p.elements[-1,-1].lat.in_(units.deg))
    # print()
    # print(p.elements[0,:].lat.in_(units.deg))

    # print()
    # print((p.nodes[1,1].position-p.nodes[0,0].position).in_(units.deg))
    # print((p.elements[1,1:].position-p.elements[0,:-1].position).in_(units.deg))

    return p


def evolve(p, tend=10 | units.day, dt=1.0 | units.day):
    tnow = p.model_time
    tend = tnow + tend

    while tnow < tend - dt / 2:

        print("ret = ", p.evolve_model(tnow + dt), flush=True)
        tnow = p.model_time

        t = tnow.value_in(units.day)
        t = int(t)
        print("evolve to", t, flush=True)


def plot_masked_contour(x, y, value, unit, lims=None, levels=None):
    plot = pyplot.contourf(x, y, value, levels=levels)
    pyplot.close()

    ticks = None
    levels = None
    colormap = 'viridis'

    # Center the plot_levels if necessary
    if plot.levels[0] < 0 and plot.levels[-1] > 0:
        if lims is None:
            lims = (plot.levels[0], plot.levels[-1])

        levels = ticker.MaxNLocator(len(plot.levels) * 2).tick_values(*lims)
        lims = (levels[0], levels[-1])

        # Always center on the same color
        lim = max(-lims[0], lims[1])
        norm = colors.Normalize(-lim, lim)

        colormap = colormaps['RdBu_r']

        # Map both segments around zero to 0.5
        color_space = colormap(numpy.linspace(0, 0.5, len(levels) // 2))
        color_space = numpy.append(color_space, colormap(numpy.linspace(0.5, 1, len(levels) // 2)), axis=0)
        colormap = colors.LinearSegmentedColormap.from_list('map', color_space)

        # Fix the under and over colors to be the next segment color
        under = None
        if lim > -lims[0]:
            under = colormap((lims[0] + lim) / (2 * lim))

        over = None
        if lim > lims[1]:
            over = colormap((lims[1] + lim) / (2 * lim))

        colormap.set_extremes(under=under, over=over)

    pyplot.figure(figsize=(7, 3.5))

    pyplot.contourf(x, y, numpy.ma.array(value.data * 0, mask=~value.mask), cmap='binary')
    pyplot.contourf(x, y, value, cmap=colormap, norm=norm, levels=levels, extend='both')

    pyplot.colorbar(label=unit, ticks=ticks)


def plot_globe(p, value, unit, name, elements=False, lims=None):
    mask = p.elements.depth.value_in(units.km) == 0
    value = numpy.ma.array(value, mask=mask)

    if elements:
        x = p.elements3d.lon[:, 0, 0].value_in(units.deg)
        y = p.elements3d.lat[0, :, 0].value_in(units.deg)
    else:
        x = p.nodes3d.lon[:, 0, 0].value_in(units.deg)
        y = p.nodes3d.lat[0, :, 0].value_in(units.deg)

    for i in range(len(x)):
        if x[i] > 180:
            x[i] = x[i] - 360

    i = numpy.argsort(x)
    i = numpy.insert(i, 0, i[-1])

    value = value[i, :]
    x = x[i]
    x[0] -= 360

    plot_masked_contour(x, y, value.T, unit, lims=lims)

    pyplot.xticks([-180, -120, -60, 0, 60, 120, 180],
                  ['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'])
    pyplot.yticks([-60, -30, 0, 30, 60],
                  ['60°S', '30°S', '0°', '30°N', '60°N'])
    pyplot.ylim(y[1], y[-2])
    pyplot.savefig(name)
    pyplot.close()


def plot_ssh(p, name="ssh.eps"):
    ssh = p.elements.ssh.value_in(units.m)
    plot_globe(p, ssh, "m", name, elements=True)


def plot_grid(p):
    lon = p.nodes.lon.value_in(units.deg).flatten()
    lat = p.nodes.lat.value_in(units.deg).flatten()
    pyplot.figure()
    pyplot.plot(lon, lat, "r+")

    lon = p.elements.lon.value_in(units.deg).flatten()
    lat = p.elements.lat.value_in(units.deg).flatten()
    pyplot.plot(lon, lat, "g.")

    pyplot.savefig("grid.png")
    pyplot.close()


def plot_velocity(p, name="velocity.eps"):
    xvel = p.nodes3d.xvel[:, :, 0].value_in(units.m / units.s)
    plot_globe(p, xvel, "m/s", name)


def plot_meridional_average(p, value, unit, name):
    z = p.elements3d.z[0, 0, :].value_in(units.m)

    depth = p.elements.depth.value_in(units.m)

    mask = [numpy.max(depth, axis=0) < zi for zi in z]
    mask = numpy.array(mask).T

    avg = value[0, :, :] * 0
    scaling = value[0, :, :] * 0

    for i, j, k in numpy.ndindex(*value.shape):
        if depth[i, j] > z[k]:
            avg[j, k] += value[i, j, k]
            scaling[j, k] += 1

    scaling[mask] = 1
    avg /= scaling

    avg = numpy.ma.array(avg, mask=mask)

    y = p.elements3d.lat[0, :, 0].value_in(units.deg)

    pyplot.figure()
    pyplot.contourf(y, -z, avg.T)
    pyplot.xticks([-60, -30, 0, 30, 60],
                  ['60°S', '30°S', '0°', '30°N', '60°N'])
    pyplot.xlim(y[1], y[-2])
    yticks = [0, -1000, -2000, -3000, -4000, -5000]
    pyplot.yticks(yticks, [str(int(abs(i))) for i in yticks])
    pyplot.ylabel('Depth (m)')
    pyplot.colorbar(label=unit)
    pyplot.savefig(name)
    pyplot.close()


def plot_salinity(p, name="salinity.eps"):
    salinity = p.elements3d.salinity.value_in(units.psu)
    plot_meridional_average(p, salinity, "psu", name)


def plot_temperature(p, name="temperature.eps"):
    temperature = p.elements3d.temperature.value_in(units.Celsius)
    plot_meridional_average(p, temperature, "°C", name)


def plot_surface_salinity(p, name="surface_salinity.eps"):
    val = p.elements3d.salinity[:, :, 0].value_in(units.psu)
    plot_globe(p, val, "psu", name, elements=True)


def plot_surface_temperature(p, name="surface_temperature.eps"):
    val = p.elements3d.temperature[:, :, 0].value_in(units.Celsius)
    plot_globe(p, val, "°C", name, elements=True)


def plot_streamplot(p, name="streamplot.eps"):
    x = p.nodes3d.lon[:, 0, 0]
    y = p.nodes3d.lat[0, :, 0]

    u = p.nodes3d.xvel[:, :, 0]
    v = p.nodes3d.yvel[:, :, 0]

    mask = p.elements.depth.value_in(units.km) == 0
    u2 = numpy.ma.array(u.value_in(units.m / units.s), mask=mask)

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), u2.T)
    pyplot.colorbar()
    pyplot.streamplot(
        x.value_in(units.deg), y.value_in(units.deg), u.T.value_in(units.m / units.s), v.T.value_in(units.m / units.s)
    )
    pyplot.savefig(name)
    pyplot.close()


def barotropic_streamfunction(p):
    u = p.nodes3d.xvel

    z = p.nodes3d.z[0, 0, :]
    z = z_from_center(z)
    dz = z[1:] - z[:-1]

    y = p.nodes3d.lat[0, :, 0]

    # Test for uniform cell size to make our life easier
    for i in range(1, len(y) - 1):
        assert abs(y[i + 1] - 2 * y[i] + y[i - 1]) < 1e-12

    dy = (y[1] - y[0]).value_in(units.rad)
    dy *= constants.Rearth

    psib = bstream.barotropic_streamfunction(u, dz, dy)
    return psib.value_in(units.Sv)


def plot_barotropic_streamfunction(p, name="bstream.eps"):
    # psib = psib.value_in(units.Sv)[:, 1:]
    psib = barotropic_streamfunction(p)
    plot_globe(p, psib[:, 1:], "Sv", name, lims=(-60, 150))


def overturning_streamfunction(p):
    v = p.nodes3d.yvel

    z = p.nodes3d.z[0, 0, :]
    z = z_from_center(z)
    dz = z[1:] - z[:-1]

    x = p.nodes3d.lon[:, 0, 0]

    # Test for uniform cell size to make our life easier
    for i in range(1, len(x) - 1):
        assert abs(x[i + 1] - 2 * x[i] + x[i - 1]) < 1e-12

    dx = x[1] - x[0]
    dx *= numpy.cos(p.nodes3d[0, :, 0].lat.value_in(units.rad))
    dx *= constants.Rearth

    psim = bstream.overturning_streamfunction(v, dz, dx)
    return psim.value_in(units.Sv)


def plot_overturning_streamfunction(p, name="mstream.eps"):
    psim = overturning_streamfunction(p)

    z = p.nodes3d.z[0, 0, :]
    z = z_from_center(z)

    mask = [numpy.max(p.elements.depth.value_in(units.km), axis=0) < zi for zi in z.value_in(units.km)]
    mask = numpy.array(mask).T
    psim = numpy.ma.array(psim, mask=mask)

    y = p.nodes3d.lat[0, :, 0]

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), -z.value_in(units.m), psim.T)
    pyplot.colorbar()
    y = y.value_in(units.deg)
    pyplot.xlim(y[1], y[-2])
    pyplot.savefig(name)
    pyplot.close()


def depth_integrated_temperature(p, max_depth=None):
    t = p.elements3d.temperature

    z = p.nodes3d.z[0, 0, :]
    z = z_from_center(z)

    if max_depth:
        i = [i for i in range(len(z)) if z[i].value_in(units.m) < max_depth]
        z = z[i]
        t = t[:, :, i[:-1]]

    dz = z[1:] - z[:-1]

    # depth integration
    tint = (t * dz).sum(axis=-1) / dz.sum()
    return tint.value_in(units.Celsius)


def plot_tdata(directory="snapshots", fname="tdata.txt"):
    with open(os.path.join(directory, fname)) as f:
        t = []
        psib_min = []
        psib_max = []
        psim_min = []
        psim_max = []
        for line in f.readlines():
            data = line.split()
            t.append(float(data[0]))
            psib_min.append(float(data[1]))
            psib_max.append(float(data[2]))
            psim_min.append(float(data[3]))
            psim_max.append(float(data[4]))

    pyplot.plot(t, psib_min)
    pyplot.plot(t, psib_max)
    pyplot.title("Barotropic streamfunction")
    pyplot.legend(("Psi min", "Psi max"))
    pyplot.xlabel("t (year)")
    pyplot.ylabel("$ \\Psi $")
    pyplot.savefig(os.path.join(directory, "psib.eps"))
    pyplot.close()

    pyplot.plot(t, psim_min)
    pyplot.plot(t, psim_max)
    pyplot.title("Meridional streamfunction")
    pyplot.legend(("Psi min", "Psi max"))
    pyplot.xlabel("t (year)")
    pyplot.ylabel("$ \\Psi $")
    pyplot.savefig(os.path.join(directory, "psim.eps"))
    pyplot.close()


def save_pop_state(p, label, directory="./"):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for d in p.data_store_names():
        # print(d,getattr(p, d))
        fname = os.path.join(directory, label + "_" + d + ".amuse")
        write_set_to_file(getattr(p, d), fname, "amuse", overwrite_file=True)
        shutil.copy(fname, os.path.join(directory, "latest_" + d + ".amuse"))


def reset_pop_state(p, label, snapdir="snapshots"):
    nodes = read_set_from_file(os.path.join(snapdir, label + "_nodes.amuse"), "amuse")
    nodes3d = read_set_from_file(os.path.join(snapdir, label + "_nodes3d.amuse"), "amuse")
    elements = read_set_from_file(os.path.join(snapdir, label + "_elements.amuse"), "amuse")
    elements3d = read_set_from_file(os.path.join(snapdir, label + "_elements3d.amuse"), "amuse")

    channel1 = nodes.new_channel_to(p.nodes)
    channel1.copy_attributes(["gradx", "grady", "vx_barotropic", "vy_barotropic"])
    channel1.copy_attributes(["gradx_old", "grady_old", "vx_barotropic_old", "vy_barotropic_old"])
    # channel1.copy_attributes(["vx_barotropic", "vy_barotropic"])

    channel2 = nodes3d.new_channel_to(p.nodes3d)
    channel2.copy_attributes(["xvel", "yvel"])
    channel2.copy_attributes(["xvel_old", "yvel_old"])

    channel3 = elements3d.new_channel_to(p.elements3d)
    channel3.copy_attributes(["rho", "salinity", "temperature"])
    channel3.copy_attributes(["rho_old", "salinity_old", "temperature_old"])
    # channel3.copy_attributes(["salinity", "temperature"])

    channel1 = elements.new_channel_to(p.elements)
    channel1.copy_attributes(["ssh"])
    channel1.copy_attributes(["ssh_old", "ssh_guess"])

    p.parameters.pressure_correction = False
    p.parameters.ts_option = "amuse_restart"

    # p.parameters.reinit_gradp = True
    # p.parameters.reinit_rho = True


def reset_pop_state_from_pop_state(p, label, snapdir="snapshots"):
    old_state = read_pop_state(label, snapdir)

    # FIXME: For some reason we need to call [:, :, :].copy() here,
    # otherwise the grids don't derive from RegularBaseGrid.
    nodes = old_state.nodes[:, :].copy()
    nodes3d = old_state.nodes3d[:, :, :].copy()
    elements = old_state.elements[:, :].copy()
    elements3d = old_state.elements3d[:, :, :].copy()

    # Now we need a transformation to set values at points that were
    # previously land points

    z = elements3d.z[0, 0, :]
    mask = numpy.stack(
        [elements.depth.value_in(units.km) > zi for zi in z.value_in(units.km)],
        axis=-1)

    def transform(quantity):
        # Obtain a mask that has at least one ocean point per row. If
        # the entire row consists of land points, the mean will be
        # zero, so pretend the entire row is ocean instead.
        _mask = mask.copy()
        _mask[:, numpy.equal(numpy.any(mask, axis=0), False)] = True

        mean = numpy.mean(quantity, axis=0, where=_mask)
        return (quantity + numpy.equal(mask, False) * mean, )

    channel = elements3d.new_channel_to(elements3d)
    channel.transform(["rho"], transform, ["rho"])
    channel.transform(["salinity"], transform, ["salinity"])
    channel.transform(["temperature"], transform, ["temperature"])

    def transform(quantity):
        # Obtain a mask that has at least one ocean point per row. If
        # the entire row consists of land points, the mean will be
        # zero, so pretend the entire row is ocean instead.
        _mask = mask[:, :, 0].copy()
        _mask[:, numpy.equal(numpy.any(mask[:, :, 0], axis=0), False)] = True

        mean = numpy.mean(quantity, axis=0, where=_mask)
        return (quantity + numpy.equal(mask[:, :, 0], False) * mean, )

    channel = elements.new_channel_to(elements)
    channel.transform(["ssh"], transform, ["ssh"])
    channel.transform(["ssh_old"], transform, ["ssh_old"])
    channel.transform(["ssh_guess"], transform, ["ssh_guess"])

    channel1 = nodes.new_remapping_channel_to(p.nodes, bilinear_2D_remapper)
    channel1.copy_attributes(["gradx", "grady", "vx_barotropic", "vy_barotropic"])
    channel1.copy_attributes(["gradx_old", "grady_old", "vx_barotropic_old", "vy_barotropic_old"])

    channel2 = nodes3d.new_remapping_channel_to(p.nodes3d, bilinear_2D_remapper_3D)
    channel2.copy_attributes(["xvel", "yvel"])
    channel2.copy_attributes(["xvel_old", "yvel_old"])

    channel3 = elements3d.new_remapping_channel_to(p.elements3d, bilinear_2D_remapper_3D)
    channel3.copy_attributes(["rho", "salinity", "temperature"])
    channel3.copy_attributes(["rho_old", "salinity_old", "temperature_old"])

    channel1 = elements.new_remapping_channel_to(p.elements, bilinear_2D_remapper)
    channel1.copy_attributes(["ssh"])
    channel1.copy_attributes(["ssh_old", "ssh_guess"])

    p.parameters.pressure_correction = True
    p.parameters.ts_option = "amuse"

    p.parameters.reinit_gradp = True
    p.parameters.reinit_rho = True


def read_pop_state(label, directory="./"):
    class FakePopInterface:
        pass

    p = FakePopInterface()
    p.nodes3d = read_set_from_file(os.path.join(directory, label + "_nodes3d.amuse"), "amuse")
    p.nodes3d.set_axes_names(["lon", "lat", "z"])

    p.elements3d = read_set_from_file(os.path.join(directory, label + "_elements3d.amuse"), "amuse")
    p.elements3d.set_axes_names(["lon", "lat", "z"])

    p.nodes = read_set_from_file(os.path.join(directory, label + "_nodes.amuse"), "amuse")
    p.nodes.set_axes_names(["lon", "lat"])

    p.elements = read_set_from_file(os.path.join(directory, label + "_elements.amuse"), "amuse")
    p.elements.set_axes_names(["lon", "lat"])

    p.forcings = read_set_from_file(os.path.join(directory, label + "_forcings.amuse"), "amuse")
    p.forcings.set_axes_names(["lon", "lat"])

    p.element_forcings = read_set_from_file(os.path.join(directory, label + "_element_forcings.amuse"), "amuse")
    p.element_forcings.set_axes_names(["lon", "lat"])

    p.mode = "x".join((str(i) for i in p.nodes3d.xvel.shape))

    return p


def long_evolve(p, tend=100.0 | units.yr, dt=100.0 | units.day, dt2=1.0 | units.day, i0=0, snapdir="snapshots"):
    tnow = p.model_time
    tstart = tnow

    t1 = time.time()

    if not os.path.exists(snapdir):
        os.mkdir(snapdir)

    tdata = os.path.join(snapdir, "tdata.txt")
    with open(tdata, "w") as f:
        f.write("")

    i = i0
    while tnow < tend + dt / 2:
        tnext = tnow + dt

        # save
        label = "state_{0:06}".format(i)
        save_pop_state(p, label, directory=snapdir)

        psib = barotropic_streamfunction(p)
        psim = overturning_streamfunction(p)

        psib_min = min(psib.flatten())
        psib_max = max(psib.flatten())

        psim_min = min(psim.flatten())
        psim_max = max(psim.flatten())

        with open(tdata, "a") as f:
            t = p.model_time.value_in(units.yr)
            f.write("%.8e %.8e %.8e %.8e %.8e\n" % (t, psib_min, psib_max, psim_min, psim_max))

        print(p.evolve_model(tnow + dt2), flush=True)

        # label = label + ".1"
        # save_pop_state(p, label, directory=snapdir)

        t2 = time.time()
        eta = (tend - tnow - dt2) / ((tnow + dt2 - tstart) / (t2 - t1))
        print((t2 - t1) / 3600, "| evolve to", tnext.in_(units.yr), " ETA (hr):", eta / 3600.0, flush=True)

        p.evolve_model(tnext)
        tnow = p.model_time
        i = i + 1


def long_restart(p, ibegin, tend=100.0 | units.yr, dt=100.0 | units.day, loaddir="snapshots", snapdir="snapshots"):
    label = "state_{0:06}".format(ibegin)
    tsnap = ibegin * dt
    reset_pop_state(p, label, snapdir=loaddir)
    long_evolve(p, tend=tend - tsnap, dt=dt, i0=ibegin, snapdir=snapdir)
