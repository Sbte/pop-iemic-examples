import os
import numpy
import time
from matplotlib import pyplot

numpy.random.seed(123451)

from omuse.community.pop.interface import POP
from omuse.units import units, constants, quantities

from amuse.io import write_set_to_file, read_set_from_file

import bstream


def simple_upscale(x, fx, fy):
    # scale 2d x with integer factors
    return numpy.kron(x, numpy.ones((fx, fy)))


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


def initialize_pop(
    depth_levels, depth_array, mode="96x120x12", number_of_workers=4, latmin=-90 | units.deg, latmax=90 | units.deg
):

    print(f"initializing POP mode {mode} with {number_of_workers} workers")

    p = POP(
        number_of_workers=number_of_workers, mode=mode, redirection="none"
    )  # , job_scheduler="slurm") #, channel_type="sockets"

    dz = depth_levels[1:] - depth_levels[:-1]
    # print(f"dz: {dz}")

    p.parameters.topography_option = "amuse"
    p.parameters.depth_index = depth_array
    p.parameters.horiz_grid_option = "amuse"
    p.parameters.lonmin = 0 | units.deg
    p.parameters.lonmax = 360 | units.deg
    p.parameters.latmin = latmin
    p.parameters.latmax = latmax
    p.parameters.vert_grid_option = "amuse"
    p.parameters.vertical_layer_thicknesses = dz
    p.parameters.surface_heat_flux_forcing = "amuse"
    p.parameters.surface_freshwater_flux_forcing = "amuse"

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

        p.evolve_model(tnow + dt)
        tnow = p.model_time

        t = tnow.value_in(units.day)
        t = int(t)
        print("evolve to", t)


def plot_sst(p, label="sst"):
    pyplot.figure()
    pyplot.imshow(p.elements.temperature.value_in(units.Celsius).T, origin="lower", vmin=0, vmax=30)
    pyplot.colorbar()
    pyplot.savefig(label + ".png")
    pyplot.close()


def plot_ssh(p, label="ssh"):
    pyplot.figure()
    pyplot.imshow(p.elements.ssh.value_in(units.m).T, origin="lower", vmin=-1, vmax=1)
    pyplot.colorbar()
    pyplot.savefig(label + ".png")
    pyplot.close()


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
    x = p.nodes3d.lon[:, 0, 0]
    y = p.nodes3d.lat[0, :, 0]

    s = p.nodes3d.xvel[:, :, 0]

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), s.T.value_in(units.m / units.s))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def z_from_center(zc):
    z = numpy.zeros(len(zc) + 1) * zc[0]

    direction = 1
    if zc[0] <= zc[0] * 0:
        direction = -1

    for i, _zc in enumerate(zc[::direction]):
        half = _zc - z[i]
        z[i + 1] = z[i] + 2 * half

    return z[::direction]


def plot_salinity(p, name="salinity.eps"):
    z = p.nodes3d.z[0, 0, :]
    y = p.nodes3d.lat[0, :, 0]

    salinity = p.elements3d.salinity
    val = salinity[0, :, :]
    for i in range(1, salinity.shape[0]):
        val += salinity[i, :, :]

    val = val / salinity.shape[0]
    val = quantities.as_vector_quantity(val)

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), -z.value_in(units.m), val.T.value_in(units.psu))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_temperature(p, name="temperature.eps"):
    z = p.nodes3d.z[0, 0, :]
    y = p.nodes3d.lat[0, :, 0]

    temperature = p.elements3d.temperature
    val = temperature[0, :, :]
    for i in range(1, temperature.shape[0]):
        val += temperature[i, :, :]

    val = val / temperature.shape[0]

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), -z.value_in(units.m), val.T.value_in(units.Celsius))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_surface_salinity(p, name="surface_salinity.eps"):
    x = p.elements3d.lon[:, 0, 0]
    y = p.elements3d.lat[0, :, 0]

    val = p.elements3d.salinity[:, :, 0]
    mask = p.nodes.depth.value_in(units.km) == 0
    val = numpy.ma.array(val.value_in(units.psu), mask=mask)

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), val.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_surface_temperature(p, name="surface_temperature.eps"):
    x = p.elements3d.lon[:, 0, 0]
    y = p.elements3d.lat[0, :, 0]

    val = p.elements3d.temperature[:, :, 0]
    mask = p.nodes.depth.value_in(units.km) == 0
    val = numpy.ma.array(val.value_in(units.Celsius), mask=mask)

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), val.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_streamplot(p, name="streamplot.eps"):
    x = p.nodes3d.lon[:, 0, 0]
    y = p.nodes3d.lat[0, :, 0]

    u = p.nodes3d.xvel[:, :, 0]
    v = p.nodes3d.yvel[:, :, 0]

    mask = p.nodes.depth.value_in(units.km) == 0
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

    dy = y[1] - y[0]
    dy *= constants.Rearth

    psib = bstream.barotropic_streamfunction(u, dz, dy)
    return psib.value_in(units.Sv)


def plot_barotropic_streamfunction(p, name="bstream.eps"):
    # psib = psib.value_in(units.Sv)[:, 1:]
    psib = barotropic_streamfunction(p)
    mask = p.nodes.depth.value_in(units.km) == 0
    psib = numpy.ma.array(psib[:, 1:], mask=mask)

    x = p.nodes3d.lon[:, 0, 0]
    y = p.nodes3d.lat[0, :, 0]

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), psib.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


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

    y = p.nodes3d.lat[0, :, 0]

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), -z.value_in(units.m), psim.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def save_pop_state(p, label, directory="./"):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for d in p.data_store_names():
        # print(d,getattr(p, d))
        write_set_to_file(getattr(p, d), os.path.join(directory, label + "_" + d + ".amuse"), "amuse", overwrite_file=True)


def reset_pop_state(p, label, snapdir="snapshots"):
    nodes = read_set_from_file(os.path.join(snapdir, label + "_nodes.amuse"), "amuse")
    nodes3d = read_set_from_file(os.path.join(snapdir, label + "_nodes3d.amuse"), "amuse")
    elements = read_set_from_file(os.path.join(snapdir, label + "_elements.amuse"), "amuse")
    elements3d = read_set_from_file(os.path.join(snapdir, label + "_elements3d.amuse"), "amuse")

    channel1 = nodes.new_channel_to(p.nodes)
    # channel1.copy_attributes(["gradx","grady", "vx_barotropic", "vy_barotropic"])
    channel1.copy_attributes(["vx_barotropic", "vy_barotropic"])

    channel2 = nodes3d.new_channel_to(p.nodes3d)
    channel2.copy_attributes(["xvel", "yvel"])

    channel3 = elements3d.new_channel_to(p.elements3d)
    # channel3.copy_attributes(["rho", "salinity", "temperature"])
    channel3.copy_attributes(["salinity", "temperature"])

    channel1 = elements.new_channel_to(p.elements)
    channel1.copy_attributes(["ssh"])


def long_evolve(p, tend=100.0 | units.yr, dt=100.0 | units.day, dt2=1.0 | units.day, i0=0, snapdir="snapshots"):
    tnow = p.model_time
    tend = tnow + tend

    t1 = time.time()

    if not os.path.exists(snapdir):
        os.mkdir(snapdir)

    tdata = os.path.join(snapdir, "tdata.txt")
    with open(tdata, "w") as f:
        f.write("")

    i = i0
    while tnow < tend - dt / 2:
        tnow = p.model_time
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

        label = label + ".1"
        p.evolve_model(tnow + dt2)
        save_pop_state(p, label, directory=snapdir)
        i = i + 1

        t2 = time.time()

        if tnow > 0 * tnow:
            eta = (tend - tnow) / (tnow / (t2 - t1))
        else:
            eta = -1

        print((t2 - t1) / 3600, "| evolve to", tnext.in_(units.yr), " ETA (hr):", eta / 3600.0)
        p.evolve_model(tnext)


def long_restart(p, ibegin, tend=100.0 | units.yr, dt=100.0 | units.day, loaddir="snapshots", snapdir="snapshots"):
    label = "state_{0:06}".format(ibegin)
    tsnap = ibegin * dt
    reset_pop_state(p, label, snapdir=loaddir)
    long_evolve(p, tend=tend - tsnap, dt=dt, i0=ibegin, snapdir=snapdir)


def evolve_test(p, days=10):
    tend1 = days | units.day
    label1 = str(days)

    t1 = time.time()

    evolve(p, tend=tend1)
    save_pop_state(p, label1)

    t2 = time.time()

    print("time:", t2 - t1)


def restart_test(label, tend):
    p = initialize_pop()

    reset_pop_state(p, label)
    # save_pop_state(p, "restart")

    t1 = time.time()

    evolve(p, tend)
    save_pop_state(p, "restart")

    t2 = time.time()

    p.stop()

    print("time:", t2 - t1)


def compare(label1="201", label2="restart"):
    nodes1 = read_set_from_file(label1 + "_nodes.amuse", "amuse")
    nodes2 = read_set_from_file(label2 + "_nodes.amuse", "amuse")

    plot_ssh(nodes1, "ssh_" + label1)
    plot_ssh(nodes2, "ssh_" + label2)


if __name__ == "__main__":
    pass
    # evolve_test()
    # restart_test("200", 1 | units.day)
    # compare(label1="201")
    # p=initialize_pop(8)
    # long_evolve(p, tend=5000 | units.yr, dt=1000. | units.day)
    # long_restart(p,1068, tend=5000. | units.yr, dt=1000. | units.day, loaddir="snapshots", snapdir="snapshots2")
