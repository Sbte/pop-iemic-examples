"""
  example for running I-EMIC in a global configuration using OMUSE
"""

import xml.etree.ElementTree as xml
import numpy
import os
import shutil

from matplotlib import pyplot

from omuse.io import write_set_to_file, read_set_from_file
from omuse.units import units, constants, quantities

from omuse.community.iemic.interface import iemic

import bstream

from fvm import Continuation

# some utility functions
# note that i-emic defines depth levels as negative numbers!


def z_from_center(zc, firstlevel=None):
    z = numpy.zeros(len(zc) + 1) * zc[0]
    direction = 1
    if zc[0] <= zc[0] * 0:
        direction = -1
    for i, _zc in enumerate(zc[::direction]):
        half = _zc - z[i]
        z[i + 1] = z[i] + 2 * half
    return z[::direction]


def depth_levels(N, stretch_factor=1.8):
    z = numpy.arange(N) / (1.0 * (N - 1))
    if stretch_factor == 0:
        return z
    else:
        return 1 - numpy.tanh(stretch_factor * (1 - z)) / numpy.tanh(stretch_factor)


def read_global_mask(Nx, Ny, Nz, filename=None):
    if filename is None:
        filename = "mask_global_{0}x{1}x{2}".format(Nx, Ny, Nz)

    mask = numpy.zeros((Nx + 2, Ny + 2, Nz + 2), dtype="int")

    f = open(filename, "r")
    for k in range(Nz + 2):
        line = f.readline()  # ignored
        for j in range(Ny + 2):
            line = f.readline()
            mask[:, j, k] = numpy.array([int(d) for d in line[:-1]])  # ignore newline

    mask = mask[1:-1, 1:-1, 1:-1]  # ignore edges

    return mask[:, ::-1, :]  # reorient


def depth_array(Nx, Ny, Nz, filename=None):
    mask = read_global_mask(Nx, Ny, Nz, filename)
    return depth_array_from_mask(mask)


def depth_array_from_mask(mask):
    Nx, Ny, Nz = mask.shape
    mask_ = numpy.ones((Nx, Ny, Nz + 1))
    mask_[:, :, 1:] = mask

    depth = mask_[:, :, :].argmin(axis=2)
    a = depth > 0

    depth[a] = Nz - (depth[a]) + 1

    return depth


"""
OCEAN parameters:

Ocean__Analyze_Jacobian: True
Ocean__Belos_Solver__FGMRES_explicit_residual_test: False
Ocean__Belos_Solver__FGMRES_iterations: 500
Ocean__Belos_Solver__FGMRES_output: 100
Ocean__Belos_Solver__FGMRES_restarts: 0
Ocean__Belos_Solver__FGMRES_tolerance: 1e-08
Ocean__Input_file: ocean_input.h5
Ocean__Load_mask: True
Ocean__Load_salinity_flux: False
Ocean__Load_state: False
Ocean__Load_temperature_flux: False
Ocean__Max_mask_fixes: 5
Ocean__Output_file: ocean_output.h5
Ocean__Save_column_integral: False
Ocean__Save_frequency: 0
Ocean__Save_mask: True
Ocean__Save_salinity_flux: True
Ocean__Save_state: True
Ocean__Save_temperature_flux: True
Ocean__THCM__Compute_salinity_integral: True
Ocean__THCM__Coriolis_Force: 1
Ocean__THCM__Coupled_Salinity: 0
Ocean__THCM__Coupled_Sea_Ice_Mask: 1
Ocean__THCM__Coupled_Temperature: 0
Ocean__THCM__Depth_hdim: 4000.0
Ocean__THCM__Fix_Pressure_Points: False
Ocean__THCM__Flat_Bottom: False
Ocean__THCM__Forcing_Type: 0
Ocean__THCM__Global_Bound_xmax: 350.0
Ocean__THCM__Global_Bound_xmin: 286.0
Ocean__THCM__Global_Bound_ymax: 74.0
Ocean__THCM__Global_Bound_ymin: 10.0
Ocean__THCM__Global_Grid_Size_l: 16
Ocean__THCM__Global_Grid_Size_m: 16
Ocean__THCM__Global_Grid_Size_n: 16
Ocean__THCM__Grid_Stretching_qz: 1.0
Ocean__THCM__Inhomogeneous_Mixing: 0
Ocean__THCM__Integral_row_coordinate_i: -1
Ocean__THCM__Integral_row_coordinate_j: -1
Ocean__THCM__Land_Mask: no_mask_specified
Ocean__THCM__Levitus_Internal_T_S: False
Ocean__THCM__Levitus_S: 1
Ocean__THCM__Levitus_T: 1
Ocean__THCM__Linear_EOS:_alpha_S: 0.00076
Ocean__THCM__Linear_EOS:_alpha_T: 0.0001
Ocean__THCM__Local_SRES_Only: False
Ocean__THCM__Mixing: 1
Ocean__THCM__Periodic: False
Ocean__THCM__Problem_Description: Unnamed
Ocean__THCM__Read_Land_Mask: False
Ocean__THCM__Read_Salinity_Perturbation_Mask: False
Ocean__THCM__Restoring_Salinity_Profile: 1
Ocean__THCM__Restoring_Temperature_Profile: 1
Ocean__THCM__Rho_Mixing: True
Ocean__THCM__Salinity_Forcing_Data: levitus/new/s00an1
Ocean__THCM__Salinity_Integral_Sign: -1
Ocean__THCM__Salinity_Perturbation_Mask: no_mask_specified
Ocean__THCM__Scaling: THCM
Ocean__THCM__Taper: 1
Ocean__THCM__Temperature_Forcing_Data: levitus/new/t00an1
Ocean__THCM__Topography: 1
Ocean__THCM__Wind_Forcing_Data: wind/trtau.dat
Ocean__THCM__Wind_Forcing_Type: 2
Ocean__Use_legacy_fort.3_output: False
Ocean__Use_legacy_fort.44_output: True

starting (derived parameters, defaults):
Ocean__THCM__Starting_Parameters__ALPC: 1.0
Ocean__THCM__Starting_Parameters__AL_T: 0.133844130562
Ocean__THCM__Starting_Parameters__ARCL: 0.0
Ocean__THCM__Starting_Parameters__CMPR: 0.0
Ocean__THCM__Starting_Parameters__CONT: 0.0
Ocean__THCM__Starting_Parameters__Combined_Forcing: 0.0
Ocean__THCM__Starting_Parameters__Energy: 100.0
Ocean__THCM__Starting_Parameters__Flux_Perturbation: 0.0
Ocean__THCM__Starting_Parameters__Horizontal_Ekman_Number: 4.22458923802e-05
Ocean__THCM__Starting_Parameters__Horizontal_Peclet_Number: 0.00156985871272
Ocean__THCM__Starting_Parameters__IFRICB: 0.0
Ocean__THCM__Starting_Parameters__LAMB: 7.6
Ocean__THCM__Starting_Parameters__MIXP: 0.0
Ocean__THCM__Starting_Parameters__MKAP: 0.0
Ocean__THCM__Starting_Parameters__NLES: 0.0
Ocean__THCM__Starting_Parameters__Nonlinear_Factor: 9.83024691358
Ocean__THCM__Starting_Parameters__P_VC: 6.37
Ocean__THCM__Starting_Parameters__RESC: 0.0
Ocean__THCM__Starting_Parameters__Rayleigh_Number: 0.0527448415545
Ocean__THCM__Starting_Parameters__Rossby_Number: 0.000107642533785
Ocean__THCM__Starting_Parameters__SPL1: 2000.0
Ocean__THCM__Starting_Parameters__SPL2: 0.01
Ocean__THCM__Starting_Parameters__Salinity_Forcing: 1.0
Ocean__THCM__Starting_Parameters__Salinity_Homotopy: 0.0
Ocean__THCM__Starting_Parameters__Salinity_Perturbation: 0.0
Ocean__THCM__Starting_Parameters__Solar_Forcing: 0.0
Ocean__THCM__Starting_Parameters__Temperature_Forcing: 10.0
Ocean__THCM__Starting_Parameters__Vertical_Ekman_Number: 2.74273176083e-07
Ocean__THCM__Starting_Parameters__Vertical_Peclet_Number: 0.0002548
Ocean__THCM__Starting_Parameters__Wind_Forcing: 1.0
"""


def initialize_global_iemic(number_of_workers=1, redirection="none"):

    print(f"initializing IEMIC with {number_of_workers} workers")

    i = iemic(
        number_of_workers=number_of_workers,
        redirection=redirection,
        channel_type="sockets",
    )

    i.parameters.Ocean__Belos_Solver__FGMRES_tolerance = 1e-03
    i.parameters.Ocean__Belos_Solver__FGMRES_iterations = 800

    i.parameters.Ocean__Save_state = False

    i.parameters.Ocean__THCM__Global_Bound_xmin = 0
    i.parameters.Ocean__THCM__Global_Bound_xmax = 360
    i.parameters.Ocean__THCM__Global_Bound_ymin = -81
    i.parameters.Ocean__THCM__Global_Bound_ymax = 81

    i.parameters.Ocean__THCM__Periodic = True
    i.parameters.Ocean__THCM__Global_Grid_Size_n = 120
    i.parameters.Ocean__THCM__Global_Grid_Size_m = 54
    i.parameters.Ocean__THCM__Global_Grid_Size_l = 12

    i.parameters.Ocean__THCM__Grid_Stretching_qz = 1.8
    i.parameters.Ocean__THCM__Depth_hdim = 5000.0

    i.parameters.Ocean__THCM__Topography = 0
    i.parameters.Ocean__THCM__Flat_Bottom = False

    i.parameters.Ocean__THCM__Read_Land_Mask = True
    i.parameters.Ocean__THCM__Land_Mask = "global_120x54x12.mask"

    i.parameters.Ocean__THCM__Rho_Mixing = False

    i.parameters.Ocean__THCM__Starting_Parameters__Combined_Forcing = 0.0
    i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing = 1.0
    i.parameters.Ocean__THCM__Starting_Parameters__Solar_Forcing = 0.0
    i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing = 10.0
    i.parameters.Ocean__THCM__Starting_Parameters__Wind_Forcing = 1.0

    # Parameters for Levitus
    # i.parameters.Ocean__THCM__Levitus_S = 0
    # i.parameters.Ocean__THCM__Levitus_T = 0
    # i.parameters.Ocean__THCM__Wind_Forcing_Type = 0
    # i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing = 1.0

    # Set salinity to 0.1 first, do a continuation to 1 later.
    # i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing = 0.1

    i.parameters.Ocean__Analyze_Jacobian = True

    return i


def get_mask(i):
    return i.grid.mask


def get_surface_forcings(i, forcings_file="forcings.amuse"):

    # these are hardcoded in iemic!
    t0 = 15.0 | units.Celsius
    s0 = 35.0 | units.psu
    tau0 = 0.1 | units.Pa

    # amplitudes
    t_a = i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing | units.Celsius
    s_a = i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing | units.psu

    def attributes(lon, lat, tatm, emip, tau_x, tau_y):
        return (lon, lat, t0 + t_a * tatm, s0 + s_a * emip, tau0 * tau_x, tau0 * tau_y)

    forcings = i.surface_forcing.empty_copy()
    channel = i.surface_forcing.new_channel_to(forcings)
    channel.transform(
        ["lon", "lat", "tatm", "emip", "tau_x", "tau_y"],
        attributes,
        ["lon", "lat", "tatm", "emip", "tau_x", "tau_y"],
    )

    write_set_to_file(forcings, forcings_file, "amuse", overwrite_file=True)

    return forcings


def save_iemic_state(i, label, directory="./"):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for d in i.data_store_names():
        fname = os.path.join(directory, label + "_" + d + ".amuse")
        write_set_to_file(
            getattr(i, d),
            fname,
            "amuse",
            overwrite_file=True,
        )
        shutil.copy(fname, os.path.join(directory, "latest_" + d + ".amuse"))

    fname = os.path.join(directory, label + "_parameters.xml")
    i.save_xml_parameters("Ocean", fname)
    shutil.copy(fname, os.path.join(directory, "latest_parameters.xml"))


def load_iemic_state(i, label, directory="./", copy_forcing=False):
    v_grid = read_set_from_file(os.path.join(directory, label + "_v_grid.amuse"), "amuse")
    t_grid = read_set_from_file(os.path.join(directory, label + "_t_grid.amuse"), "amuse")

    channel = v_grid.new_channel_to(i.v_grid)
    channel.copy_attributes(["u_velocity", "v_velocity"])

    channel = t_grid.new_channel_to(i.t_grid)
    channel.copy_attributes(["pressure", "temperature", "salinity"])

    i.load_xml_parameters("Ocean", os.path.join(directory, label + "_parameters.xml"))

    if not copy_forcing:
        return

    surface_v_grid = read_set_from_file(os.path.join(directory, label + "_surface_v_grid.amuse"), "amuse")
    surface_t_grid = read_set_from_file(os.path.join(directory, label + "_surface_t_grid.amuse"), "amuse")

    channel = surface_v_grid.new_channel_to(i.surface_v_grid)
    channel.copy_attributes(["tau_x", "tau_y"])

    channel = surface_t_grid.new_channel_to(i.surface_t_grid)
    channel.copy_attributes(["tatm", "emip"])


def read_iemic_parameters(label, directory="./"):
    tree = xml.parse(os.path.join(directory, label + "_parameters.xml"))
    return tree.getroot()


def read_iemic_state_with_units(label, directory="./"):
    class FakeIemicInterface:
        def __init__(self):
            self.parameters = None

        def _get_parameter(self, root, *args):
            elem = root.find(".//*[@name='{}']".format(args[0]))
            if len(args) > 1:
                return self._get_parameter(elem, *args[1:])

            return float(elem.get("value"))

        def get_parameter(self, name):
            return self._get_parameter(self.parameters, *name.split("->"))

    i = FakeIemicInterface()
    i.parameters = read_iemic_parameters(label, directory)

    i.v_grid = read_set_from_file(os.path.join(directory, label + "_v_grid.amuse"), "amuse")
    i.v_grid = get_grid_with_units(i.v_grid)
    i.v_grid.set_axes_names(["lon", "lat", "z"])

    i.t_grid = read_set_from_file(os.path.join(directory, label + "_t_grid.amuse"), "amuse")
    i.t_grid = get_grid_with_units(i.t_grid)
    i.t_grid.set_axes_names(["lon", "lat", "z"])

    i.surface_v_grid = read_set_from_file(os.path.join(directory, label + "_surface_v_grid.amuse"), "amuse")
    i.surface_v_grid = get_forcing_with_units(i, i.surface_v_grid)
    i.surface_v_grid.set_axes_names(["lon", "lat"])

    i.surface_t_grid = read_set_from_file(os.path.join(directory, label + "_surface_t_grid.amuse"), "amuse")
    i.surface_t_grid = get_forcing_with_units(i, i.surface_t_grid)
    i.surface_t_grid.set_axes_names(["lon", "lat"])

    return i


# convenience function to get grid with physical units
# this should really be available on the iemic interface
def get_grid_with_units(grid):
    result = grid.empty_copy()

    channel = grid.new_channel_to(result)

    attributes = grid.get_attribute_names_defined_in_store()
    channel.copy_attributes([attr for attr in ["mask", "lon", "lat", "z"] if attr in attributes])

    # hardcoded constants in I-EMIC
    rho0 = 1.024e03 | units.kg / units.m ** 3
    r0 = 6.37e06 | units.m
    omega0 = 7.292e-05 | units.s ** -1
    uscale = 0.1 | units.m / units.s
    t0 = 15.0 | units.Celsius
    s0 = 35.0 | units.psu
    pscale = 2 * omega0 * r0 * uscale * rho0
    s_scale = 1.0 | units.psu
    t_scale = 1 | units.Celsius

    def add_units_v(*args):
        return [uscale * arg for arg in args]

    # note pressure is pressure anomaly (ie difference from hydrostatic)
    def add_units_t(mask, pressure, salt, temp):
        # salt and temp need to account for mask
        # _salt = s0 * (mask == 0) + s_scale * salt
        # _temp = t0 * (mask == 0) + t_scale * temp
        _salt = s0 + s_scale * salt
        _temp = t0 + t_scale * temp
        _salt = quantities.as_vector_quantity(_salt)
        return (
            pscale * pressure,
            _salt,
            _temp,
        )

    if "u_velocity" in attributes:
        channel.transform(
            [
                "u_velocity",
                "v_velocity",
            ],
            add_units_v,
            [
                "u_velocity",
                "v_velocity",
            ],
        )

        channel.transform(
            [
                "u_forcing",
                "v_forcing",
            ],
            add_units_v,
            [
                "u_forcing",
                "v_forcing",
            ],
        )
    elif "w_velocity" in attributes:
        channel.transform(
            [
                "w_velocity",
            ],
            add_units_v,
            [
                "w_velocity",
            ],
        )

        channel.transform(
            [
                "w_forcing",
            ],
            add_units_v,
            [
                "w_forcing",
            ],
        )
    else:
        channel.transform(
            [
                "pressure",
                "salinity",
                "temperature",
            ],
            add_units_t,
            [
                "mask",
                "pressure",
                "salinity",
                "temperature",
            ],
        )

        channel.transform(
            [
                "pressure_forcing",
                "salinity_forcing",
                "temperature_forcing",
            ],
            add_units_t,
            [
                "mask",
                "pressure_forcing",
                "salinity_forcing",
                "temperature_forcing",
            ],
        )

    return result


def get_forcing_with_units(i, grid):
    result = grid.empty_copy()

    channel = grid.new_channel_to(result)

    attributes = grid.get_attribute_names_defined_in_store()
    channel.copy_attributes([attr for attr in ["lon", "lat"] if attr in attributes])

    # these are hardcoded in iemic!
    t0 = 15.0 | units.Celsius
    s0 = 35.0 | units.psu
    tau0 = 0.1 | units.Pa

    # amplitudes
    cf = i.get_parameter("THCM->Starting Parameters->Combined Forcing")
    t_a = cf * i.get_parameter("THCM->Starting Parameters->Temperature Forcing") | units.Celsius
    s_a = cf * i.get_parameter("THCM->Starting Parameters->Salinity Forcing") | units.psu

    def add_units_v(tau_x, tau_y):
        return (tau0 * tau_x, tau0 * tau_y)

    def add_units_t(tatm, emip):
        return (t0 + t_a * tatm, s0 + s_a * emip)

    if "tau_x" in attributes:
        channel.transform(
            [
                "tau_x",
                "tau_y",
            ],
            add_units_v,
            [
                "tau_x",
                "tau_y",
            ],
        )
    else:
        channel.transform(
            [
                "tatm",
                "emip",
            ],
            add_units_t,
            [
                "tatm",
                "emip",
            ],
        )

    return result


def plot_u_velocity(state, name="u_velocity.eps"):
    x = state.v_grid.lon[:, 0, 0]
    y = state.v_grid.lat[0, :, 0]

    u = state.v_grid.u_velocity[:, :, -1]

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), u.T.value_in(units.m / units.s))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_v_velocity(state, name="v_velocity.eps"):
    x = state.v_grid.lon[:, 0, 0]
    y = state.v_grid.lat[0, :, 0]

    v = state.v_grid.v_velocity[:, :, -1]

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), v.T.value_in(units.m / units.s))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_surface_pressure(state, name="surface_pressure.eps"):
    x = state.t_grid.lon[:, 0, 0]
    y = state.t_grid.lat[0, :, 0]

    val = state.t_grid.pressure[:, :, -1]
    val = numpy.ma.array(val.value_in(units.Pa), mask=state.t_grid.mask[:, :, -1])

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), val.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_surface_salinity(state, name="surface_salinity.eps"):
    x = state.t_grid.lon[:, 0, 0]
    y = state.t_grid.lat[0, :, 0]

    val = state.t_grid.salinity[:, :, -1]
    val = numpy.ma.array(val.value_in(units.psu), mask=state.t_grid.mask[:, :, -1])

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), val.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_surface_temperature(state, name="surface_temperature.eps"):
    x = state.t_grid.lon[:, 0, 0]
    y = state.t_grid.lat[0, :, 0]

    val = state.t_grid.temperature[:, :, -1]
    val = numpy.ma.array(val.value_in(units.Celsius), mask=state.t_grid.mask[:, :, -1])

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), val.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_salinity(state, name="salinity.eps"):
    z = state.t_grid.z[0, 0, :]
    y = state.t_grid.lat[0, :, 0]

    s = state.t_grid.salinity[0, :, :]
    for i in range(1, state.t_grid.salinity.shape[0]):
        s += state.t_grid.salinity[i, :, :]

    s = s / state.t_grid.salinity.shape[0]
    s = quantities.as_vector_quantity(s)

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), z.value_in(units.m), s.T.value_in(units.psu))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_temperature(state, name="temperature.eps"):
    z = state.t_grid.z[0, 0, :]
    y = state.t_grid.lat[0, :, 0]

    t = state.t_grid.temperature[0, :, :]
    for i in range(1, state.t_grid.temperature.shape[0]):
        t += state.t_grid.temperature[i, :, :]

    t = t / state.t_grid.temperature.shape[0]

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), z.value_in(units.m), t.T.value_in(units.Celsius))
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def plot_streamplot(state, name="streamplot.eps"):
    x = state.v_grid.lon[:, 0, 0]
    y = state.v_grid.lat[0, :, 0]

    u = state.v_grid.u_velocity[:, :, -1]
    v = state.v_grid.v_velocity[:, :, -1]

    u2 = numpy.ma.array(u.value_in(units.m / units.s), mask=state.t_grid.mask[:, :, -1])

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), u2.T)
    pyplot.colorbar()
    pyplot.streamplot(
        x.value_in(units.deg),
        y.value_in(units.deg),
        u.T.value_in(units.m / units.s),
        v.T.value_in(units.m / units.s),
    )
    pyplot.savefig(name)
    pyplot.close()


def barotropic_streamfunction(state):
    u = state.v_grid.u_velocity

    z = state.v_grid.z[0, 0, :]
    z = z_from_center(z)
    dz = z[1:] - z[:-1]

    y = state.v_grid.lat[0, :, 0]

    # Test for uniform cell size to make our life easier
    for i in range(1, len(y) - 1):
        assert abs(y[i + 1] - 2 * y[i] + y[i - 1]) < 1e-12

    dy = (y[1] - y[0]).value_in(units.rad)
    dy *= constants.Rearth

    psib = bstream.barotropic_streamfunction(u, dz, dy)
    return psib.value_in(units.Sv)


def plot_barotropic_streamfunction(state, name="bstream.eps"):
    psib = barotropic_streamfunction(state)
    psib = numpy.ma.array(psib[:, 1:], mask=state.t_grid.mask[:, :, -1])

    x = state.v_grid.lon[:, 0, 0]
    y = state.v_grid.lat[0, :, 0]

    pyplot.figure()
    pyplot.contourf(x.value_in(units.deg), y.value_in(units.deg), psib.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def overturning_streamfunction(state):
    v = state.v_grid.v_velocity

    z = state.v_grid.z[0, 0, :]
    z = z_from_center(z)
    dz = z[1:] - z[:-1]

    x = state.v_grid.lon[:, 0, 0]

    # Test for uniform cell size to make our life easier
    for i in range(1, len(x) - 1):
        assert abs(x[i + 1] - 2 * x[i] + x[i - 1]) < 1e-12

    dx = x[1] - x[0]
    dx *= numpy.cos(state.v_grid.lat.value_in(units.rad)[0, :, 0])
    dx *= constants.Rearth

    psim = bstream.overturning_streamfunction(v, dz, dx)
    return psim.value_in(units.Sv)


def plot_overturning_streamfunction(state, name="mstream.eps"):
    psim = overturning_streamfunction(state)

    z = state.v_grid.z[0, 0, :]
    z = z_from_center(z)

    y = state.v_grid.lat[0, :, 0]

    pyplot.figure()
    pyplot.contourf(y.value_in(units.deg), z.value_in(units.m), psim.T)
    pyplot.colorbar()
    pyplot.savefig(name)
    pyplot.close()


def get_ssh(i):
    surface = i.t_grid[:, :, -1]  # note surface is -1
    result = surface.empty_copy()

    channel = surface.new_channel_to(result)
    channel.copy_attributes(["lon", "lat"])

    # values hardcoded in IEMIC
    rho0 = 1.024e03 | units.kg / units.m ** 3
    g = 9.8 | units.m / units.s ** 2

    result.ssh = surface.pressure / (rho0 * g)

    return result


def get_barotropic_velocities(i):
    surface = i.v_grid[:, :, -1]  # note surface is -1
    result = surface.empty_copy()

    channel = surface.new_channel_to(result)

    channel.copy_attributes(["lon", "lat"])

    z = i.v_grid[0, 0, :].z

    z_ = z_from_center(z)
    dz = z_[1:] - z_[:-1]

    def average_vel(v, dz):
        return (v * dz).sum(axis=-1) / dz.sum()

    result.uvel_barotropic = average_vel(i.v_grid.u_velocity, dz)
    result.vvel_barotropic = average_vel(i.v_grid.v_velocity, dz)

    return result


# get surface grid with mask, lon, lat, ssh, uvel_barotropic, vvel_barotropic
def get_surface_grid(grid):
    surface = grid[:, :, -1]  # note surface is -1
    result = surface.empty_copy()

    channel = surface.new_channel_to(result)

    channel.copy_attributes(["mask", "lon", "lat"])

    z = grid[0, 0, :].z

    z_ = z_from_center(z)
    dz = z_[1:] - z_[:-1]

    def average_vel(v, dz):
        return (v * dz).sum(axis=-1) / dz.sum()

    result.uvel_barotropic = average_vel(grid.u_velocity, dz)
    result.vvel_barotropic = average_vel(grid.v_velocity, dz)

    # values hardcoded in IEMIC
    rho0 = 1.024e03 | units.kg / units.m ** 3
    g = 9.8 | units.m / units.s ** 2

    result.ssh = surface.pressure / (rho0 * g)

    return result


def get_equilibrium_state(instance, iemic_state_file="iemic_state.amuse"):
    print("starting equilibrium state by continuation of combined forcing from zero")

    # the following line optionally redirects iemic output to file
    # instance.set_output_file("output.%p")

    print("state1:", instance.get_name_of_current_state())

    print("all parameters (initial)")
    print(instance.parameters)

    x = instance.get_state()

    print("THCM parameters (after init)")
    print(instance.Ocean__THCM__Starting_Parameters)

    # numerical parameters for the continuation
    parameters = {
        "Newton Tolerance": 1.0e-2,
        "Verbose": True,
        "Minimum Step Size": 0.001,
        "Maximum Step Size": 0.2,
        "Delta": 1.0e-6,
    }

    # setup continuation object
    continuation = Continuation(instance, parameters)

    # Converge to an initial steady state
    x = continuation.newton(x)

    print("start continuation, this may take a while")

    print("state:", instance.get_name_of_current_state())

    x, mu = continuation.continuation(x, "Ocean->THCM->Starting Parameters->Combined Forcing", 0.0, 1.0, 0.005)

    print("continuation done")

    print(f"writing grid to {iemic_state_file}")
    write_set_to_file(x.grid, iemic_state_file, "amuse", overwrite_file=True)

    return x.grid.copy()
