"""
  example for running I-EMIC in a global configuration using OMUSE
  
"""

import numpy

from omuse.units import units, constants
from omuse.units import trigo

from omuse.community.iemic.interface import iemic

from omuse.io import write_set_to_file, read_set_from_file

from fvm import Continuation

from omuse.units import units

import numpy

# some utility functions
# note that i-emic defines depth levels as negative numbers!

def z_from_cellcenterz(zc, firstlevel=None):
    if firstlevel is None:
      top=0*zc[0]
    z=numpy.zeros(len(zc)+1)*zc[0]
    direction=1
    if zc[0]<=zc[0]*0:
      direction=-1
    for i,_zc in enumerate(zc[::direction]):
      half=_zc-z[i]
      z[i+1]=z[i]+2*half
    return z[::direction]

def depth_levels(N, stretch_factor=1.8): #1.8
    z=numpy.arange(N)/(1.*(N-1))
    if stretch_factor==0:
        return z
    else:
        return 1 - numpy.tanh(stretch_factor*(1-z))/numpy.tanh(stretch_factor)
    
#~ print h

def read_global_mask(Nx,Ny,Nz, filename=None):
    if filename is None:
        filename="mask_global_{0}x{1}x{2}".format(Nx, Ny, Nz)
    
    mask=numpy.zeros((Nx+2,Ny+2,Nz+2), dtype='int')
    
    f=open(filename,'r')
    for k in range(Nz+2):
      line=f.readline() # ignored
      for j in range(Ny+2):
          line=f.readline()
          mask[:,j,k]=numpy.array([int(d) for d in line[:-1]]) # ignore newline
    
    mask=mask[1:-1,1:-1,1:-1] # ignore edges
          
    return mask[:,::-1,:] # reorient

def depth_array(Nx,Ny,Nz, filename=None):
    mask=read_global_mask(Nx,Ny,Nz, filename)
    return depth_array_from_mask(mask)

def depth_array_from_mask(mask):
    Nx,Ny,Nz=mask.shape
    mask_=numpy.ones((Nx,Ny,Nz+1))
    mask_[:,:,1:]=mask

    depth=mask_[:,:,:].argmin(axis=2)
    a=depth>0

    depth[a]=Nz-(depth[a])+1
    
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
      
    i = iemic(number_of_workers=number_of_workers,redirection=redirection, channel_type="sockets")

    i.parameters.Ocean__Belos_Solver__FGMRES_tolerance=1e-03
    i.parameters.Ocean__Belos_Solver__FGMRES_iterations=800

    i.parameters.Ocean__Save_state=False
    
    i.parameters.Ocean__THCM__Global_Bound_xmin=0
    i.parameters.Ocean__THCM__Global_Bound_xmax=360
    i.parameters.Ocean__THCM__Global_Bound_ymin=-85.5
    i.parameters.Ocean__THCM__Global_Bound_ymax=85.5
    
    i.parameters.Ocean__THCM__Periodic=True
    i.parameters.Ocean__THCM__Global_Grid_Size_n=96
    i.parameters.Ocean__THCM__Global_Grid_Size_m=38 
    i.parameters.Ocean__THCM__Global_Grid_Size_l=12
    
    i.parameters.Ocean__THCM__Grid_Stretching_qz=2.25
    i.parameters.Ocean__THCM__Depth_hdim=5000.

    i.parameters.Ocean__THCM__Topography=0
    i.parameters.Ocean__THCM__Flat_Bottom=False
  
    i.parameters.Ocean__THCM__Read_Land_Mask=True
    i.parameters.Ocean__THCM__Land_Mask="global_96x38x12.mask"
    
    i.parameters.Ocean__THCM__Rho_Mixing=False
    
    i.parameters.Ocean__THCM__Starting_Parameters__Combined_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing=1.
    i.parameters.Ocean__THCM__Starting_Parameters__Solar_Forcing=0.
    i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing=10.
    i.parameters.Ocean__THCM__Starting_Parameters__Wind_Forcing=1.
              
    i.parameters.Ocean__Analyze_Jacobian=True

    return i

def get_mask(i):
    return i.grid.mask

def get_surface_forcings(i, forcings_file="forcings.amuse"):

    # these are hardcoded in iemic!
    t0=15. | units.Celsius
    s0=35. | units.psu
    tau0 = 0.1 | units.Pa 
    
    # amplitudes
    t_a=i.parameters.Ocean__THCM__Starting_Parameters__Temperature_Forcing | units.Celsius
    s_a=i.parameters.Ocean__THCM__Starting_Parameters__Salinity_Forcing | units.psu

    def attributes(lon,lat,tatm,emip,tau_x,tau_y):
      return (lon,lat, t0+t_a*tatm, s0+s_a*emip, tau0*tau_x, tau0*tau_y) 

    forcings=i.surface_forcing.empty_copy()
    channel=i.surface_forcing.new_channel_to(forcings)
    channel.transform(["lon","lat","tatm","emip","tau_x","tau_y"], 
                       attributes, 
                       ["lon","lat","tatm","emip","tau_x","tau_y"])

    write_set_to_file(forcings, forcings_file ,"amuse", overwrite_file=True)
    
    return forcings

# convenience function to get grid with physical units
# this should really be available on the iemic interface
def get_grid_with_units(grid):
    result=grid.empty_copy()

    channel=grid.new_channel_to(result)
    
    channel.copy_attributes(["mask", "lon", "lat","z"])

    # hardcoded constants in I-EMIC
    rho0=1.024e+03 | units.kg/units.m**3
    r0=6.37e+06 | units.m
    omega0=7.292e-05 | units.s**-1
    uscale=0.1 | units.m/units.s
    t0=15. | units.Celsius
    s0=35. | units.psu
    pscale=2*omega0*r0*uscale*rho0
    s_scale=1. | units.psu
    t_scale=1| units.Celsius
        
    # note pressure is pressure anomaly (ie difference from hydrostatic)
    def add_units(mask, xvel, yvel, zvel, pressure, salt, temp):
        # salt and temp need to account for mask
        #~ _salt=s0*(mask==0)+s_scale*salt
        #~ _temp=t0*(mask==0)+t_scale*temp
        _salt=s0+s_scale*salt
        _temp=t0+t_scale*temp
        return uscale*xvel, uscale*yvel, uscale*zvel, pscale*pressure, _salt, _temp 

    channel.transform(["u_velocity", "v_velocity", "w_velocity", "pressure", "salinity", "temperature"], 
                       add_units,
                       ["mask", "u_velocity", "v_velocity", "w_velocity", "pressure", "salinity", "temperature"])

    return result

# get surface grid with mask, lon, lat, ssh, uvel_barotropic, vvel_barotropic
def get_surface_grid(grid):
    surface=grid[:,:,-1] # note surface is -1
    result=surface.empty_copy()
    
    channel=surface.new_channel_to(result)
    
    channel.copy_attributes(["mask", "lon", "lat"])

    z=grid[0,0,:].z

    z_=z_from_cellcenterz(z)
    dz=z_[1:]-z_[:-1]

    def average_vel(v, dz):
      return (v*dz).sum(axis=-1)/dz.sum()
    
    result.uvel_barotropic=average_vel(grid.u_velocity,dz)
    result.vvel_barotropic=average_vel(grid.v_velocity,dz)

    # values hardcoded in IEMIC
    rho0=1.024e+03 | units.kg/units.m**3
    g=9.8 | units.m/units.s**2
    
    result.ssh=surface.pressure/(rho0*g)

    return result

def get_equilibrium_state(instance, iemic_state_file="iemic_state.amuse"):
    print("starting equilibrium state by continuation of combined forcing from zero")

    # the following line optionally redirects iemic output to file
    #~ instance.set_output_file("output.%p")

    print("state1:", instance.get_name_of_current_state())

    print("all parameters (initial)")
    print(instance.parameters)
    
    x = instance.get_state()

    print("THCM parameters (after init)")
    print(instance.Ocean__THCM__Starting_Parameters)
    
    # numerical parameters for the continuation
    parameters={"Newton Tolerance" : 1.e-2, "Verbose" : True,
                "Minimum Step Size" : 0.001,
                "Maximum Step Size" : 0.2,
                "Delta" : 1.e-6 }

    # setup continuation object
    continuation=Continuation(instance, parameters)
    
    # Converge to an initial steady state
    x = continuation.newton(x)
      
    print("start continuation, this may take a while")

    print("state:", instance.get_name_of_current_state())

    x, mu = continuation.continuation(x, 'Ocean->THCM->Starting Parameters->Combined Forcing', 0., 1., 0.005)

    print("continuation done")

    print(f"writing grid to {iemic_state_file}")
    write_set_to_file(x.grid, iemic_state_file,"amuse", overwrite_file=True)
    
    return x.grid.copy()
