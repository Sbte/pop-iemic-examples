import numpy

from omuse.units import units,trigo

from iemic import initialize_global_iemic, get_mask,depth_array_from_mask, get_surface_forcings, depth_levels

from pop import initialize_pop, long_evolve, plot_forcings_and_depth, evolve_test

from amuse.ext.grid_remappers import bilinear_2D_remapper, nearest_2D_remapper
from amuse.datamodel import new_regular_grid

from functools import partial

bilinear_2D_remapper=partial(bilinear_2D_remapper, check_inside=False)
nearest_2D_remapper=partial(nearest_2D_remapper, check_inside=False)

def simple_upscale(x, fx, fy):
  # scale 2d x with integer factors
  return numpy.kron(x, numpy.ones((fx,fy)))

def initialize_pop_with_iemic_setup(pop_number_of_workers=8):

    Nx=96
    Ny=40
    Nz=12
    
    iemic=initialize_global_iemic(number_of_workers=1)
  
    stretch_factor=iemic.parameters.Ocean__THCM__Grid_Stretching_qz
    Hdim=iemic.parameters.Ocean__THCM__Depth_hdim | units.m

    mask=get_mask(iemic) # get 3d mask
  
    surface_forcings=get_surface_forcings(iemic)
  
    iemic.stop()

    depth=numpy.zeros((Nx,Ny))
    depth[:,1:-1]=depth_array_from_mask(mask) # convert to (index) depth array

    levels=depth_levels(Nz+1,stretch_factor=stretch_factor)*Hdim

    # no interpolation for bathymetry
    depth=simple_upscale(depth, 1,3)

    pop=initialize_pop( levels, depth, mode="96x120x12", number_of_workers=pop_number_of_workers)#, latmin=latmin, latmax=latmax)
      
    channel=surface_forcings.new_remapping_channel_to(pop.forcings,  bilinear_2D_remapper)
    channel.copy_attributes(["tau_x", "tau_y"])

    channel=surface_forcings.new_remapping_channel_to(pop.element_forcings,  bilinear_2D_remapper)
    channel.copy_attributes(["tatm", "emip"], target_names=["restoring_temp","restoring_salt"])

    plot_forcings_and_depth(pop)

    return pop

if __name__=="__main__":
    pop=initialize_pop_with_iemic_setup()
    evolve_test(pop)
    #~ long_evolve(pop, tend=5000 | units.yr, dt=1000. | units.day)
    plot_forcings_and_depth(pop)
