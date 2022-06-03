import os
import numpy
import math
import sys
import time
from matplotlib import pyplot

numpy.random.seed(123451)

from omuse.community.pop.interface import POP
from omuse.units import units,constants

from iemic_grid import depth_array,depth_levels

from amuse.io import write_set_to_file, read_set_from_file

def simple_upscale(x, fx, fy):
  # scale 2d x with integer factors
  return numpy.kron(x, numpy.ones((fx,fy)))

def plot_forcings_and_depth(p):
  pyplot.figure()
  val=p.nodes.depth.value_in(units.km).T
  mask=(val==0)
  val=numpy.ma.array(val, mask=mask)
  pyplot.imshow(val, origin="lower")
  cbar=pyplot.colorbar()
  cbar.set_label("depth (km)")
  pyplot.savefig("depth.png")

  pyplot.figure()
  val=p.forcings.tau_x.value_in(units.Pa).T
  val=numpy.ma.array(val, mask=mask)  
  pyplot.imshow(val, origin="lower")
  cbar=pyplot.colorbar()
  cbar.set_label("wind stress (Pa)")
  pyplot.savefig("taux.png")
  
  pyplot.figure()
  val=p.element_forcings.restoring_temp.value_in(units.Celsius).T
  val=numpy.ma.array(val, mask=mask)
  pyplot.imshow(val, origin="lower")
  cbar=pyplot.colorbar()
  cbar.set_label("restoring T (C)")
  pyplot.savefig("restoring_temp.png")
  
  pyplot.figure()
  val=p.element_forcings.restoring_salt.value_in(units.g/units.kg).T
  val=numpy.ma.array(val, mask=mask)  
  pyplot.imshow(val, origin="lower")
  cbar=pyplot.colorbar()
  cbar.set_label("restoring salt (psu)")
  pyplot.savefig("restoring_salt.png")

def initialize_pop(depth_levels, depth_array, mode='96x120x12', number_of_workers=4,
       latmin=-90 | units.deg, latmax=90 | units.deg):

  print(f"initializing POP mode {mode} with {number_of_workers} workers")

  p=POP(number_of_workers=number_of_workers, mode=mode, redirection="none")#, job_scheduler="slurm") #, channel_type="sockets" 

  cwd=os.getcwd()

  dz=depth_levels[1:]-depth_levels[:-1]
  #~ print(f"dz: {dz}")

  p.parameters.topography_option='amuse' 
  p.parameters.depth_index=depth_array
  p.parameters.horiz_grid_option='amuse'
  p.parameters.lonmin=0 | units.deg
  p.parameters.lonmax=360 | units.deg
  p.parameters.latmin=latmin
  p.parameters.latmax=latmax
  p.parameters.vert_grid_option='amuse'     
  p.parameters.vertical_layer_thicknesses=dz
  p.parameters.surface_heat_flux_forcing='amuse'
  p.parameters.surface_freshwater_flux_forcing='amuse'
  
  #~ print(p.nodes[0,0].lon.in_(units.deg))
  #~ print(p.nodes[0,0].lat.in_(units.deg))
  #~ print(p.nodes[-1,-1].lon.in_(units.deg))
  #~ print(p.nodes[-1,-1].lat.in_(units.deg))
  #~ print()
  #~ print(p.nodes[0,:].lat.in_(units.deg))


  #~ print()
  #~ print(p.elements[0,0].lon.in_(units.deg))
  #~ print(p.elements[0,0].lat.in_(units.deg))
  #~ print(p.elements[-1,-1].lon.in_(units.deg))
  #~ print(p.elements[-1,-1].lat.in_(units.deg))
  #~ print()
  #~ print(p.elements[0,:].lat.in_(units.deg))

  #~ print()
  #~ print((p.nodes[1,1].position-p.nodes[0,0].position).in_(units.deg))
  #~ print((p.elements[1,1:].position-p.elements[0,:-1].position).in_(units.deg))

  return p

def evolve(p, tend=10| units.day, dt=1. | units.day):

    tnow=p.model_time
    tend=tnow+tend
  
    while tnow< tend-dt/2:
        
        p.evolve_model(tnow+dt)
        tnow=p.model_time
        
        t = tnow.value_in(units.day)
        t = int(t)
        print("evolve to" ,t)

def plot_sst(p):
    pyplot.figure()
    pyplot.imshow(p.elements.temperature.value_in(units.Celsius).T, origin="lower", vmin=0, vmax=30)
    pyplot.colorbar()
    pyplot.savefig("sst.png")

def plot_ssh(nodes, label="ssh"):
    pyplot.figure()
    pyplot.imshow(nodes.ssh.value_in(units.m).T, origin="lower", vmin=-1, vmax=1)
    pyplot.colorbar()
    pyplot.savefig(label+".png")

def plot_grid(p):
    
    lon=p.nodes.lon.value_in(units.deg).flatten()
    lat=p.nodes.lat.value_in(units.deg).flatten()
    pyplot.figure()
    pyplot.plot(lon,lat, 'r+')

    lon=p.elements.lon.value_in(units.deg).flatten()
    lat=p.elements.lat.value_in(units.deg).flatten()
    pyplot.plot(lon,lat, 'g.')

    pyplot.savefig("grid.png")

def save_pop_state(p, label, directory="./"):
    if not os.path.exists(directory):
      os.mkdir(directory)
  
    for d in p.data_store_names():
        #~ print(d,getattr(p, d))
        write_set_to_file(getattr(p, d), os.path.join(directory, label+"_"+d+".amuse"),"amuse", overwrite_file=True)

def reset_pop_state(p,label, snapdir="snapshots"):
    nodes=read_set_from_file(os.path.join(snapdir, label+"_nodes.amuse"),"amuse")
    nodes3d=read_set_from_file(os.path.join(snapdir, label+"_nodes3d.amuse"),"amuse")
    elements=read_set_from_file(os.path.join(snapdir, label+"_elements.amuse"),"amuse")
    elements3d=read_set_from_file(os.path.join(snapdir, label+"_elements3d.amuse"),"amuse")

    channel1=nodes.new_channel_to(p.nodes)
    #~ channel1.copy_attributes(["gradx","grady", "vx_barotropic", "vy_barotropic"])
    channel1.copy_attributes(["vx_barotropic", "vy_barotropic"])

    channel2=nodes3d.new_channel_to(p.nodes3d)
    channel2.copy_attributes(["xvel","yvel"])

    channel3=elements3d.new_channel_to(p.elements3d)
    #~ channel3.copy_attributes(["rho", "salinity", "temperature"])
    channel3.copy_attributes(["salinity", "temperature"])

    channel1=elements.new_channel_to(p.elements)
    channel1.copy_attributes(["ssh"])

def long_evolve(p,tend=100. | units.yr, dt=100. | units.day, dt2=1. | units.day, i0=0, snapdir="snapshots"):
    
    tnow=p.model_time
    tend=tnow+tend

    t1=time.time()
  
    i=i0
    while tnow< tend-dt/2:
        tnow=p.model_time
        tnext=tnow+dt
        
        # save
        label="state_{0:06}".format(i)
        save_pop_state(p, label, directory=snapdir)
        label=label+".1"
        p.evolve_model(tnow+dt2)
        save_pop_state(p, label, directory=snapdir)
        i=i+1

        t2=time.time()

        if tnow>0*tnow:
          eta=(tend-tnow)/(tnow/(t2-t1))
        else:
          eta=999999

        print((t2-t1)/3600, "| evolve to" ,tnext.in_(units.yr), " ETA (hr):", eta/3600.)
        p.evolve_model(tnext)

def long_restart(p, ibegin, tend=100. | units.yr, dt=100. | units.day, loaddir="snapshots", snapdir="snapshots"):
    label="state_{0:06}".format(ibegin)
    tsnap=ibegin*dt
    reset_pop_state(p, label, snapdir=loaddir)
    long_evolve(p,tend=tend-tsnap, dt=dt, i0=ibegin, snapdir=snapdir) 
    
def evolve_test(p):
  
    tend1=10. | units.day
    label1="10"

    t1=time.time()

    evolve(p, tend=tend1)
    save_pop_state(p, label1)

    t2=time.time()

    print("time:", t2-t1)

def restart_test(label, tend):
    p=initialize_pop()
    
    reset_pop_state(p, label)
    #~ save_pop_state(p, "restart")

    t1=time.time()

    evolve(p, tend)
    save_pop_state(p, "restart")

    t2=time.time()

    p.stop()

    print("time:", t2-t1)
    
def compare(label1="201", label2="restart"):

    nodes1=read_set_from_file(label1+"_nodes.amuse","amuse")
    nodes2=read_set_from_file(label2+"_nodes.amuse","amuse")
    
    plot_ssh(nodes1, "ssh_"+label1)
    plot_ssh(nodes2, "ssh_"+label2)

if __name__=="__main__":
    pass
    #~ evolve_test()
    #~ restart_test("200", 1 | units.day)
    #~ compare(label1="201")
    #~ p=initialize_pop(8)
    #~ long_evolve(p, tend=5000 | units.yr, dt=1000. | units.day)
    #~ long_restart(p,1068, tend=5000. | units.yr, dt=1000. | units.day, loaddir="snapshots", snapdir="snapshots2")
    
  
