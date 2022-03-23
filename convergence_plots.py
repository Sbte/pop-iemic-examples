import os
import numpy


from omuse.units import units,quantities, constants

from amuse.io import read_set_from_file

from matplotlib import pyplot

from bstream import barotropic_streamfunction, overturning_streamfunction, z_from_cellcenterz

def dxi_dt(n, dt=1. | units.day, label="state", directory="snapshots"):
    """ calculate and return approximate average abs residual for snapshot, 
        return drho/dt, dvx/dt, dvy/dt, dT/dt, dS/dt  """

    indexstring=f"{n:06d}"
    elements1=read_set_from_file(os.path.join(directory, label+"_"+indexstring+".1_elements3d.amuse"),"amuse")
    elements=read_set_from_file(os.path.join(directory, label+"_"+indexstring+"_elements3d.amuse"),"amuse")
    
    nodes1=read_set_from_file(os.path.join(directory, label+"_"+indexstring+".1_nodes3d.amuse"),"amuse")
    nodes=read_set_from_file(os.path.join(directory, label+"_"+indexstring+"_nodes3d.amuse"),"amuse")

    drhodt=abs((elements1.rho-elements.rho)/dt).mean()
    dvxdt=abs((nodes1.xvel-nodes.xvel)/dt).mean()
    dvydt=abs((nodes1.yvel-nodes.yvel)/dt).mean()
    dTdt=abs((elements1.temperature-elements.temperature)/dt).mean()
    dSdt=abs((elements1.salinity-elements.salinity)/dt).mean()

    return abs(drhodt).max(),dvxdt,dvydt,dTdt,dSdt


def residual_plot(n,dt):
    """ plot approximate residual (delta X / delta t ) for """

    xi_rho=quantities.AdaptingVectorQuantity()
    xi_vx=quantities.AdaptingVectorQuantity()
    xi_vy=quantities.AdaptingVectorQuantity()
    xi_T=quantities.AdaptingVectorQuantity()
    xi_S=quantities.AdaptingVectorQuantity()
    
    time=numpy.arange(n)*dt
    
    for i in range(n):
      drho,dvx,dvy,dT,dS=dxi_dt(i)
      xi_rho.append(drho)
      xi_vx.append(dvx)
      xi_vy.append(dvy)
      xi_T.append(dT)
      xi_S.append(dS)

    pyplot.semilogy(time.value_in(units.yr), xi_rho/xi_rho[0], label="rho")
    pyplot.semilogy(time.value_in(units.yr), xi_vx/xi_vx[0], label="vx")
    pyplot.semilogy(time.value_in(units.yr), xi_vy/xi_vy[0], label="vy")
    pyplot.semilogy(time.value_in(units.yr), xi_T/xi_T[0], label="T")
    pyplot.semilogy(time.value_in(units.yr), xi_S/xi_S[0],label="S")
    pyplot.legend(loc="upper right")
    pyplot.xlabel("time (yr)")
    pyplot.ylabel("residual (arbitrary units)")
    pyplot.savefig("residual.pdf")

def get_streamfunction_maxs(n, directory="snapshots", label="state"):
    """ calculate and return max of barotropic and meridional stream functions """

    nodes=read_set_from_file(os.path.join(directory, label+"_"+f"{n:06d}"+"_nodes3d.amuse"),"amuse")
       
    n,m,l=nodes.shape   
    zc=nodes[0,0,:].z    
    z=z_from_cellcenterz(zc)
            
    dz=z[1:]-z[:-1]
    dy=constants.Rearth*(nodes[0,1,0].lat-nodes[0,0,0].lat)
    dx=constants.Rearth*(nodes[1,0,0].lon-nodes[0,0,0].lon)
    dx=dx*numpy.cos(nodes[0,:,0].lat.value_in(units.rad))

    psib=barotropic_streamfunction(nodes.xvel,dz,dy)
    psim=overturning_streamfunction(nodes.yvel,dz,dx)

    #~ print(psib.min().value_in(units.Sv),psib.max().value_in(units.Sv))
    #~ print(psim.min().value_in(units.Sv),psim.max().value_in(units.Sv))

    return abs(psib).max(),abs(psim).max()

def streamfunction_plot(n,dt):    
    """ plot abs max of streamfunctions psi_b and psi_m """ 
    
    psi_b=quantities.AdaptingVectorQuantity()
    psi_m=quantities.AdaptingVectorQuantity()
    
    time=numpy.arange(n)*dt
    
    for i in range(n):
      psib,psim=get_streamfunction_maxs(i)
      psi_b.append(psib)
      psi_m.append(psim)

    pyplot.semilogy(time.value_in(units.yr), psi_b.value_in(units.Sv), label="psi_b")
    pyplot.semilogy(time.value_in(units.yr), psi_m.value_in(units.Sv), label="psi_m")
    pyplot.legend(loc="upper right")
    pyplot.xlabel("time (yr)")
    pyplot.ylabel("max of stream function (Sv)")
    pyplot.savefig("max_psi.pdf")



if __name__=="__main__":
    dt=100. | units.day
    n=365
    streamfunction_plot(n,dt)
    #~ residual_plot(n, dt)
      #~ get_streamfunction_maxs(100)
