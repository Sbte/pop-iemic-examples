import os
import numpy

from omuse.units import units, constants

from amuse.units import trigo

from omuse.io import read_set_from_file

from bstream import barotropic_streamfunction, overturning_streamfunction, z_from_cellcenterz

from matplotlib import pyplot

def six_panel(x):

    #~ x=read_set_from_file("global_96x38x12.amuse","hdf5")

    n,m,l=x.shape

    lats=x[0,:,0].lat
    lons=x[:,0,0].lon

    alllons=trigo.in_deg(x[...].lon)
    alllats=trigo.in_deg(x[...].lat)
  
    atlantic_mask=((alllons>290)+(alllons<20))*(alllats>-40) # selection mask for atlantic
    
    atlantic_ilon=abs(330-trigo.in_deg(lons)).argmin() # find index for atlantic north-south slice
            
    dlat=x[0,1,0].lat-x[0,0,0].lat
    dlon=x[1,0,0].lon-x[0,0,0].lon
        
    latmin=x.lat.min()-dlat/2
    lonmin=x.lon.min()-dlon/2
    
    lats1=numpy.arange(len(lats)+1)*dlat+latmin
    lons1=numpy.arange(len(lons)+1)*dlon+lonmin
        
    xmin=trigo.in_deg(lons1.min())
    xmax=trigo.in_deg(lons1.max())
    ymin=trigo.in_deg(lats1.min())
    ymax=trigo.in_deg(lats1.max())

    extent=[xmin,xmax,ymin,ymax]
  
    # for barotropic
    xmin1=trigo.in_deg(lons1.min()-dlon/2)
    xmax1=trigo.in_deg(lons1.max()+dlon/2)
    ymin1=trigo.in_deg(lats1.min()-dlat/2)
    ymax1=trigo.in_deg(lats1.max()+dlat/2)

    extent1=[xmin1,xmax1,ymin1,ymax1]

    print("extent of grid:", extent)  
      
    uvel=x[:,:,:].u_velocity
    vvel=x[:,:,:].v_velocity
    
    umax=numpy.abs(uvel).max()
    vmax=numpy.abs(vvel).max()
   
    zc=x[0,0,:].z    
    z=z_from_cellcenterz(zc)
            
    fig, ax=pyplot.subplots(3,2,figsize=(12,9))

    dz=z[1:]-z[:-1]
    dy=constants.Rearth*(x[0,1,0].lat-x[0,0,0].lat)
    dx=constants.Rearth*(x[1,0,0].lon-x[0,0,0].lon)

    mask=x[:,:,-1].mask
    surfacemask=numpy.ma.array(numpy.ones_like(mask),mask=mask==0)
    
    #
    # plot barotropic stream function
    #

    psib=barotropic_streamfunction(x[...].u_velocity | 0.1*units.m/units.s,dz,dy)

    vmax=numpy.abs(psib).max().value_in(units.Sv)

    bmask=numpy.zeros_like(psib.number)
    bmask[:,0]=1
    bmask[:,-1]=1
    bmask[:,1:]+=mask
    bmask[:,:-1]+=mask

    val=psib.value_in(units.Sv)

    val=numpy.ma.array(val, mask=bmask==2)
    
    val=val.T
    
    im=ax[0,0].contourf(trigo.in_deg(lons),trigo.in_deg(lats1),val, cmap="seismic", 
       levels=20, vmax=vmax, vmin=-vmax,corner_mask=False)

    ax[0,0].contour(im, colors='k')

    ax[0,0].pcolormesh(trigo.in_deg(lons1),trigo.in_deg(lats1), surfacemask.T)

    fig.colorbar(im,ax=ax[0,0],label="psib [Sv]")

    ax[0,0].set_title("Barotropic streamfunction")
    ax[0,0].set_xlabel("longitude (deg)")
    ax[0,0].set_ylabel("lattitude (deg)")

    #
    # plot (A)MOC
    #

    dx=constants.Rearth*(x[1,0,0].lon-x[0,0,0].lon)
    dx=dx*numpy.cos(x[0,:,0].lat.value_in(units.rad))

    
    vvel=x[...].v_velocity
    vvel=vvel*atlantic_mask
    vvel=numpy.roll(vvel, -n//2,axis=0)
    vvel=vvel| 0.1*units.m/units.s

    psim=overturning_streamfunction(vvel,dz,dx)

    psim=psim.in_(units.Sv)

    vmax=numpy.abs(psim).max().number#value_in(units.Sv)
    zmin=z.min().value_in(units.m)
    zmax=z.max().value_in(units.m)

    extent2=[ymin,ymax1,zmin/1000,zmax/1000]

    val=psim.number.T
    
    mmask=0.*val
    
    mmask[:, trigo.in_deg(lats)<-40.]=1
    
    val=numpy.ma.array(val,mask=mmask)
    
    im=ax[0,1].contourf(trigo.in_deg(lats),z.value_in(units.km), val, 
      vmax=vmax, vmin=-vmax, cmap="seismic", levels=20)
    ax[0,1].contour(im, colors='k')

    fig.colorbar(im,ax=ax[0,1],label="psim [Sv]")

    ax[0,1].set_title("Atlantic overturning streamfunction")
    ax[0,1].set_xlabel("lattitude (deg)")
    ax[0,1].set_ylabel("depth (km)")


    #
    # plot temperature atlantic slice
    #

    Tslice=x[atlantic_ilon,:,:].temperature + 15

    vmax=Tslice.max()
    vmin=Tslice.min()

    val=Tslice

    tmask=x[atlantic_ilon,:,:].mask
    val=numpy.ma.array(val, mask=tmask==1) 
    val=val.T

    im=ax[1,0].contourf(trigo.in_deg(lats),zc.value_in(units.km), val, vmax=vmax, vmin=vmin,
       cmap="seismic", levels=20)
    ax[1,0].contour(im, colors='k')

    fig.colorbar(im,ax=ax[1,0],label="Temperature [C]")

    ax[1,0].set_title("Atlantic temperature")
    ax[1,0].set_xlabel("lattitude (deg)")
    ax[1,0].set_ylabel("depth (km)")

    #
    # plot surface temperature
    #

    Tsurf=x[:,:,-1].temperature + 15

    vmax=Tsurf.max()
    vmin=Tsurf.min()

    val=Tsurf
    
    val=numpy.ma.array(val, mask=surfacemask==1)
    val=val
    
    im=ax[1,1].contourf(trigo.in_deg(lons),trigo.in_deg(lats), val.T, vmax=vmax, vmin=vmin,
        cmap='seismic', levels=20,corner_mask=False)
    ax[1,1].contour(im, colors='k')

    fig.colorbar(im,ax=ax[1,1],label="Temperature [C]")

    ax[1,1].set_title("Surface temperature")
    ax[1,1].set_xlabel("longitude (deg)")
    ax[1,1].set_ylabel("lattitude (deg)")

    #
    # plot salinity atlantic slice
    #

    Sslice=x[atlantic_ilon,:,:].salinity + 35

    Sslice=numpy.maximum(Sslice,34)

    vmax=Sslice.max()
    vmin=Sslice.min()

    val=Sslice

    val=numpy.ma.array(val, mask=tmask==1) 
    val=val.T

    im=ax[2,0].contourf(trigo.in_deg(lats),zc.value_in(units.km), val, vmax=vmax, vmin=vmin, 
       cmap="seismic", levels=20)
    ax[2,0].contour(im, colors='k')

    fig.colorbar(im,ax=ax[2,0],label="Salinity [psu]")

    ax[2,0].set_title("Atlantic salinity")
    ax[2,0].set_xlabel("lattitude (deg)")
    ax[2,0].set_ylabel("depth (km)")

    #
    # plot surface salinity
    #

    Ssurf=x[:,:,-1].salinity + 35

    Ssurf=numpy.maximum(Ssurf,34)


    vmax=Ssurf.max()
    vmin=Ssurf.min()
    print("vmin, vamx:", vmin,vmax)
    vmax=36
    vmin=34


    val=Ssurf
    val=numpy.ma.array(val, mask=surfacemask==1)

    im=ax[2,1].contourf(trigo.in_deg(lons),trigo.in_deg(lats), val.T, vmax=vmax, vmin=vmin,
        cmap='seismic', levels=20,corner_mask=False)
    ax[2,1].contour(im, colors='k')

    fig.colorbar(im,ax=ax[2,1],label="Salinity [psu]")

    ax[2,1].set_title("Surface salinity")
    ax[2,1].set_xlabel("longitude (deg)")
    ax[2,1].set_ylabel("lattitude (deg)")

    pyplot.tight_layout()

    # save

    pyplot.savefig("all.png")
    pyplot.show()
        
    print("done")

def import_to_iemic_grid(label, directory="."):
    nodes=read_set_from_file(os.path.join(directory, label+"_nodes.amuse"),"amuse")
    elements=read_set_from_file(os.path.join(directory,label+"_elements.amuse"),"amuse")
    nodes3d=read_set_from_file(os.path.join(directory,label+"_nodes3d.amuse"),"amuse")
    elements3d=read_set_from_file(os.path.join(directory,label+"_elements3d.amuse"),"amuse")
        
    depth=elements.depth
    z=nodes3d.z
    
    mask=((depth.T-z.T)>0| units.m).T
    mask=1-mask
    
    x=nodes3d.empty_copy()
    x.mask=mask[:,:,::-1]
    a=x.mask==0
            
    x.salinity=elements3d[:,:,::-1].salinity.value_in(units.psu)
    x[a].salinity=(x[a].salinity-35)
    x.temperature=elements3d[:,:,::-1].temperature.value_in(units.Celsius)-15
    x.u_velocity=nodes3d[:,:,::-1].xvel.value_in(0.1*units.m/units.s)
    x.v_velocity=nodes3d[:,:,::-1].yvel.value_in(0.1*units.m/units.s)
    x.lat=nodes3d[:,:,::-1].lat
    x.lon=nodes3d[:,:,::-1].lon
    x.z=-elements3d[:,:,::-1].z
    
    return x


if __name__=="__main__":
    n=10
    label="state_"+f"{n:06d}"
    #~ label=f"{n}"
    directory="snapshots"
    #~ directory="./"
    x=import_to_iemic_grid(label,directory)
    
    six_panel(x)
    
