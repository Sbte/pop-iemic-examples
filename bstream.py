import numpy

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


def barotropic_streamfunction(u, dz, dy):
    """
    calculate barotropic stream function
    
    u: longitude velocity, 3 dim array(lon,lat, z)
    dz: z layer height (possibly array)
    dy: lattitude (physical) cell size (possibly array)
    """
    if len(u.shape)!=3:
        raise Exception("u dim !=3")
                
    # depth integration
    uint=(u*dz).sum(axis=-1)
    # lattitude integration (note the flip)
    uint=(uint*dy)[:,::-1].cumsum(axis=-1)[:,::-1]
    
    psib=numpy.zeros( (u.shape[0],u.shape[1]+1))*uint[0,0]
    psib[:,1:]=-uint
    return psib
    
def overturning_streamfunction(v, dz, dx):
    """ 
    calculate meriodional overturning streamfunction
    
    v: lattitudinal velocity, 3 dim array (lon, lat, z)
    dz: z layer height (possibly array)
    dx: longitudinal cell size (probably array for lattitude dependend)
    """
    if len(v.shape)!=3:
        raise Exception("v dim !=3")

    #integrate over longitude
    vint=(v.transpose((0,2,1))*dx).transpose((0,2,1))
    vint=vint.sum(axis=0)
    
    #depth integration
    vint=(vint*dz).cumsum(axis=-1)
    
    psim=numpy.zeros( (v.shape[1], v.shape[2]+1))*vint[0,0]
    psim[:,1:]=-vint
    
    return psim
    
     
     
