import numpy

from omuse.units import units


def barotropic_streamfunction(u, dz, dy):
    """
    Calculate barotropic stream function

    u: longitudinal velocity, 3 dim array(lon, lat, z)
    dz: z layer height (possibly array)
    dy: latitude (physical) cell size (possibly array)
    """
    if len(u.shape) != 3:
        raise Exception("u dim != 3")

    # depth integration
    uint = (u * dz).sum(axis=-1)
    # lattitude integration (note the flip)
    uint = (uint * dy)[:, ::-1].cumsum(axis=-1)[:, ::-1]

    psib = numpy.zeros((u.shape[0], u.shape[1] + 1)) * uint[0, 0]
    psib[:, 1:] = uint

    return psib


def overturning_streamfunction(v, dz, dx):
    """
    Calculate meriodional overturning streamfunction

    v: latitudinal velocity, 3 dim array (lon, lat, z)
    dz: z layer height (possibly array)
    dx: longitudinal cell size (probably array for lattitude dependent)
    """
    if len(v.shape) != 3:
        raise Exception("v dim != 3")

    # integrate over longitude
    vint = (v.transpose((0, 2, 1)) * dx).transpose((0, 2, 1))
    vint = vint.sum(axis=0)

    # depth integration
    vint = (vint * dz).cumsum(axis=-1)

    psim = numpy.zeros((v.shape[1], v.shape[2] + 1)) * vint[0, 0]
    psim[:, 1:] = vint

    return psim


def read_global_mask(filename):
    mask = None
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith('%'):
                Nx, Ny, Nz, level = [int(i) for i in line.split(' ')[1:]]

                j = 0
                if mask is None:
                    mask = numpy.zeros((Nx + 2, Ny + 2, Nz + 2), dtype="int")

                continue

            mask[:, j, level - 1] = numpy.array([int(d) for d in line[:-1]])  # ignore newline
            j += 1

    mask = mask[1:-1, 1:-1, 1:-1]  # ignore edges

    return mask[:, ::-1, :]  # reorient


def depth_levels(N, stretch_factor=1.8):
    z = numpy.arange(N) / (1.0 * (N - 1))
    if stretch_factor == 0:
        return z
    else:
        return 1 - numpy.tanh(stretch_factor * (1 - z)) / numpy.tanh(stretch_factor)


def compute_depth_index_from_mask(mask):
    levels = depth_levels(13) * 5000 | units.m

    depth = numpy.zeros((mask.shape[0], mask.shape[1] + 2), dtype=int)
    for k in range(mask.shape[2]):
        for j in range(mask.shape[1]):
            for i in range(mask.shape[0]):
                if not depth[i, j+1]:
                    if mask[i, j, k] == 0:
                        depth[i, j+1] = 12 - k

    return levels, depth
