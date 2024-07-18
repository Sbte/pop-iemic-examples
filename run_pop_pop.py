import sys
import os

from omuse.units import units

import pop_iemic
import pop
import utils


def run(tend=10 | units.day, dt=1 | units.day, argv=[]):
    if len(argv) < 2:
        directory = 'snapshots'
    else:
        directory = argv[1]

    if len(argv) < 3:
        files = sorted(os.listdir(directory), reverse=True)
        for f in files:
            if f.startswith('state_'):
                label = f[:12]
                break
    else:
        label = "state_{0:06}".format(int(argv[2]))

    print('Loading from ' + directory + ' with start label ' + label)

    snapdir = directory + '-2'

    mask = utils.read_global_mask('mkmask/global_240x108x12.mask')
    pop_instance = pop_iemic.initialize_pop_with_pop_setup(
        6, label=label, snapdir=directory, iemic_mask=mask)

    pop.plot_ssh(pop_instance)
    pop.plot_velocity(pop_instance)
    pop.plot_surface_salinity(pop_instance)
    pop.plot_surface_temperature(pop_instance)
    pop.plot_streamplot(pop_instance)
    pop.plot_forcings_and_depth(pop_instance)

    pop.plot_barotropic_streamfunction(pop_instance, "bstream-pop-pop.eps")
    pop.plot_overturning_streamfunction(pop_instance, "mstream-pop-pop.eps")
    pop.plot_salinity(pop_instance, "salinity-pop-pop.eps")
    pop.plot_temperature(pop_instance, "temperature-pop-pop.eps")

    # dt = 10 | units.yr
    # tend = 5000 | units.yr
    pop.long_evolve(pop_instance, tend=tend, dt=dt, snapdir=snapdir)

    pop.plot_ssh(pop_instance, "ssh_" + str(tend) + ".eps")
    pop.plot_velocity(pop_instance, "velocity_" + str(tend) + ".eps")
    pop.plot_surface_salinity(pop_instance, "surface_salinity_" + str(tend) + ".eps")
    pop.plot_surface_temperature(pop_instance, "surface_temperature_" + str(tend) + ".eps")
    pop.plot_streamplot(pop_instance, "streamplot_" + str(tend) + ".eps")
    pop.plot_forcings_and_depth(pop_instance, "pop_" + str(tend))

    pop.plot_barotropic_streamfunction(pop_instance, "bstream-pop-pop-" + str(tend) + ".eps")
    pop.plot_overturning_streamfunction(pop_instance, "mstream-pop-pop-" + str(tend) + ".eps")
    pop.plot_salinity(pop_instance, "salinity-pop-pop-" + str(tend) + ".eps")
    pop.plot_temperature(pop_instance, "temperature-pop-pop-" + str(tend) + ".eps")


if __name__ == "__main__":
    run(argv=sys.argv)
