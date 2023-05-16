import sys
import os

from omuse.units import units

import pop_iemic
import pop


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
        label = "state_{0:06}".format(argv[2])

    pop_instance = pop_iemic.initialize_pop()

    print('Loading from ' + directory + ' with start label ' + label)

    i0 = int(label[6:])
    snapdir = directory + '-2'

    # dt = 1000 | units.day
    # tend = 1000 | units.yr
    pop.long_restart(pop_instance, i0, tend=tend, dt=dt, loaddir=directory, snapdir=snapdir)

    pop.plot_ssh(pop_instance, "ssh_" + str(tend))
    pop.plot_sst(pop_instance, "sst_" + str(tend))
    pop.plot_velocity(pop_instance, "velocity_" + str(tend) + ".eps")
    pop.plot_salinity(pop_instance, "salinity_" + str(tend) + ".eps")
    pop.plot_temperature(pop_instance, "temperature_" + str(tend) + ".eps")
    pop.plot_surface_salinity(pop_instance, "surface_salinity_" + str(tend) + ".eps")
    pop.plot_surface_temperature(pop_instance, "surface_temperature_" + str(tend) + ".eps")
    pop.plot_streamplot(pop_instance, "streamplot_" + str(tend) + ".eps")
    pop.plot_forcings_and_depth(pop_instance, "pop_" + str(tend))

    pop.plot_barotropic_streamfunction(pop_instance, "bstream-pop-" + str(tend) + ".eps")
    pop.plot_overturning_streamfunction(pop_instance, "mstream-pop-" + str(tend) + ".eps")
    pop.plot_salinity(pop_instance, "salinity-pop-" + str(tend) + ".eps")
    pop.plot_temperature(pop_instance, "temperature-pop-" + str(tend) + ".eps")


if __name__ == '__main__':
    run(argv=sys.argv)
