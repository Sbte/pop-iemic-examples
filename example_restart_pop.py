import sys
import os

from omuse.units import units

import pop_iemic
import pop


def run(tend=10 | units.day, dt=1 | units.day):
    if len(sys.argv) < 2:
        directory = 'snapshots'
    else:
        directory = sys.argv[1]

    if len(sys.argv) < 3:
        files = sorted(os.listdir(directory), reverse=True)
        for f in files:
            if f.startswith('state_'):
                label = f[:12]
                break
    else:
        label = "state_{0:06}".format(sys.argv[2])

    pop_instance = pop_iemic.initialize_pop()

    pop.reset_pop_state(pop_instance, label, directory)

    pop.plot_ssh(pop_instance)
    pop.plot_sst(pop_instance)
    pop.plot_velocity(pop_instance)
    pop.plot_salinity(pop_instance)
    pop.plot_temperature(pop_instance)
    pop.plot_surface_salinity(pop_instance)
    pop.plot_surface_temperature(pop_instance)
    pop.plot_streamplot(pop_instance)
    pop.plot_forcings_and_depth(pop_instance)

    print('Running in ' + directory + ' with start label ' + label)

    i0 = int(label[6:])
    snapdir = directory + '-2'

    # dt = 1000 | units.day
    # tend = 1000 | units.yr
    pop.long_evolve(pop_instance, tend=tend, dt=dt, i0=i0, snapdir=snapdir)

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
    run()
