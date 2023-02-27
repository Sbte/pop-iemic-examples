import sys
import os

import pop_iemic
import pop

if __name__ == '__main__':
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

    pop.plot_tdata(directory)

    pop.plot_ssh(pop_instance, os.path.join(directory, "ssh"))
    pop.plot_sst(pop_instance, os.path.join(directory, "sst"))
    pop.plot_velocity(pop_instance, os.path.join(directory, "velocity.eps"))
    pop.plot_salinity(pop_instance, os.path.join(directory, "salinity.eps"))
    pop.plot_temperature(pop_instance, os.path.join(directory, "temperature.eps"))
    pop.plot_surface_salinity(pop_instance, os.path.join(directory, "surface_salinity.eps"))
    pop.plot_surface_temperature(pop_instance, os.path.join(directory, "surface_temperature.eps"))
    pop.plot_streamplot(pop_instance, os.path.join(directory, "streamplot.eps"))
    pop.plot_forcings_and_depth(pop_instance, os.path.join(directory, "pop"))

    pop.plot_barotropic_streamfunction(pop_instance, os.path.join(directory, "bstream.eps"))
    pop.plot_overturning_streamfunction(pop_instance, os.path.join(directory, "mstream.eps"))
    pop.plot_amoc(pop_instance, os.path.join(directory, "amoc.eps"))
    pop.plot_salinity(pop_instance, os.path.join(directory, "salinity.eps"))
    pop.plot_temperature(pop_instance, os.path.join(directory, "temperature.eps"))

    print('Plots were generated in ' + directory + ' with label ' + label)
