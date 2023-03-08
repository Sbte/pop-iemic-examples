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

    pop_state = pop.read_pop_state(label, directory)

    pop.plot_tdata(directory, tdata)

    pop.plot_ssh(pop_state, os.path.join(directory, "ssh"))
    pop.plot_sst(pop_state, os.path.join(directory, "sst"))
    pop.plot_velocity(pop_state, os.path.join(directory, "velocity.eps"))
    pop.plot_salinity(pop_state, os.path.join(directory, "salinity.eps"))
    pop.plot_temperature(pop_state, os.path.join(directory, "temperature.eps"))
    pop.plot_surface_salinity(pop_state, os.path.join(directory, "surface_salinity.eps"))
    pop.plot_surface_temperature(pop_state, os.path.join(directory, "surface_temperature.eps"))
    pop.plot_streamplot(pop_state, os.path.join(directory, "streamplot.eps"))
    pop.plot_forcings_and_depth(pop_state, os.path.join(directory, "pop"))

    pop.plot_barotropic_streamfunction(pop_state, os.path.join(directory, "bstream.eps"))
    pop.plot_overturning_streamfunction(pop_state, os.path.join(directory, "mstream.eps"))
    pop_iemic.plot_amoc(pop_state, os.path.join(directory, "amoc.eps"))
    pop.plot_salinity(pop_state, os.path.join(directory, "salinity.eps"))
    pop.plot_temperature(pop_state, os.path.join(directory, "temperature.eps"))

    print('Plots were generated in ' + directory + ' with label ' + label)
