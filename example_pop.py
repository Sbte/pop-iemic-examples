from omuse.units import units

import pop_iemic
import pop


if __name__ == "__main__":
    pop_instance = pop_iemic.initialize_pop()

    pop.plot_ssh(pop_instance)
    pop.plot_sst(pop_instance)
    pop.plot_velocity(pop_instance)
    pop.plot_salinity(pop_instance)
    pop.plot_temperature(pop_instance)
    pop.plot_surface_salinity(pop_instance)
    pop.plot_surface_temperature(pop_instance)
    pop.plot_streamplot(pop_instance)
    pop.plot_forcings_and_depth(pop_instance)

    tend = 10
    pop.evolve_test(pop_instance, tend)
    # pop.long_evolve(pop_instance, tend=tend | units.yr, dt=100. | units.day)
    # pop.long_evolve(pop_instance, tend=tend | units.yr, dt=1000. | units.day)

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
