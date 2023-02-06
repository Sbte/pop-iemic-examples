from omuse.units import units

import pop_iemic
import pop


if __name__ == "__main__":
    pop_interface = pop_iemic.initialize_pop_with_iemic_setup()

    # pop.plot_barotropic_streamfunction(pop_interface, "bstream-pop-iemic.eps")
    # pop.plot_overturning_streamfunction(pop_interface, "mstream-pop-iemic.eps")
    # pop.plot_salinity(pop_interface, "salinity-pop-iemic.eps")
    # pop.plot_temperature(pop_interface, "temperature-pop-iemic.eps")

    pop.plot_ssh(pop_interface)
    pop.plot_sst(pop_interface)
    pop.plot_velocity(pop_interface)
    pop.plot_salinity(pop_interface)
    pop.plot_temperature(pop_interface)
    pop.plot_surface_salinity(pop_interface)
    pop.plot_surface_temperature(pop_interface)
    pop.plot_streamplot(pop_interface)
    pop.plot_forcings_and_depth(pop_interface)

    tend = 10
    pop.evolve_test(pop_interface, tend)
    # pop.long_evolve(pop_interface, tend=tend | units.yr, dt=100. | units.day)
    # pop.long_evolve(pop_interface, tend=tend | units.yr, dt=1000. | units.day)

    pop.plot_ssh(pop_interface, "ssh_" + str(tend))
    pop.plot_sst(pop_interface, "sst_" + str(tend))
    pop.plot_velocity(pop_interface, "velocity_" + str(tend) + ".eps")
    pop.plot_salinity(pop_interface, "salinity_" + str(tend) + ".eps")
    pop.plot_temperature(pop_interface, "temperature_" + str(tend) + ".eps")
    pop.plot_surface_salinity(pop_interface, "surface_salinity_" + str(tend) + ".eps")
    pop.plot_surface_temperature(pop_interface, "surface_temperature_" + str(tend) + ".eps")
    pop.plot_streamplot(pop_interface, "streamplot_" + str(tend) + ".eps")
    pop.plot_forcings_and_depth(pop_interface, "pop_" + str(tend))

    pop.plot_barotropic_streamfunction(pop_interface, "bstream-pop-iemic-" + str(tend) + ".eps")
    pop.plot_overturning_streamfunction(pop_interface, "mstream-pop-iemic-" + str(tend) + ".eps")
    pop.plot_salinity(pop_interface, "salinity-pop-iemic-" + str(tend) + ".eps")
    pop.plot_temperature(pop_interface, "temperature-pop-iemic-" + str(tend) + ".eps")
