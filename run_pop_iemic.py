from omuse.units import units

import pop_iemic
import pop


def run(tend=10 | units.day, dt=1 | units.day):
    pop_instance = pop_iemic.initialize_pop_with_iemic_setup(6)

    # pop.plot_barotropic_streamfunction(pop_instance, "bstream-pop-iemic.eps")
    # pop.plot_overturning_streamfunction(pop_instance, "mstream-pop-iemic.eps")
    # pop.plot_salinity(pop_instance, "salinity-pop-iemic.eps")
    # pop.plot_temperature(pop_instance, "temperature-pop-iemic.eps")

    pop.plot_ssh(pop_instance)
    pop.plot_velocity(pop_instance)
    pop.plot_salinity(pop_instance)
    pop.plot_temperature(pop_instance)
    pop.plot_surface_salinity(pop_instance)
    pop.plot_surface_temperature(pop_instance)
    pop.plot_streamplot(pop_instance)
    pop.plot_forcings_and_depth(pop_instance)

    # dt = 10 | units.yr
    # tend = 5000 | units.yr
    pop.long_evolve(pop_instance, tend=tend, dt=dt)

    pop.plot_ssh(pop_instance, "ssh_" + str(tend) + ".eps")
    pop.plot_velocity(pop_instance, "velocity_" + str(tend) + ".eps")
    pop.plot_salinity(pop_instance, "salinity_" + str(tend) + ".eps")
    pop.plot_temperature(pop_instance, "temperature_" + str(tend) + ".eps")
    pop.plot_surface_salinity(pop_instance, "surface_salinity_" + str(tend) + ".eps")
    pop.plot_surface_temperature(pop_instance, "surface_temperature_" + str(tend) + ".eps")
    pop.plot_streamplot(pop_instance, "streamplot_" + str(tend) + ".eps")
    pop.plot_forcings_and_depth(pop_instance, "pop_" + str(tend))

    pop.plot_barotropic_streamfunction(pop_instance, "bstream-pop-iemic-" + str(tend) + ".eps")
    pop.plot_overturning_streamfunction(pop_instance, "mstream-pop-iemic-" + str(tend) + ".eps")
    pop.plot_salinity(pop_instance, "salinity-pop-iemic-" + str(tend) + ".eps")
    pop.plot_temperature(pop_instance, "temperature-pop-iemic-" + str(tend) + ".eps")


if __name__ == "__main__":
    run()
