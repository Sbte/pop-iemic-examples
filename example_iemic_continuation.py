import iemic

from fvm import Continuation


def run_continuation(target=1.0):
    instance = iemic.initialize_global_iemic()

    print("starting")

    # the following line optionally redirects iemic output to file
    # ~ instance.set_output_file("output.%p")

    # print out all initial parameters
    print(instance.parameters)

    x = instance.get_state()

    # print out actually used THCM parameters
    print(instance.Ocean__THCM__Starting_Parameters)

    # numerical parameters for the continuation
    parameters = {
        "Newton Tolerance": 1.0e-2,
        "Verbose": True,
        "Minimum Step Size": 0.001,
        "Maximum Step Size": 0.2,
        "Delta": 1.0e-6,
    }

    # setup continuation object
    continuation = Continuation(instance, parameters)

    print("start continuation, this may take a while")

    ds = 0.005
    x, mu = continuation.continuation(x, "Ocean->THCM->Starting Parameters->Combined Forcing", 0.0, target, ds)

    iemic.save_iemic_state(instance, 'global_state')

    print("continuation done")

    instance.stop()

    state = iemic.read_iemic_state_with_units('global_state')

    iemic.plot_u_velocity(state)
    iemic.plot_v_velocity(state)
    iemic.plot_surface_pressure(state)
    iemic.plot_surface_salinity(state)
    iemic.plot_surface_temperature(state)
    iemic.plot_salinity(state)
    iemic.plot_temperature(state)
    iemic.plot_streamplot(state)
    iemic.plot_barotropic_streamfunction(state)
    iemic.plot_overturning_streamfunction(state)

    print("done")


if __name__ == "__main__":
    run_continuation()
