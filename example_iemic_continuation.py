import os

import iemic

from fvm import Continuation


def postprocess(instance, x, mu, directory):
    cdata = os.path.join(directory, "cdata.txt")
    if mu == 0:
        with open(cdata, "w") as f:
            f.write("")

    iemic.save_iemic_state(instance, "%.4f" % mu, directory)

    state = iemic.read_iemic_state_with_units("%.4f" % mu, directory)

    psib = iemic.barotropic_streamfunction(state)
    psim = iemic.overturning_streamfunction(state)

    psib_min = min(psib.flatten())
    psib_max = max(psib.flatten())

    psim_min = min(psim.flatten())
    psim_max = max(psim.flatten())

    with open(cdata, "a") as f:
        f.write("%.8e %.8e %.8e %.8e %.8e\n" % (mu, psib_min, psib_max, psim_min, psim_max))


def run_continuation(target=1.0):
    instance = iemic.initialize_global_iemic(4)

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

    snapdir = 'idealized_120x54x12'
    if not os.path.exists(snapdir):
        os.mkdir(snapdir)

    postprocess(instance, x, 0, snapdir)

    # setup continuation object
    parameters['Postprocess'] = lambda instance, x, mu: postprocess(instance, x, mu, snapdir)
    continuation = Continuation(instance, parameters)

    print("start continuation, this may take a while")

    ds = 0.005
    x, mu = continuation.continuation(x, "Ocean->THCM->Starting Parameters->Combined Forcing", 0.0, target, ds)

    iemic.save_iemic_state(instance, 'idealized_120x54x12')

    print("continuation done")

    instance.stop()

    state = iemic.read_iemic_state_with_units('idealized_120x54x12')

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
