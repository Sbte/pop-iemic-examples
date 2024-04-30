import os

import iemic

from fvm import Continuation, utils


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


def get_labels(snapdir):
    files = sorted(os.listdir(snapdir), reverse=True)
    label = None
    last_label = None
    for f in files:
        if f.endswith("xml"):
            label = f.split("_")[0]
            try:
                if last_label:
                    float(label)
                    float(last_label)
                    break
            except ValueError:
                pass

            last_label = label

    return last_label, label


def get_dx(instance, label, prev_label, snapdir, parameter_name):
    iemic.load_iemic_state(instance, prev_label, snapdir, load_parameters=False)
    x_prev = instance.get_state().copy()
    mu_prev = instance.get_parameter(parameter_name)

    iemic.load_iemic_state(instance, label, snapdir, load_parameters=False)
    x = instance.get_state().copy()
    mu = instance.get_parameter(parameter_name)

    dx = x - x_prev
    dmu = mu - mu_prev

    print("dx norm", utils.norm(dx), flush=True)
    print("dmu", dmu, flush=True)

    return dx, dmu


def run_continuation(target=1.0):
    instance = iemic.initialize_global_iemic(6)

    dmu = None
    dx = None

    parameter_name = "Ocean->THCM->Starting Parameters->Combined Forcing"

    snapdir = f"idealized_{iemic.Nx}x{iemic.Ny}x{iemic.Nz}"
    if not os.path.exists(snapdir):
        os.mkdir(snapdir)

        print("starting")
    else:
        label, prev_label = get_labels(snapdir)

        print(f"Reading states from {label} and {prev_label}", flush=True)

        if label:
            # Load the state to set the parameters correctly
            iemic.load_iemic_state(instance, label, snapdir)

            dx, dmu = get_dx(instance, label, prev_label, snapdir, parameter_name)

            print("restarting")

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
    parameters["Postprocess"] = lambda instance, x, mu: postprocess(instance, x, mu, snapdir)
    continuation = Continuation(instance, parameters)

    print("start continuation, this may take a while")

    ds = 0.005
    start = instance.get_parameter(parameter_name)

    if start == 0:
        postprocess(instance, x, start, snapdir)

    x, mu = continuation.continuation(x, parameter_name, start, target, ds, dmu=dmu, dx=dx)

    iemic.save_iemic_state(instance, f"idealized_{iemic.Nx}x{iemic.Ny}x{iemic.Nz}")

    print("continuation done")

    instance.stop()

    state = iemic.read_iemic_state_with_units(f"idealized_{iemic.Nx}x{iemic.Ny}x{iemic.Nz}")

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
