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

    print("done")


if __name__ == "__main__":
    run_continuation()
