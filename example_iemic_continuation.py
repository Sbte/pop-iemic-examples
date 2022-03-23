from iemic import initialize_global_iemic

from fvm import Continuation

from omuse.io import write_set_to_file

if __name__=="__main__":
    instance=initialize_global_iemic()

    print("starting")

    # the following line optionally redirects iemic output to file
    #~ instance.set_output_file("output.%p")

    #print out all initial parameters
    print(instance.parameters)
    
    x = instance.get_state()

    # print out actually used THCM parameters
    print(instance.Ocean__THCM__Starting_Parameters)
    

    # numerical parameters for the continuation
    parameters={"Newton Tolerance" : 1.e-2, "Verbose" : True,
                "Minimum Step Size" : 0.001,
                "Maximum Step Size" : 0.2,
                "Delta" : 1.e-6 }

    # setup continuation object
    continuation=Continuation(instance, parameters)
    
    # Converge to an initial steady state
    x = continuation.newton(x, 1e-10)
      
    print("start continuation, this may take a while")

    x, mu, data = continuation.continuation(x, 'Ocean->THCM->Starting Parameters->Combined Forcing', 0., 1., 0.005)

    write_set_to_file(x.grid, "global_96x38x12.amuse","amuse", overwrite_file=True)

    print("continuation done")

    instance.stop()
        
    print("done")
