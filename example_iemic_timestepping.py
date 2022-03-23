from iemic import initialize_global_iemic, timestepper

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
    
    instance.evolve_model( 100 ) # days?
    
    x=instance.get_state()

    write_set_to_file(x.grid, "global_96x38x12.amuse","amuse", overwrite_file=True)

    print("timestepping done")

    instance.stop()
        
    print("done")
