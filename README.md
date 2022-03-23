Examples of POP and IEMIC runs
==============================

This repository contains example OMUSE run scripts that illustrate various 
couplings between the POP ocean model and the I-EMIC implicit ocean model.
In addition to this it has a number of utility scripts for plotting and 
setting up the models. Data for a number of I-EMIC grids is included as well
as an initial POP parameter namelist.

Prerequisites
-------------

In order to run the examples OMUSE must be installed.

Examples
--------

The examples included:

  - ```example_pop_iemic_setup.py```: sets up a configuration for POP where both the grid
    and forcings are taken from the I-EMIC (default) setup. The long term evolution
    should be very close to the I-EMIC equilibrium solution.
  - ```example_iemic_continuation.py```: Continuation run for I-EMIC. The final equilibrium 
    solution is written out.
  - ```example_iemic_timestepping.py```: Evolve I-EMIC with the included implicit time 
    stepper.
  - ```example_pop_iemic_state.py```: Restart POP from a I-EMIC equilibrium state.
  - ```example_iemic_equilibrium.py```: Calculate equilibrium starting from a non-zero state.

Other files
-----------
