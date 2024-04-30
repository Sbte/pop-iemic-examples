Examples of POP and IEMIC runs
==============================

This repository contains OMUSE run scripts that illustrate various 
couplings between the POP ocean model and the I-EMIC implicit ocean model.
In addition to this it has a number of utility scripts for plotting and 
setting up the models. Data for a number of I-EMIC grids is included as well
as an initial POP parameter namelist.

Prerequisites
-------------

In order to run the examples OMUSE must be installed with the I-EMIC and POP community codes enabled.

Run files
---------

  - `run_iemic_continuation.py`: Continuation run for I-EMIC. The
  final equilibrium solution is written out.
  - `run_pop.py`: sets up a configuration for POP where both the grid
  and forcings are taken from the I-EMIC (default) setup. The long
  term evolution should be very close to the I-EMIC equilibrium
  solution.
  - `run_pop_iemic.py`: Restart POP from a I-EMIC equilibrium state.
  - `run_pop_pop.py`: Restart POP from a POP state at a different
  resolution.
  - `run_pop_restart.py`: Restart POP from a previous POP state.

Utility files
-------------

  - `iemic.py`: Utility functions for interfacing with the OMUSE
    I-EMIC interface and I-EMIC state files.
  - `pop.py`: Utility functions for interfacing with the OMUSE POP
  interface and POP state files.
  - `pop_iemic.py`: Utility functions that implement the coupling
    between POp and I-EMIC.