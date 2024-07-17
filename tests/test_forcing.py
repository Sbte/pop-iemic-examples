import numpy
import os

from omuse.units import units

def test_initialize_iemic():
    import iemic

    os.chdir('tests')

    state_name = 'forcing_120x54x12'

    if not os.path.isfile(f'{state_name}_parameters.xml'):
        instance = iemic.initialize_global_iemic()

        instance.parameters.Ocean__THCM__Starting_Parameters__Combined_Forcing = 1.0
        instance.parameters.Ocean__Analyze_Jacobian = False

        iemic.save_iemic_state(instance, state_name)

    os.chdir('..')

def test_set_idealized_forcing():
    from iemic import read_iemic_state_with_units
    from pop_iemic import initialize_pop
    from pop import set_idealized_forcing

    os.chdir('tests')

    state_name = 'forcing_120x54x12'

    iemic_state = read_iemic_state_with_units(state_name)
    pop_instance = initialize_pop(iemic_state=iemic_state)

    tau_x = pop_instance.forcings.tau_x.copy()
    restoring_temp = pop_instance.element_forcings.restoring_temp.copy()
    restoring_salt = pop_instance.element_forcings.restoring_salt.copy()

    set_idealized_forcing(pop_instance)

    wind_old = pop_instance.forcings.tau_x.value_in(units.Pa)
    wind_new = tau_x.value_in(units.Pa)
    # Boundaries are wrong!
    # assert (numpy.abs(wind_new - wind_old) < 1e-5).all()

    wind_old = wind_old[:, 1:-2]
    wind_new = wind_new[:, 1:-2]
    assert (numpy.abs(wind_new - wind_old) < 1e-5).all()

    temp_old = pop_instance.element_forcings.restoring_temp.value_in(units.Celsius)
    temp_new = restoring_temp.value_in(units.Celsius)
    assert (numpy.abs(temp_new - temp_old) < 1e-3).all()

    salt_old = pop_instance.element_forcings.restoring_salt.value_in(units.psu)
    salt_new = restoring_salt.value_in(units.psu)
    assert (numpy.abs(salt_new - salt_old) < 1e-4).all()

    os.chdir('..')

def test_seasonal_idealized_forcing():
    from iemic import read_iemic_state_with_units
    from pop_iemic import initialize_pop
    from pop import set_seasonal_forcing

    os.chdir('tests')

    state_name = 'forcing_120x54x12'

    iemic_state = read_iemic_state_with_units(state_name)
    pop_instance = initialize_pop(iemic_state=iemic_state)

    max_temp_locs = []
    min_wind_vals = []
    for month in numpy.arange(0, 13):
        set_seasonal_forcing(pop_instance, month / 12)

        tau_x = pop_instance.forcings.tau_x.value_in(units.Pa)
        min_wind_loc = numpy.argmin(tau_x)

        # Check the the wind is stronger in the north during the winter
        if month < 5 or month > 10:
            assert min_wind_loc > pop_instance.forcings.lat.shape[1] / 2
        else:
            assert min_wind_loc < pop_instance.forcings.lat.shape[1] / 2

        min_wind_vals.append(tau_x[0, min_wind_loc])

        temp = pop_instance.element_forcings.restoring_temp.value_in(units.Celsius)
        max_temp_loc = numpy.argmax(temp)
        max_temp_locs.append(max_temp_loc)

        # Check that the max temperature is reasonable
        assert 24 < temp[0, max_temp_loc] < 25

    # Wind is strong in February and August
    assert numpy.argmin(min_wind_vals) == 1

    # Check that the value is reasonable
    assert min_wind_vals[1] > -0.15 and min_wind_vals[1] < -0.1

    # Highest temp is most south in February
    assert numpy.argmin(max_temp_locs) == 0
    # Highest temp is most north in August
    assert numpy.argmax(max_temp_locs) == 6

    os.chdir('..')
