import os
import shutil

from omuse.units import units

def test_iemic_continuation():
    from example_iemic_continuation import run_continuation

    os.chdir('tests')

    run_continuation(0.0001)

    os.chdir('..')


def test_pop():
    from example_pop import run

    os.chdir('tests')

    if os.path.isdir('snapshots'):
        shutil.rmtree('snapshots')

    run(1 | units.day)

    assert os.path.isfile('snapshots/state_000000_nodes3d.amuse')
    assert os.path.isfile('snapshots/state_000001_nodes3d.amuse')
    assert os.path.isfile('snapshots/latest_nodes3d.amuse')
    assert not os.path.isfile('snapshots/state_000002_nodes3d.amuse')

    os.chdir('..')


def test_restart_pop():
    from example_restart_pop import run

    os.chdir('tests')

    if os.path.isdir('snapshots-2'):
        shutil.rmtree('snapshots-2')

    run(2 | units.day)

    assert not os.path.isfile('snapshots-2/state_000000_nodes3d.amuse')
    assert os.path.isfile('snapshots-2/state_000001_nodes3d.amuse')
    assert os.path.isfile('snapshots-2/state_000002_nodes3d.amuse')
    assert os.path.isfile('snapshots-2/latest_nodes3d.amuse')

    shutil.rmtree('snapshots')
    shutil.rmtree('snapshots-2')
    os.chdir('..')


def test_pop_iemic_state():
    from example_pop_iemic_state import run

    os.chdir('tests')

    run(1 | units.day)

    os.chdir('..')
