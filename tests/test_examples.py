import os
import shutil

from omuse.units import units

def test_iemic_continuation():
    from run_iemic_continuation import run_continuation

    os.chdir('tests')

    directory = 'idealized_120x54x12'
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    run_continuation(0.003)

    assert os.path.isfile(os.path.join(directory, '0.0000_t_grid.amuse'))
    assert os.path.isfile(os.path.join(directory, '0.0030_t_grid.amuse'))
    assert os.path.isfile(os.path.join(directory, 'latest_t_grid.amuse'))

    os.chdir('..')


def test_restart_iemic_continuation():
    from run_iemic_continuation import run_continuation

    os.chdir('tests')

    directory = 'idealized_120x54x12'
    run_continuation(0.006)

    assert os.path.isfile(os.path.join(directory, '0.0000_t_grid.amuse'))
    assert os.path.isfile(os.path.join(directory, '0.0061_t_grid.amuse'))
    assert os.path.isfile(os.path.join(directory, 'latest_t_grid.amuse'))

    os.chdir('..')


def test_pop():
    from run_pop import run

    os.chdir('tests')

    directory = 'snapshots'
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    run(1 | units.day)

    assert os.path.isfile(os.path.join(directory, 'state_000000_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'state_000001_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'latest_nodes3d.amuse'))
    assert not os.path.isfile(os.path.join(directory, 'state_000002_nodes3d.amuse'))

    os.chdir('..')


def test_pop_pop():
    from run_pop_pop import run

    os.chdir('tests')

    directory = 'snapshots-2'

    run(1 | units.day)

    assert os.path.isfile(os.path.join(directory, 'state_000000_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'state_000001_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'latest_nodes3d.amuse'))
    assert not os.path.isfile(os.path.join(directory, 'state_000002_nodes3d.amuse'))

    os.chdir('..')


def test_pop_restart():
    from run_pop_restart import run

    os.chdir('tests')

    directory = 'snapshots-2'
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    run(2 | units.day)

    assert not os.path.isfile(os.path.join(directory, 'state_000000_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'state_000001_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'state_000002_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'latest_nodes3d.amuse'))

    shutil.rmtree('snapshots')
    shutil.rmtree('snapshots-2')
    os.chdir('..')


def test_pop_iemic():
    from run_pop_iemic import run

    os.chdir('tests')

    directory = 'snapshots'
    if os.path.isdir(directory):
        shutil.rmtree(directory)

    run(1 | units.day)

    assert os.path.isfile(os.path.join(directory, 'state_000000_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'state_000001_nodes3d.amuse'))
    assert os.path.isfile(os.path.join(directory, 'latest_nodes3d.amuse'))
    assert not os.path.isfile(os.path.join(directory, 'state_000002_nodes3d.amuse'))

    os.chdir('..')
