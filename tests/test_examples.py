import os


def test_iemic_continuation():
    from example_iemic_continuation import run_continuation

    os.chdir('tests')
    run_continuation(0.0001)
    os.chdir('..')


def test_pop():
    from example_pop import run

    os.chdir('tests')
    run(1)
    os.chdir('..')
