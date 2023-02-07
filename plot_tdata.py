import sys

import pop

if __name__ == '__main__':
    if len(sys.argv) < 2:
        pop.plot_tdata()
    else:
        pop.plot_tdata(sys.argv[1])
