import matplotlib.pyplot as pl
import numpy as np
import json

def plot_timings(json_file):

    with open(json_file, 'r') as file:
        cf = json.load(file)
    n = len(cf)
    
    time = np.array([cf[i]['time'] for i in range(n)])
    
    if 'block_size_x' in cf[0]:
        block_x = np.array([cf[i]['block_size_x'] for i in range(n)])
    else:
        block_x = np.ones(n, dtype=int)
    
    if 'block_size_y' in cf[0]:
        block_y = np.array([cf[i]['block_size_y'] for i in range(n)])
    else:
        block_y = np.ones(n, dtype=int)
    
    if 'block_size_z' in cf[0]:
        block_z = np.array([cf[i]['block_size_z'] for i in range(n)])
    else:
        block_z = np.ones(n, dtype=int)
    
    block_tot = block_x * block_y * block_z

    # Find best config:
    i = time.argmin()
    x_best = block_x[i]
    y_best = block_y[i]
    z_best = block_z[i]
    
    # Plot!
    pl.figure(figsize=(6,4))
    ax=pl.subplot(111)
    pl.title(json_file, loc='left')
    pl.scatter(block_tot, time, marker='x', label='Best blocksize = ({}x{}x{})'.format(
        x_best, y_best, z_best))
    pl.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pl.xlabel('Total block size')
    pl.ylabel('Timing kernel')
    pl.tight_layout()

    pl.savefig(json_file[:-5] + "_plot.png", dpi=150, format="png")

    pl.show()


def plot_xy(json_file):

    with open(json_file, 'r') as file:
        cf = json.load(file)
    n = len(cf)
    time = np.array([cf[i]['time'] for i in range(n)])

    block_x = np.array([cf[i]['block_size_x'] for i in range(n)])
    block_y = np.array([cf[i]['block_size_y'] for i in range(n)])

    time = 1e9 / time

    # Plot!
    pl.figure(figsize=(6,4))
    ax=pl.subplot(111)
    pl.title(json_file, loc='left')
    pl.scatter(block_x, block_y, c=time)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    pl.xlabel('block_size_x')
    pl.ylabel('block_size_y')
    pl.tight_layout()

    pl.savefig(json_file[:-5] + "_xyplot.png", dpi=150, format="png")

    pl.show()






if __name__ == '__main__':
    #plot_timings('timings_interpolation_kernel.json')
    #plot_timings('timings_sw_2stream_kernel.json')
    plot_timings('timings_compute_tau_minor.json')
    plot_xy('timings_compute_tau_minor.json')
