"""
Script to plot the timings from iSAM2, MH-iSAM2 and Hybrid Factor Graphs when running on the City10000 dataset.
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_cumulative_time():
    isam2_times = np.loadtxt("ISAM2_City10000_time.txt",
                             delimiter=",",
                             dtype=float)
    mh_isam_times = np.loadtxt("MH_ISAM2_City10000_time.txt",
                               delimiter=",",
                               dtype=float)
    hybrid_times = np.loadtxt("Hybrid_City10000_time.txt",
                              delimiter=",",
                              dtype=float)
    linewidth = 2

    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    # fig.set_dpi(10)

    N = 4700
    timesteps = np.arange(N)
    plt.plot(timesteps,
             isam2_times[:N],
             linewidth=linewidth,
             label='iSAM2',
             color=(0.1, 0.1, 0.9))
    plt.plot(timesteps,
             mh_isam_times[:N],
             linewidth=linewidth,
             label='MH-iSAM2',
             color=(0.7, 0.1, 0.1))
    plt.plot(timesteps,
             hybrid_times[:N],
             linewidth=linewidth,
             label='Hybrid Factor Graphs',
             color=(0.1, 0.9, 0.1))
    plt.grid(True)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Cumulative time (secs)')
    plt.legend()
    plt.show()


def plot_per_update_time():
    """Main function"""
    isam2_times = np.loadtxt("ISAM2_City10000_timing.txt",
                             delimiter=",",
                             dtype=float)
    mh_isam_times = np.loadtxt("MH_ISAM2_City10000_timing.txt",
                               delimiter=",",
                               dtype=float)
    hybrid_times = np.loadtxt("Hybrid_City10000_timing.txt",
                              delimiter=",",
                              dtype=float)

    linewidth = 2

    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    # fig.set_dpi(10)

    skip = 20
    plt.semilogy(isam2_times[::skip, 0],
                 isam2_times[::skip, 1],
                 linewidth=linewidth,
                 label='iSAM2',
                 color=(0.1, 0.1, 0.9))
    plt.semilogy(mh_isam_times[::skip, 0],
                 mh_isam_times[::skip, 1],
                 linewidth=linewidth,
                 label='MH-iSAM2',
                 color=(0.7, 0.1, 0.1))
    plt.semilogy(hybrid_times[::skip, 0],
                 hybrid_times[::skip, 1],
                 linewidth=linewidth,
                 label='Hybrid Factor Graphs',
                 color=(0.1, 0.9, 0.1))
    plt.grid(True)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Per update time (secs) in log scale')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_per_update_time()
    plot_cumulative_time()
