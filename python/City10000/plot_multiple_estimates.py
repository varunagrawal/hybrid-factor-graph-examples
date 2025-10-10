"""
Script to plot City10000 results at different timesteps.

Usage:
```
python plot_results.py ISAM2_GT_city10000.txt \
    --estimates Hybrid_City10000.txt \
    --indices 100 1000 2000 5000 10000
```

NOTE: We can pass in as many estimates as we need,
though we also need to pass in the same number of --colors and --labels.

You can generate estimates by running
- `make ISAM2_City10000.run` for the ISAM2 version
- `make Hybrid_City10000.run` for the Hybrid Smoother version

Author: Varun Agrawal
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", help="The ground truth data file.")
    parser.add_argument(
        "--estimates",
        nargs='+',
        help="File(s) with estimates (as .txt), can be more than one.")
    parser.add_argument("--indices",
                        nargs='+',
                        help="The time indices at which to plot th estimate.",
                        default=(100, 1000, 2000, 5000, 10000))
    return parser.parse_args()


def plot_estimates(gt,
                   estimates,
                   fignum: int,
                   estimate_color=(0.1, 0.1, 0.9, 0.4),
                   estimate_label="Hybrid Factor Graphs"):
    """Plot the City10000 estimates against the ground truth.

    Args:
        gt (np.ndarray): The ground truth trajectory as xy values.
        estimates (np.ndarray): The estimates trajectory as xy values.
        fignum (int): The figure number for multiple plots.
        estimate_color (tuple, optional): The color to use for the graph of estimates.
            Defaults to (0.1, 0.1, 0.9, 0.4).
        estimate_label (str, optional): Label for the estimates, used in the legend.
            Defaults to "Hybrid Factor Graphs".
    """
    fig = plt.figure(fignum)
    ax = fig.gca()
    ax.axis('equal')
    ax.axis((-65.0, 65.0, -75.0, 60.0))
    ax.plot(gt[:, 0],
            gt[:, 1],
            '--',
            linewidth=1,
            color=(0.1, 0.7, 0.1, 0.5),
            label="Ground Truth")
    ax.plot(estimates[:, 0],
            estimates[:, 1],
            '-',
            linewidth=1,
            color=estimate_color,
            label=estimate_label)
    ax.legend()


def main():
    """Main runner"""
    args = parse_args()
    gt = np.loadtxt(args.ground_truth, delimiter=" ")


    for i, time_index in enumerate(args.indices):
        estimate_poses = np.loadtxt(args.estimates[i], delimiter=" ")
        idx = min(int(time_index), estimate_poses.shape[0])

        # Limit ground truth to the number of estimates so the plot looks cleaner
        gt_poses = gt[:idx]

        num_hypothesis = 16
        plot_estimates(gt_poses,
                       estimate_poses,
                       i + 1,
                       estimate_label=f"Hybrid_City_{time_index}_{num_hypothesis}")

    plt.show()


if __name__ == "__main__":
    main()
