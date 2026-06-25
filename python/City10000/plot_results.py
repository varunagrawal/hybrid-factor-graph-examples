"""
Script to plot City10000 results.
Can be used to plot results from both C++ and python scripts.

This script will additionally print the ATE and RTE for each of the estimates passed in.

Usage:
```
python python/City10000/plot_results.py ISAM2_GT_city10000.txt \
    --estimates results/ISAM2_city10000.txt \
        results/DCSAM_City10000.txt \
        results/MH_ISAM2/MH_ISAM2_City10000.txt \
        results/Hybrid_City10000.txt
```

NOTE: We can pass in as many estimates as we need,
though we also need to pass in the same number of --colors and --labels.

You can generate estimates by running
- `make ISAM2_City10000.run` for the ISAM2 version
- `make Hybrid_City10000.run` for the Hybrid Smoother version

Author: Varun Agrawal
"""

import argparse
import pprint

import numpy as np
from evo.core import metrics, sync, trajectory
from matplotlib import pyplot as plt


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", help="The ground truth data file.")
    parser.add_argument(
        "--estimates",
        nargs="+",
        help="File(s) with estimates (as .txt), can be more than one.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Label to apply to the estimate graph.",
        default=("ISAM2", "DCSAM", "MH-iSAM2", "Hybrid Factor Graphs"),
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        help="The color to apply to each of the estimate graphs.",
        default=(
            (0.9, 0.1, 0.1, 0.4),
            (0.6, 0.3, 0.6, 0.4),
            (0.6, 0.3, 0.6, 0.4),
            (0.6, 0.3, 0.6, 0.4),
            (0.1, 0.1, 0.9, 0.4),
        ),
    )
    return parser.parse_args()


def compute_error_metrics(gt, estimates, label):
    def convert_to_xyz(x, y):
        tx = x
        ty = y
        tz = 0
        return (tx, ty, tz)

    def convert_to_quaternion(theta):
        qx = 0
        qy = 0
        qz = np.sin(theta / 2)
        qw = np.cos(theta / 2)
        return (qw, qx, qy, qz)

    def get_evo_trajectory(data):
        translations = [convert_to_xyz(x, y) for x, y, _ in data]
        rotations = [convert_to_quaternion(theta) for _, _, theta in data]
        timestamps = np.arange(len(translations)).astype(
            float
        )  # Use index as timestamp
        return trajectory.PoseTrajectory3D(
            np.asarray(translations), np.asarray(rotations), timestamps
        )

    gt_trajectory = get_evo_trajectory(gt)
    estimated_trajectory = get_evo_trajectory(estimates)

    gt_sync, est_sync = sync.associate_trajectories(gt_trajectory, estimated_trajectory)
    ape_metric = metrics.APE(metrics.PoseRelation.full_transformation)
    ape_metric.process_data((gt_sync, est_sync))
    ape_stats = ape_metric.get_all_statistics()
    print(f"{label} ATE:")
    pprint.pprint(ape_stats["rmse"])

    rpe_metric = metrics.RPE(metrics.PoseRelation.full_transformation)
    rpe_metric.process_data((gt_sync, est_sync))
    rpe_stats = rpe_metric.get_all_statistics()
    print(f"{label} RTE:")
    pprint.pprint(rpe_stats["rmse"])


def plot_estimates(
    gt,
    estimates,
    fignum: int,
    estimate_color=(0.1, 0.1, 0.9, 0.4),
    estimate_label="Hybrid Factor Graphs",
):
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
    fig = plt.figure(fignum, figsize=(4, 4))
    ax = fig.gca()
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        "--",
        linewidth=1,
        color=(0.1, 0.7, 0.1, 0.5),
        label="Ground Truth",
    )
    ax.plot(
        estimates[:, 0],
        estimates[:, 1],
        "-",
        linewidth=1,
        color=estimate_color,
        label=estimate_label,
    )
    # ax.axis("equal")
    ax.axis((-60.0, 60.0, -70.0, 60.0))
    ax.legend(loc="lower right")


def main():
    """Main runner"""
    args = parse_args()
    gt = np.loadtxt(args.ground_truth, delimiter=" ")

    for i in range(len(args.estimates)):
        h_poses = np.loadtxt(args.estimates[i], delimiter=" ")
        # Limit ground truth to the number of estimates so the plot looks cleaner
        limited_gt = gt[: h_poses.shape[0]]

        compute_error_metrics(limited_gt, h_poses, args.labels[i])

        plot_estimates(
            limited_gt,
            h_poses,
            i + 1,
            estimate_color=args.colors[i],
            estimate_label=args.labels[i],
        )

    plt.show()


if __name__ == "__main__":
    main()
