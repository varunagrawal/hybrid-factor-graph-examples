"""
Functions to plot boxplots for PGO monte carlo results.
"""

import argparse

import gtsam
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from g2o_ate import g2o_ate_rot, g2o_ate_tran
from matplotlib.lines import Line2D

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 16})


def parse_times(fname):
    f = open(fname, "r")
    contents = f.readlines()
    # File format is:
    # DCSAM: {dcsam_time}
    # GNC: {gnc_time}
    # LM: {lm_time}
    # HFG: {hfg_time}
    dcsam_time = float(contents[0].split()[1])
    gnc_time = float(contents[1].split()[1])
    lm_time = float(contents[2].split()[1])
    hfg_time = float(contents[3].split()[1])
    f.close()
    return dcsam_time, gnc_time, lm_time, hfg_time


def get_box_patches(ax):
    # print(ax.patches)
    # plt.yscale("log")
    box_patches = [
        patch for patch in ax.patches if isinstance(patch, matplotlib.patches.PathPatch)
    ]
    # In matplotlib older than 3.5, the boxes are stored in ax2.artists
    if len(box_patches) == 0:
        box_patches = ax.artists

    return box_patches


def customize_plot(ax):
    """Customize the box plot lines and whiskers."""

    box_patches = get_box_patches(ax)

    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        patch.set_facecolor("None")

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot : (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers

    # # Also fix the legend
    # for legpatch in ax.get_legend().get_patches():
    #     col = legpatch.get_facecolor()
    #     legpatch.set_edgecolor(col)
    #     legpatch.set_facecolor('None')

    # # iterate over boxes
    # for i,box in enumerate(ax.artists):
    #      box.set_edgecolor('black')
    #      box.set_facecolor('white')

    #      # iterate over whiskers and median lines
    #      for j in range(6*i,6*(i+1)):
    #          ax.lines[j].set_color('black')
    #          plt.legend()

    return ax


def print_stats(
    outlier_pct,
    hfg_ate_tran,
    hfg_ate_rot,
    dcsam_ate_tran,
    dcsam_ate_rot,
    gnc_ate_tran,
    gnc_ate_rot,
    lm_ate_tran,
    lm_ate_rot,
):
    """Print the statistics for each method and outlier percentage."""
    print(f"Outlier Percentage: {outlier_pct}%")
    print(
        f"HFG: "
        # f"Mean Time = {sum(hfg_times[outlier_pct]) / len(hfg_times[outlier_pct]):.4f}s, "
        f"Mean Translation Error = {sum(hfg_ate_tran[outlier_pct]) / len(hfg_ate_tran[outlier_pct]):.4f}m, "
        f"Mean Rotation Error = {sum(hfg_ate_rot[outlier_pct]) / len(hfg_ate_rot[outlier_pct]):.4f}deg"
    )
    print(
        f"DCSAM: "
        # f"Mean Time = {sum(dcsam_times[outlier_pct]) / len(dcsam_times[outlier_pct]):.4f}s, "
        f"Mean Translation Error = {sum(dcsam_ate_tran[outlier_pct]) / len(dcsam_ate_tran[outlier_pct]):.4f}m, "
        f"Mean Rotation Error = {sum(dcsam_ate_rot[outlier_pct]) / len(dcsam_ate_rot[outlier_pct]):.4f}deg"
    )
    print(
        f"GNC: "
        # f"Mean Time = {sum(gnc_times[outlier_pct]) / len(gnc_times[outlier_pct]):.4f}s, "
        f"Mean Translation Error = {sum(gnc_ate_tran[outlier_pct]) / len(gnc_ate_tran[outlier_pct]):.4f}m, "
        f"Mean Rotation Error = {sum(gnc_ate_rot[outlier_pct]) / len(gnc_ate_rot[outlier_pct]):.4f}deg"
    )
    print(
        f"LM: "
        # f"Mean Time = {sum(lm_times[outlier_pct]) / len(lm_times[outlier_pct]):.4f}s,"
        f" Mean Translation Error = {sum(lm_ate_tran[outlier_pct]) / len(lm_ate_tran[outlier_pct]):.4f}m, "
        f"Mean Rotation Error = {sum(lm_ate_rot[outlier_pct]) / len(lm_ate_rot[outlier_pct]):.4f}deg"
    )


def collect_data(prefix_path, is3D, outlier_pcts=(10, 20, 30, 40)):
    """Collect the data to plot from the different .g2o files."""
    gt_graph, gt_vals = gtsam.readG2o(prefix_path + "0/1/out_nonrobust.g2o", is3D)

    hfg_ate_tran = {}
    hfg_ate_rot = {}

    dcsam_ate_tran = {}
    dcsam_ate_rot = {}

    gnc_ate_tran = {}
    gnc_ate_rot = {}

    lm_ate_tran = {}
    lm_ate_rot = {}

    all_data = pd.DataFrame(
        columns=[
            "Method",
            "Outlier Rate (\\%)",
            "Time (s)",
            "Average Translation Error (m)",
            "Average Rotation Error (deg)",
        ]
    )

    for outlier_pct in outlier_pcts:
        hfg_ate_tran[outlier_pct] = []
        hfg_ate_rot[outlier_pct] = []

        dcsam_ate_tran[outlier_pct] = []
        dcsam_ate_rot[outlier_pct] = []

        gnc_ate_tran[outlier_pct] = []
        gnc_ate_rot[outlier_pct] = []

        lm_ate_tran[outlier_pct] = []
        lm_ate_rot[outlier_pct] = []

        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # Hybrid Factor Graph Processing
            hfg_graph, hfg_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_hfg_robust.g2o", is3D
            )
            hfg_tran = g2o_ate_tran(hfg_est, gt_vals, is3D)
            hfg_rot = g2o_ate_rot(hfg_est, gt_vals, is3D)
            hfg_ate_tran[outlier_pct].append(hfg_tran)
            hfg_ate_rot[outlier_pct].append(hfg_rot)

            # DCSAM Processing
            dcsam_graph, dcsam_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_robust.g2o", is3D
            )
            dcsam_tran = g2o_ate_tran(dcsam_est, gt_vals, is3D)
            dcsam_rot = g2o_ate_rot(dcsam_est, gt_vals, is3D)
            dcsam_ate_tran[outlier_pct].append(dcsam_tran)
            dcsam_ate_rot[outlier_pct].append(dcsam_rot)

            # GNC Processing
            gnc_graph, gnc_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_gnc.g2o", is3D
            )
            gnc_tran = g2o_ate_tran(gnc_est, gt_vals, is3D)
            gnc_rot = g2o_ate_rot(gnc_est, gt_vals, is3D)
            gnc_ate_tran[outlier_pct].append(gnc_tran)
            gnc_ate_rot[outlier_pct].append(gnc_rot)

            # LM Processing
            lm_graph, lm_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_nonrobust.g2o", is3D
            )
            lm_tran = g2o_ate_tran(lm_est, gt_vals, is3D)
            lm_rot = g2o_ate_rot(lm_est, gt_vals, is3D)
            lm_ate_tran[outlier_pct].append(lm_tran)
            lm_ate_rot[outlier_pct].append(lm_rot)

            # Get times
            dcsam_time, gnc_time, lm_time, hfg_time = parse_times(
                prefix_path + f"{outlier_pct}/{seed}/times.txt"
            )

            all_data.loc[len(all_data.index)] = [
                "Hybrid Factor Graph",
                outlier_pct,
                hfg_time,
                hfg_tran,
                hfg_rot,
            ]

            all_data.loc[len(all_data.index)] = [
                "DCSAM",
                outlier_pct,
                dcsam_time,
                dcsam_tran,
                dcsam_rot,
            ]

            all_data.loc[len(all_data.index)] = [
                "GNC",
                outlier_pct,
                gnc_time,
                gnc_tran,
                gnc_rot,
            ]

            all_data.loc[len(all_data.index)] = [
                "LM",
                outlier_pct,
                lm_time,
                lm_tran,
                lm_rot,
            ]

        # Print statistics for each outlier percentage
        print_stats(
            outlier_pct,
            hfg_ate_tran,
            hfg_ate_rot,
            dcsam_ate_tran,
            dcsam_ate_rot,
            gnc_ate_tran,
            gnc_ate_rot,
            lm_ate_tran,
            lm_ate_rot,
        )

    return all_data


def plot_times(dataset_name, all_data, custom_labels, custom_lines, colors):
    fig, ax = plt.subplots(1)
    sns.boxplot(
        x="Outlier Rate (\\%)",
        y="Time (s)",
        hue="Method",
        data=all_data,
        palette=colors,
        ax=ax,
    )

    ax = customize_plot(ax)

    ax.legend(custom_lines, custom_labels)

    plt.savefig(f"{dataset_name}_times.png", dpi=600, bbox_inches="tight")
    plt.show()


def plot_average_translation_error(
    dataset_name, all_data, custom_labels, custom_lines, colors
):
    fig, ax = plt.subplots(1)
    sns.boxplot(
        x="Outlier Rate (\\%)",
        y="Average Translation Error (m)",
        hue="Method",
        data=all_data,
        palette=colors,
        ax=ax,
        flierprops={"marker": "D"},
        # log_scale=True,
    )

    ax = customize_plot(ax)

    ax.legend(custom_lines, custom_labels)

    plt.savefig(f"{dataset_name}_tran.png", dpi=600, bbox_inches="tight")
    plt.show()


def plot_average_rotation_error(
    dataset_name, all_data, custom_labels, custom_lines, colors
):
    fig, ax = plt.subplots(1)
    sns.boxplot(
        x="Outlier Rate (\\%)",
        y="Average Rotation Error (deg)",
        hue="Method",
        data=all_data,
        palette=colors,
        ax=ax,
        flierprops={"marker": "D"},
        # log_scale=True,
    )

    ax = customize_plot(ax)

    ax.legend(custom_lines, custom_labels)

    plt.savefig(f"{dataset_name}_rot.png", dpi=600, bbox_inches="tight")
    plt.show()


def parse_args():
    """Parse commandline args."""

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--is3D", action="store_true", default=False)
    parser.add_argument("--dataset_path", default="../output/robust_pgo_vanilla")

    return parser.parse_args()


def main():
    """Main runner."""

    args = parse_args()

    dataset_name = args.dataset_name
    is3D = args.is3D
    prefix_path = f"{args.dataset_path}/{dataset_name}/"

    if dataset_name.lower() == "intel":
        outlier_pcts = (10, 20, 30, 40, 50)
    else:
        outlier_pcts = (10, 20, 30, 40, 50, 60, 70)

    all_data = collect_data(
        prefix_path=prefix_path,
        is3D=is3D,
        outlier_pcts=outlier_pcts,
    )

    sns.set_style("white")
    # sns.set_palette("bright")
    # pal = "bright"

    colors = ["purple", "red", "green", "blue"]
    labels = ["Ours", "DCSAM", "GNC", "LM"]
    lines = [Line2D([0], [0], color=color, lw=1) for color in colors]

    plot_times(dataset_name, all_data, labels, lines, colors)
    plot_average_translation_error(dataset_name, all_data, labels, lines, colors)
    plot_average_rotation_error(dataset_name, all_data, labels, lines, colors)


if __name__ == "__main__":
    main()
