"""
Functions to plot boxplots for PGO monte carlo results.
"""

import sys

import g2o_ate
import gtsam
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    dcsam_time = float(contents[0].split()[1])
    gnc_time = float(contents[1].split()[1])
    lm_time = float(contents[2].split()[1])
    f.close()
    return dcsam_time, gnc_time, lm_time


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:  python3 plot_g2o_results.py [ dataset_name ] [ is3D 0/1 ]")
        sys.exit()

    dataset_name = sys.argv[1]
    is3D = 1 if (int(sys.argv[2]) > 0) else 0
    prefix_path = f"../output/robust_pgo_vanilla/{dataset_name}/"
    # prefix_path = f"../output_backup/{dataset_name}/"

    dcsam_ate_tran = {}
    dcsam_ate_rot = {}
    dcsam_times = {}

    gnc_ate_tran = {}
    gnc_ate_rot = {}
    gnc_times = {}

    lm_ate_tran = {}
    lm_ate_rot = {}
    lm_times = {}
    gt_graph, gt_vals = gtsam.readG2o(prefix_path + "0/1/out_nonrobust.g2o", is3D)
    all_data = pd.DataFrame(
        columns=[
            "Method",
            "Outlier Rate (%%)",
            "Time (s)",
            "Average Translation Error (m)",
            "Average Rotation Error (deg)",
        ]
    )
    for outlier_pct in [10, 20, 30, 40]:  # [10, 20, 30, 40, 50, 60, 70, 80]:
        dcsam_ate_tran[outlier_pct] = []
        dcsam_ate_rot[outlier_pct] = []
        dcsam_times[outlier_pct] = []

        gnc_ate_tran[outlier_pct] = []
        gnc_ate_rot[outlier_pct] = []
        gnc_times[outlier_pct] = []

        lm_ate_tran[outlier_pct] = []
        lm_ate_rot[outlier_pct] = []
        lm_times[outlier_pct] = []
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # DCSAM Processing
            dcsam_graph, dcsam_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_robust.g2o", is3D
            )
            dcsam_tran = g2o_ate.g2o_ate_tran(dcsam_est, gt_vals, is3D)
            dcsam_rot = g2o_ate.g2o_ate_rot(dcsam_est, gt_vals, is3D)
            dcsam_ate_tran[outlier_pct].append(dcsam_tran)
            dcsam_ate_rot[outlier_pct].append(dcsam_rot)

            # GNC Processing
            gnc_graph, gnc_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_gnc.g2o", is3D
            )
            gnc_tran = g2o_ate.g2o_ate_tran(gnc_est, gt_vals, is3D)
            gnc_rot = g2o_ate.g2o_ate_rot(gnc_est, gt_vals, is3D)
            gnc_ate_tran[outlier_pct].append(gnc_tran)
            gnc_ate_rot[outlier_pct].append(gnc_rot)

            # LM Processing
            lm_graph, lm_est = gtsam.readG2o(
                prefix_path + f"{outlier_pct}/{seed}/out_nonrobust.g2o", is3D
            )
            lm_tran = g2o_ate.g2o_ate_tran(lm_est, gt_vals, is3D)
            lm_rot = g2o_ate.g2o_ate_rot(lm_est, gt_vals, is3D)
            lm_ate_tran[outlier_pct].append(lm_tran)
            lm_ate_rot[outlier_pct].append(lm_rot)

            # Get times
            dcsam_time, gnc_time, lm_time = parse_times(
                prefix_path + f"{outlier_pct}/{seed}/times.txt"
            )
            all_data.loc[len(all_data.index)] = [
                "DCSAM",
                outlier_pct,
                dcsam_time,
                dcsam_tran,
                dcsam_rot,
            ]
            dcsam_times[outlier_pct].append(dcsam_time)

            all_data.loc[len(all_data.index)] = [
                "GNC",
                outlier_pct,
                gnc_time,
                gnc_tran,
                gnc_rot,
            ]
            gnc_times[outlier_pct].append(gnc_time)

            all_data.loc[len(all_data.index)] = [
                "LM",
                outlier_pct,
                lm_time,
                lm_tran,
                lm_rot,
            ]
            lm_times[outlier_pct].append(lm_time)

    sns.set_style("white")
    # sns.set_palette("bright")
    colors = ["red", "green", "blue"]

    pal = "bright"
    fig, ax = plt.subplots(1)
    sns.boxplot(
        x="Outlier Rate (%%)",
        y="Time (s)",
        hue="Method",
        data=all_data,
        palette=colors,
        ax=ax,
    )
    print(ax.patches)
    box_patches = [
        patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch
    ]
    if (
        len(box_patches) == 0
    ):  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
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

    # for i,artist in enumerate(ax.patches):
    #     # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    #     col = artist.get_facecolor()
    #     print(col)
    #     artist.set_edgecolor(col)
    #     artist.set_facecolor('None')

    #     # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    #     # Loop over them here, and use the same colour as above
    #     for j in range(i*6,i*6+6):
    #         line = ax.lines[j]
    #         line.set_color(col)
    #         line.set_mfc(col)
    #         line.set_mec(col)

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

    # cmap = sns.color_palette("hls", 8, as_cmap=True)

    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=1),
        Line2D([0], [0], color=colors[1], lw=1),
        Line2D([0], [0], color=colors[2], lw=1),
    ]
    ax.legend(custom_lines, ["Ours", "GNC", "LM"])
    plt.savefig(f"{dataset_name}_times.png", dpi=600, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(
        x="Outlier Rate (%%)",
        y="Average Translation Error (m)",
        hue="Method",
        data=all_data,
        palette=colors,
        ax=ax,
    )
    print(ax.patches)
    # plt.yscale("log")
    box_patches = [
        patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch
    ]
    if (
        len(box_patches) == 0
    ):  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
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

    # for i,artist in enumerate(ax.patches):
    #     # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    #     col = artist.get_facecolor()
    #     print(col)
    #     artist.set_edgecolor(col)
    #     artist.set_facecolor('None')

    #     # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    #     # Loop over them here, and use the same colour as above
    #     for j in range(i*6,i*6+6):
    #         line = ax.lines[j]
    #         line.set_color(col)
    #         line.set_mfc(col)
    #         line.set_mec(col)

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

    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=1),
        Line2D([0], [0], color=colors[1], lw=1),
        Line2D([0], [0], color=colors[2], lw=1),
    ]
    ax.legend(custom_lines, ["Ours", "GNC", "LM"])
    plt.savefig(f"{dataset_name}_tran.png", dpi=600, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(
        x="Outlier Rate (%%)",
        y="Average Rotation Error (deg)",
        hue="Method",
        data=all_data,
        palette=colors,
        ax=ax,
    )
    # plt.yscale("log")
    print(ax.patches)
    box_patches = [
        patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch
    ]
    if (
        len(box_patches) == 0
    ):  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
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

    # for i,artist in enumerate(ax.patches):
    #     # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    #     col = artist.get_facecolor()
    #     print(col)
    #     artist.set_edgecolor(col)
    #     artist.set_facecolor('None')

    #     # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    #     # Loop over them here, and use the same colour as above
    #     for j in range(i*6,i*6+6):
    #         line = ax.lines[j]
    #         line.set_color(col)
    #         line.set_mfc(col)
    #         line.set_mec(col)

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

    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=1),
        Line2D([0], [0], color=colors[1], lw=1),
        Line2D([0], [0], color=colors[2], lw=1),
    ]
    ax.legend(custom_lines, ["Ours", "GNC", "LM"])

    plt.savefig(f"{dataset_name}_rot.png", dpi=600, bbox_inches="tight")
    plt.show()
