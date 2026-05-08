"""Plotting utilities for the City10000 dataset."""

import numpy as np
from matplotlib import pyplot as plt


def plot_results(
    ground_truth,
    all_results,
    estimate_label,
    estimate_color,
    num_iters,
    text="",
    filename="city10000_results.svg",
):
    """Plot the City10000 estimates against the ground truth.

    Args:
        ground_truth: The ground truth trajectory as xy values.
        all_results (List[Tuple(np.ndarray, str)]): All the estimates trajectory as xy values,
            as well as assginment strings.
        estimate_label (str): Label for the estimates, used in the legend.
        estimate_color (tuple): The color to use for the graph of estimates.
        num_iters (int): The number of iterations, used in the title.
        text (str): The text to display at the bottom of the plot.
        filename (str): The name of the file to save the plot to.
    """
    if len(all_results) == 1:
        fig, axes = plt.subplots(1, 1)
        axes = [axes]
    else:
        fig, axes = plt.subplots(int(np.ceil(len(all_results) / 2)), 2)
        axes = axes.flatten()

    for i, (estimates, s, prob) in enumerate(all_results):
        ax = axes[i]
        ax.axis("equal")
        ax.axis((-75.0, 100.0, -75.0, 75.0))

        gt = ground_truth[: estimates.shape[0]]
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
        # ax.legend()
        ax.set_title(f"P={prob:.3f}\n{s}", fontdict={"fontsize": 10})

    fig.suptitle(f"After {num_iters} iterations")

    num_chunks = int(np.ceil(len(text) / 90))
    text = "\n".join(text[i * 60 : (i + 1) * 60] for i in range(num_chunks))
    fig.text(0.5, 0.015, s=text, wrap=True, horizontalalignment="center", fontsize=12)

    fig.savefig(filename, format="svg")
