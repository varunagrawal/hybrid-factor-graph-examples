"""
Script for running hybrid estimator on the City10000 dataset.

Author: Varun Agrawal
"""

import argparse

import gtsam

from hfg_examples.city10000 import HybridEstimator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        help="The path to the City10000 data file",
        default="T1_city10000_04.txt",
    )
    parser.add_argument(
        "--num_timesteps",
        "-n",
        type=int,
        default=10000,
        help="The maximum number of loops to run over the dataset",
    )
    parser.add_argument(
        "--update_frequency",
        "-u",
        type=int,
        default=3,
        help="After how many steps to run the smoother update.",
    )
    parser.add_argument(
        "--max_num_hypotheses",
        "-m",
        type=int,
        default=10,
        help="The maximum number of hypotheses to keep at any time.",
    )
    parser.add_argument(
        "--plot_hypotheses",
        "-p",
        action="store_true",
        help="Plot all hypotheses. NOTE: This is exponential, use with caution.",
    )
    return parser.parse_args()


def main():
    """Main runner"""
    args = parse_arguments()

    estimator = HybridEstimator(
        gtsam.findExampleDataFile(args.data_file),
        num_timesteps=args.num_timesteps,
        update_frequency=args.update_frequency,
        max_num_hypotheses=args.max_num_hypotheses,
        plot_hypotheses=args.plot_hypotheses,
    )
    estimator.run()


if __name__ == "__main__":
    main()
