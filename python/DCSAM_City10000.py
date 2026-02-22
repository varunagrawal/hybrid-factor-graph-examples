"""
Script for running DCSAM on the City10000 dataset.

Author: Varun Agrawal
"""

import argparse

import gtsam

from hfg_examples.city10000 import DCSAMEstimator


def main():
    """Main runner"""
    args = parse_arguments()

    estimator = DCSAMEstimator(
        gtsam.findExampleDataFile(args.data_file),
        num_timesteps=args.num_timesteps,
        update_frequency=args.update_frequency,
        max_num_hypotheses=args.max_num_hypotheses,
        plot_hypotheses=args.plot_hypotheses,
    )
    estimator.run()


if __name__ == "__main__":
    main()
