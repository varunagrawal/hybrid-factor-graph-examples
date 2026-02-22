"""
Script for running a hybrid factor graph estimator on the City10000 dataset.

Author: Varun Agrawal
"""

import time
from pathlib import Path

import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose2,
    HybridSmoother,
    Pose2,
    PriorFactorPose2,
    Values,
)
from gtsam.symbol_shorthand import L, M, X
from hfg_examples.city10000.plot import plot_results

from hfg_examples.city10000 import BaseEstimator


class HybridEstimator(BaseEstimator):
    """A estimator using our developed Hybrid Factor Graphs and a Hybrid Smoother."""

    def __init__(
        self,
        filename: str,
        num_timesteps: int = 10000,
        update_frequency: int = 4,
        max_num_hypotheses: int = 10,
        relinearization_frequency: int = 6,
        marginal_threshold: float = 0.9999,
        plot_hypotheses: bool = False,
        save_path: Path = Path("results"),
    ):
        super().__init__(
            filename=filename,
            num_timesteps=num_timesteps,
            plot_hypotheses=plot_hypotheses,
            save_path=save_path,
        )

        self.name = "Hybrid"

        self.smoother_ = HybridSmoother(marginal_threshold)

        self.update_frequency = update_frequency
        self.max_num_hypotheses = max_num_hypotheses
        self.relinearization_frequency = relinearization_frequency

    def smoother_update(self) -> float:
        """Perform smoother update and optimize the graph."""
        print(f"Smoother update: {self.new_factors_.size()}")
        before_update = time.time()
        self.smoother_.update(self.new_factors_, self.initial_, self.max_num_hypotheses)
        self.new_factors_.resize(0)
        after_update = time.time()
        return after_update - before_update

    def reinitialize(self) -> float:
        """Re-linearize, solve ALL, and re-initialize smoother."""
        print(f"================= Re-Initialize: {self.smoother_.allFactors().size()}")
        before_update = time.time()
        self.smoother_.relinearize()
        self.initial_ = self.smoother_.linearizationPoint()
        after_update = time.time()
        print(f"Took {after_update - before_update} seconds.")
        return after_update - before_update

    def run(self):
        """Run the main experiment with a given num_timesteps."""
        # Initialize local variables
        discrete_count = 0
        index = 0
        loop_count = 0
        update_count = 0

        time_list = []  # list[(int, float)]

        # Set up initial prior
        priorPose = Pose2(0, 0, 0)
        self.initial_.insert(X(0), priorPose)
        self.new_factors_.push_back(
            PriorFactorPose2(X(0), priorPose, self.noise_models_.prior_noise_model)
        )

        # Initial update
        update_time = self.smoother_update()
        smoother_update_times = []  # list[(int, float)]
        smoother_update_times.append((index, update_time))

        # Flag to decide whether to run smoother update
        number_of_hybrid_factors = 0

        # Start main loop
        result = Values()
        start_time = time.time()

        while index < self.num_timesteps:
            pose_array, keys, is_ambiguous_loop = self.dataset_.next()
            if pose_array is None:
                break
            key_s = keys[0]
            key_t = keys[1]

            num_measurements = len(pose_array)

            # Take the first one as the initial estimate
            odom_pose = pose_array[0]

            if key_s == key_t - 1:
                # Odometry factor
                if num_measurements > 1:
                    # Add hybrid factor
                    m = (M(discrete_count), num_measurements)
                    mixture_factor = self.hybrid_odometry_factor(
                        key_s, key_t, m, pose_array
                    )
                    self.new_factors_.push_back(mixture_factor)

                    discrete_count += 1
                    number_of_hybrid_factors += 1
                    print(f"mixture_factor: {key_s} {key_t}")
                else:
                    self.new_factors_.push_back(
                        BetweenFactorPose2(
                            X(key_s),
                            X(key_t),
                            odom_pose,
                            self.noise_models_.pose_noise_model,
                        )
                    )

                # Insert next pose initial guess
                self.initial_.insert(
                    X(key_t), self.initial_.atPose2(X(key_s)) * odom_pose
                )

            else:
                # Loop closure
                if is_ambiguous_loop:
                    loop_factor = self.hybrid_loop_closure_factor(
                        loop_count, key_s, key_t, odom_pose
                    )

                else:
                    loop_factor = BetweenFactorPose2(
                        X(key_s),
                        X(key_t),
                        odom_pose,
                        self.noise_models_.pose_noise_model,
                    )

                # print loop closure event keys:
                print(f"Loop closure: {key_s} {key_t}")
                self.new_factors_.push_back(loop_factor)
                number_of_hybrid_factors += 1
                loop_count += 1

            if number_of_hybrid_factors >= self.update_frequency:
                update_time = self.smoother_update()
                smoother_update_times.append((index, update_time))
                number_of_hybrid_factors = 0
                update_count += 1

                if update_count % self.relinearization_frequency == 0:
                    self.reinitialize()

            #  Record timing for odometry edges only
            if key_s == key_t - 1:
                cur_time = time.time()
                time_list.append(cur_time - start_time)

            # Print some status every 100 steps
            if index % 100 == 0:
                print(f"Index: {index}")

                if len(time_list) != 0:
                    print(f"Accumulate time: {time_list[-1]} seconds")

            index += 1

        # Final update
        update_time = self.smoother_update()
        smoother_update_times.append((index, update_time))

        # Final optimize
        delta = self.smoother_.optimize()

        result.insert_or_assign(self.initial_.retract(delta.continuous()))

        print(f"Final error: {self.smoother_.hybridBayesNet().error(delta)}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time} seconds")

        self.save_results_and_timing(result, key_t + 1, time_list)

        if self.plot_hypotheses:
            # Get all the discrete values
            discrete_keys = gtsam.DiscreteKeys()
            for key in delta.discrete().keys():
                # TODO Get cardinality from DiscreteFactor
                discrete_keys.push_back((key, 2))
            print("plotting all hypotheses")
            self.plot_all_hypotheses(discrete_keys, key_t + 1, index)
