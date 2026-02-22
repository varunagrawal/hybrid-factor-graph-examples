"""
Script for running DCSAM based estimator on the City10000 dataset.

Author: Varun Agrawal
"""

import time
from pathlib import Path

from gtsam import (
    DCSAM,
    BetweenFactorPose2,
    DiscreteValues,
    HybridValues,
    Pose2,
    PriorFactorPose2,
    Values,
    VectorValues,
)
from gtsam.symbol_shorthand import M, X

from .estimator import BaseEstimator


class DCSAMEstimator(BaseEstimator):
    """An estimator using DCSAM for comparison."""

    def __init__(
        self,
        filename: str,
        num_timesteps: int = 150,
        update_frequency: int = 3,
        max_num_hypotheses: int = 10,
        relinearization_frequency: int = 10,
        plot_hypotheses: bool = False,
        save_path: Path = Path("results"),
    ):
        super().__init__(
            filename=filename,
            num_timesteps=num_timesteps,
            plot_hypotheses=plot_hypotheses,
            save_path=save_path,
        )

        self.name = "DCSAM"

        self.num_timesteps = num_timesteps
        self.update_frequency = update_frequency
        self.max_num_hypotheses = max_num_hypotheses
        self.relinearization_frequency = relinearization_frequency

        self.smoother_ = DCSAM()
        self.all_initial_ = Values()

    def smoother_update(self) -> float:
        """Perform smoother update and optimize the graph."""
        print(f"Smoother update: {self.new_factors_.size()}")
        before_update = time.time()
        self.smoother_.update(
            self.new_factors_,
            HybridValues(VectorValues(), DiscreteValues(), self.initial_),
        )
        self.new_factors_.resize(0)
        self.all_initial_.insert_or_assign(self.initial_)
        self.initial_.clear()
        after_update = time.time()
        return after_update - before_update

    def add_odometry_factor(
        self,
        key_s,
        key_t,
        odom_pose,
        pose_array,
        discrete_count,
        number_of_hybrid_factors,
    ) -> tuple[int, int]:
        """Add odometry factor, which can be a hybrid factor if there are multiple measurements."""
        # Get the number of odometry measurements
        num_measurements = len(pose_array)

        if num_measurements > 1:
            # Add hybrid factor
            m = (M(discrete_count), num_measurements)
            mixture_factor = self.hybrid_odometry_factor(key_s, key_t, m, pose_array)
            self.new_factors_.push_back(mixture_factor)

            self.discrete_cardinalities_[M(discrete_count)] = num_measurements

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
        self.initial_.insert(X(key_t), self.all_initial_.atPose2(X(key_s)) * odom_pose)
        self.all_initial_.insert_or_assign(self.initial_)

        return number_of_hybrid_factors, discrete_count

    def add_loop_closure_factor(
        self,
        key_s,
        key_t,
        odom_pose,
        is_ambiguous_loop,
        loop_count,
        number_of_hybrid_factors,
    ) -> tuple[int, int]:
        """Add loop closure factor, which can be a hybrid factor if the loop is ambiguous."""
        if is_ambiguous_loop:
            loop_factor = self.hybrid_loop_closure_factor(
                loop_count, key_s, key_t, odom_pose
            )

            self.discrete_cardinalities_[L(loop_count)] = 2

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

        loop_count += 1

        # Increment number of hybrid factors so we run more smoother updates.
        number_of_hybrid_factors += 1

        return number_of_hybrid_factors, loop_count

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
        self.all_initial_.insert_or_assign(self.initial_)
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

            # Take the first one as the initial estimate
            odom_pose = pose_array[0]

            if key_s == key_t - 1:
                number_of_hybrid_factors, discrete_count = self.add_odometry_factor(
                    key_s,
                    key_t,
                    odom_pose,
                    pose_array,
                    discrete_count,
                    number_of_hybrid_factors,
                )

            else:
                number_of_hybrid_factors, loop_count = self.add_loop_closure_factor(
                    key_s,
                    key_t,
                    odom_pose,
                    is_ambiguous_loop,
                    loop_count,
                    number_of_hybrid_factors,
                )

            if number_of_hybrid_factors >= self.update_frequency:
                update_time = self.smoother_update()
                smoother_update_times.append((index, update_time))
                number_of_hybrid_factors = 0
                update_count += 1

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
        estimates = self.smoother_.calculateEstimate()

        result.insert_or_assign(estimates.nonlinear())

        print(f"Final error: {self.smoother_.error(estimates.continuous())}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time} seconds")

        self.save_results_and_timing(result, key_t + 1, time_list)
