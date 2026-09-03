"""
Script for running a hybrid factor graph estimator on the City10000 dataset.

Author: Varun Agrawal
"""

import time
from pathlib import Path

from gtsam import (
    BetweenFactorPose2,
    HybridSmoother,
)
from gtsam.symbol_shorthand import L, M, X

from .estimator import BaseEstimator


class HybridEstimator(BaseEstimator):
    """An estimator using our developed Hybrid Factor Graphs and a Hybrid Smoother."""

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
        # print(f"Smoother update: {self.new_factors_.size()}")
        before_update = time.time()
        self.smoother_.update(self.new_factors_, self.initial_, self.max_num_hypotheses)
        self.new_factors_.resize(0)
        after_update = time.time()
        return after_update - before_update

    def reinitialize(self) -> float:
        """Re-linearize, solve ALL, and re-initialize smoother."""
        # print(f"================= Re-Initialize: {self.smoother_.allFactors().size()}")
        before_update = time.time()
        self.smoother_.relinearize()
        self.initial_.insert_or_assign(self.smoother_.linearizationPoint())
        after_update = time.time()
        # print(f"Took {after_update - before_update} seconds.")
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
            # print(f"mixture_factor: {key_s} {key_t}")

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
        self.initial_.insert(X(key_t), self.initial_.atPose2(X(key_s)) * odom_pose)

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
        # print(f"Loop closure: {key_s} {key_t}")
        self.new_factors_.push_back(loop_factor)

        loop_count += 1

        # Increment number of hybrid factors so we run more smoother updates.
        number_of_hybrid_factors += 1

        return number_of_hybrid_factors, loop_count
