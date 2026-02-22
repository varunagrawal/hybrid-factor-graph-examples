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
    HybridNonlinearFactor,
    HybridNonlinearFactorGraph,
    HybridSmoother,
    Pose2,
    PriorFactorPose2,
    Values,
)
from gtsam.symbol_shorthand import L, M, X
from hfg_examples.city10000.plot import plot_results

from hfg_examples.city10000 import dataset


class HybridEstimator:
    """A estimator using our developed Hybrid Factor Graphs and a Hybrid Smoother."""

    def __init__(
        self,
        filename: str,
        max_loop_count: int = 10000,
        update_frequency: int = 4,
        max_num_hypotheses: int = 10,
        relinearization_frequency: int = 6,
        marginal_threshold: float = 0.9999,
        plot_hypotheses: bool = False,
        save_path: Path = Path("results"),
    ):
        self.dataset_ = dataset.City10000Dataset(filename)
        self.max_loop_count = max_loop_count
        self.update_frequency = update_frequency
        self.max_num_hypotheses = max_num_hypotheses
        self.relinearization_frequency = relinearization_frequency

        self.noise_models_ = dataset.NoiseModels()

        self.smoother_ = HybridSmoother(marginal_threshold)
        self.new_factors_ = HybridNonlinearFactorGraph()
        self.initial_ = Values()

        self.plot_hypotheses = plot_hypotheses
        self.save_path_ = save_path

    def hybrid_loop_closure_factor(
        self, loop_counter, key_s, key_t, measurement: Pose2
    ):
        """
        Create a hybrid loop closure factor where
        0 - loose noise model and 1 - loop noise model.
        """
        loop_key = (L(loop_counter), 2)
        f0 = BetweenFactorPose2(
            X(key_s), X(key_t), measurement, self.noise_models_.open_loop_model
        )
        f1 = BetweenFactorPose2(
            X(key_s), X(key_t), measurement, self.noise_models_.pose_noise_model
        )
        factors = [
            (f0, self.noise_models_.open_loop_constant),
            (f1, self.noise_models_.pose_noise_constant),
        ]
        mixture_factor = HybridNonlinearFactor(loop_key, factors)
        return mixture_factor

    def hybrid_odometry_factor(
        self, key_s, key_t, m, pose_array
    ) -> HybridNonlinearFactor:
        """Create hybrid odometry factor with discrete measurement choices."""
        f0 = BetweenFactorPose2(
            X(key_s), X(key_t), pose_array[0], self.noise_models_.pose_noise_model
        )
        f1 = BetweenFactorPose2(
            X(key_s), X(key_t), pose_array[1], self.noise_models_.pose_noise_model
        )

        factors = [
            (f0, self.noise_models_.pose_noise_constant),
            (f1, self.noise_models_.pose_noise_constant),
        ]
        mixture_factor = HybridNonlinearFactor(m, factors)

        return mixture_factor

    def smoother_update(self, max_num_hypotheses) -> float:
        """Perform smoother update and optimize the graph."""
        print(f"Smoother update: {self.new_factors_.size()}")
        before_update = time.time()
        self.smoother_.update(self.new_factors_, self.initial_, max_num_hypotheses)
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
        """Run the main experiment with a given max_loop_count."""
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
        update_time = self.smoother_update(self.max_num_hypotheses)
        smoother_update_times = []  # list[(int, float)]
        smoother_update_times.append((index, update_time))

        # Flag to decide whether to run smoother update
        number_of_hybrid_factors = 0

        # Start main loop
        result = Values()
        start_time = time.time()

        while index < self.max_loop_count:
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
                update_time = self.smoother_update(self.max_num_hypotheses)
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
        update_time = self.smoother_update(self.max_num_hypotheses)
        smoother_update_times.append((index, update_time))

        # Final optimize
        delta = self.smoother_.optimize()

        result.insert_or_assign(self.initial_.retract(delta.continuous()))

        print(f"Final error: {self.smoother_.hybridBayesNet().error(delta)}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time} seconds")

        self.save_results_and_timing(
            result, key_t + 1, time_list, save_path=self.save_path_
        )

        if self.plot_hypotheses:
            # Get all the discrete values
            discrete_keys = gtsam.DiscreteKeys()
            for key in delta.discrete().keys():
                # TODO Get cardinality from DiscreteFactor
                discrete_keys.push_back((key, 2))
            print("plotting all hypotheses")
            self.plot_all_hypotheses(discrete_keys, key_t + 1, index)

    def plot_all_hypotheses(self, discrete_keys, num_poses, num_iters=0):
        """Plot all possible hypotheses."""

        # Get ground truth
        gt = np.loadtxt(
            gtsam.findExampleDataFile("ISAM2_GT_city10000.txt"), delimiter=" "
        )

        dkeys = gtsam.DiscreteKeys()
        for i in range(discrete_keys.size()):
            key, cardinality = discrete_keys.at(i)
            if key not in self.smoother_.fixedValues().keys():
                dkeys.push_back((key, cardinality))
        fixed_values_str = " ".join(
            f"{gtsam.DefaultKeyFormatter(k)}:{v}"
            for k, v in self.smoother_.fixedValues().items()
        )

        all_assignments = gtsam.cartesianProduct(dkeys)

        all_results = []
        for assignment in all_assignments:
            result = gtsam.Values()
            gbn = self.smoother_.hybridBayesNet().choose(assignment)

            # Check to see if the GBN has any nullptrs, if it does it is null overall
            is_invalid_gbn = False
            for i in range(gbn.size()):
                if gbn.at(i) is None:
                    is_invalid_gbn = True
                    break
            if is_invalid_gbn:
                continue

            delta = self.smoother_.hybridBayesNet().optimize(assignment)
            result.insert_or_assign(self.initial_.retract(delta))

            poses = np.zeros((num_poses, 3))
            for i in range(num_poses):
                pose = result.atPose2(X(i))
                poses[i] = np.asarray((pose.x(), pose.y(), pose.theta()))

            assignment_string = " ".join(
                [f"{gtsam.DefaultKeyFormatter(k)}={v}" for k, v in assignment.items()]
            )

            conditional = (
                self.smoother_.hybridBayesNet()
                .at(self.smoother_.hybridBayesNet().size() - 1)
                .asDiscrete()
            )
            discrete_values = self.smoother_.fixedValues()
            for k, v in assignment.items():
                discrete_values[k] = v

            if conditional is None:
                probability = 1.0
            else:
                probability = conditional.evaluate(discrete_values)

            all_results.append((poses, assignment_string, probability))

        plot_results(
            gt,
            all_results,
            num_iters=num_iters,
            estimate_label="Hybrid Factor Graphs",
            estimate_color=(0.1, 0.1, 0.9, 0.4),
            text=fixed_values_str,
            filename=f"city10000_results_{num_iters}.svg",
        )

    def save_results_and_timing(self, result, final_key, time_list, save_path):
        """Save results to file."""
        # Write results to file
        self.write_result(result, final_key, save_path / "Hybrid_City10000.txt")

        # Write timing info to file
        self.write_timing_info(
            time_list=time_list, filename=save_path / "Hybrid_City10000_time.txt"
        )

    def write_result(self, result, num_poses, filename="Hybrid_city10000.txt"):
        """
        Write the result of optimization to file.

        Args:
            result (Values): he Values object with the final result.
            num_poses (int): The number of poses to write to the file.
            filename (str): The file name to save the result to.
        """
        with open(filename, "w") as outfile:
            for i in range(num_poses):
                out_pose = result.atPose2(X(i))
                outfile.write(f"{out_pose.x()} {out_pose.y()} {out_pose.theta()}\n")

        print(f"Output written to {filename}")

    def write_timing_info(self, time_list, filename):
        """Log all the timing information to a file"""

        with open(filename, "w") as out_file_time:
            for acc_time in time_list:
                out_file_time.write(f"{acc_time}\n")

        print(f"Timing saved to {filename}.")
