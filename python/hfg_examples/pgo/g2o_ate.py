"""
Utils to compute the Absolute Translation Error and
the Absolute Rotation error for PGO estimates from g2o files.

Adapted from Kevin Doherty's code on keevindoherty/dcsam-examples

Usage:
`python3 g2o_ate.py [ estimate.g2o ] [ reference.g2o ] [ is3D 0/1 (default false) ]`

We expect that both "estimate.g2o" and "reference.g2o" have vertex values assigned
Basic procedure:
- load poses from g2o files using gtsam.readG2O
- convert to suitable format for evo
- use evo to compute statistics

By aggregating average ATE over multiple MC trials, we can produce box plots as desired.
"""

import argparse
import pprint

import gtsam
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D


def se2_to_se3(pose):
    """Convert a SE(2) pose represented as a (3 x 3) matrix to its representation
    embedded in SE(3). Specifically, suppose we have: pose := [R t; 0 1] where
    R is a 2x2 matrix representing an element of SO(2) and t is a vector in R2.
    This function returns the (4 x 4) matrix:

    [R 0 t; 0 0 1 0; 0 0 0 1]

    `pose`: The pose to convert
    returns pose as an SE(3) element.
    """
    Rblock = pose[0:2, 0:2]
    tblock = pose[0:2, 2][:, np.newaxis]  # Extract translation block as column
    pose3D = np.block(
        [
            [Rblock, np.array([[0.0], [0.0]]), tblock],  # top 2x4 block
            [0.0, 0.0, 1.0, 0.0],  # added 1x4 block
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # make it homogeneous
    return pose3D


def g2o_to_traj(values, is3D):
    """Convert a set of gtsam.Values of type gtsam.pose{D} (D = 2 or 3) to a
    trajectory that can parsed by evo.

    `values`: a set of gtsam.Values of type gtsam.Pose{D} where D = 2 or 3
    `is3D`: true or false (0/1). Set to true of `values` contains gtsam.Pose3

    returns PosePath3D object for processing with evo
    """
    traj = []  # a List[np.ndarray] where each item is a 4x4 SE(3) matrix
    for k in values.keys():
        if is3D:
            pose = values.atPose3(k).matrix()
            traj.append(pose)
        else:
            pose2 = values.atPose2(k).matrix()
            traj.append(se2_to_se3(pose2))

    return PoseTrajectory3D(
        poses_se3=traj,
        timestamps=np.array(range(len(values.keys()))).astype(np.float64),
    )


def g2o_ate_tran(estimate, reference, is3D):
    traj_est = g2o_to_traj(estimate, is3D)
    traj_ref = g2o_to_traj(reference, is3D)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    data = (traj_ref, traj_est)
    ape_translation_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_translation_metric.process_data(data)
    ape_stats = ape_translation_metric.get_all_statistics()
    return ape_stats["mean"]


def g2o_ate_rot(estimate, reference, is3D):
    traj_est = g2o_to_traj(estimate, is3D)
    traj_ref = g2o_to_traj(reference, is3D)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    data = (traj_ref, traj_est)
    ape_translation_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_translation_metric.process_data(data)
    ape_stats = ape_translation_metric.get_all_statistics()
    return ape_stats["mean"]


# def parse_args():
#     """Parse commandline arguments."""
#     parser = argparse.ArgumentParser(
#         description="Usage:  python3 g2o_ate.py [ estimate.g2o ] [ reference.g2o ] [ is3D 0/1 ]"
#     )

#     return parser.parse_args()


# def main():
#     """Main runner."""
#     args = parse_args()
#     estimate_path = args.estimate
#     reference_path = args.reference
#     is3D = args.is3D
#     print(
#         f"\t Estimate path: {estimate_path} \n\t Reference path: {reference_path} \n\t is3D: {is3D}"
#     )

#     est_graph, estimate = gtsam.readG2o(estimate_path, is3D)
#     ref_graph, reference = gtsam.readG2o(reference_path, is3D)

#     traj_est = g2o_to_traj(estimate, is3D)
#     traj_ref = g2o_to_traj(reference, is3D)

#     traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

#     umeyama_result = traj_est.align(
#         traj_ref, correct_scale=False, correct_only_scale=False
#     )

#     data = (traj_ref, traj_est)

#     # Begin evaluating trajectories
#     ## Start with translations
#     ape_translation_metric = metrics.APE(metrics.PoseRelation.translation_part)
#     ape_translation_metric.process_data(data)
#     ape_stats = ape_translation_metric.get_all_statistics()
#     pprint.pprint(ape_stats)

#     ## Now we check rotations
#     pose_relation = metrics.PoseRelation.rotation_angle_deg
#     pprint.pprint(ape_stats)


# if __name__ == "__main__":
#     main()