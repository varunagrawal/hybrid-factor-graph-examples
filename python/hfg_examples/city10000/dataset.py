"""Module for loading the City10000 dataset + defining noise models."""

import gtsam
import numpy as np
from gtsam import Pose2


class NoiseModels:
    """Class with all the noise models used in the estimation."""

    open_loop_model = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3) * 10)
    open_loop_constant = open_loop_model.negLogConstant()

    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([0.0001, 0.0001, 0.0001])
    )

    pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([1.0 / 20.0, 1.0 / 20.0, 1.0 / 100.0])
    )
    pose_noise_constant = pose_noise_model.negLogConstant()


class City10000Dataset:
    """Class representing the City10000 dataset."""

    def __init__(self, filename):
        self.filename_ = filename
        try:
            self.f_ = open(self.filename_, "r")
        except OSError:
            print(f"Failed to open file: {self.filename_}")

    def __del__(self):
        self.f_.close()

    def read_line(self, line: str, delimiter: str = " "):
        """Read a `line` from the dataset, separated by the `delimiter`."""
        return line.split(delimiter)

    def parse_line(self, line: str) -> tuple[list[Pose2], tuple[int, int], bool]:
        """Parse line from file"""
        parts = self.read_line(line)

        key_s = int(parts[1])
        key_t = int(parts[3])

        is_ambiguous_loop = bool(int(parts[4]))

        num_measurements = int(parts[5])
        pose_array = [Pose2()] * num_measurements

        for i in range(num_measurements):
            x = float(parts[6 + 3 * i])
            y = float(parts[7 + 3 * i])
            rad = float(parts[8 + 3 * i])
            pose_array[i] = Pose2(x, y, rad)

        return pose_array, (key_s, key_t), is_ambiguous_loop

    def next(self):
        """Read and parse the next line."""
        line = self.f_.readline()
        if line:
            return self.parse_line(line)
        else:
            return None, None, None
