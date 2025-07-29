import numpy as np
from gtsam.symbol_shorthand import X
from matplotlib import pyplot as plt

from gtsam import Pose2, Values, findExampleDataFile


class City10000Dataset:
    """Class representing the City10000 dataset."""

    def __init__(self, filename):
        self.filename_ = filename
        try:
            self.f_ = open(self.filename_, 'r')
        except OSError:
            print(f"Failed to open file: {self.filename_}")

    def __del__(self):
        self.f_.close()

    def read_line(self, line: str, delimiter: str = " "):
        """Read a `line` from the dataset, separated by the `delimiter`."""
        return line.split(delimiter)

    def parse_line(self,
                   line: str) -> tuple[list[Pose2], tuple[int, int], bool]:
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


def get_poses_from_values(initial, num_poses):
    poses = np.zeros((num_poses, 3))
    for i in range(num_poses):
        pose = initial.atPose2(X(i))
        poses[i] = np.asarray((pose.x(), pose.y(), pose.theta()))
    return poses


def main():
    """Main runner"""
    # Initialize local variables
    index = 0
    max_loop_count = 99

    initial_0 = Values()
    initial_1 = Values()

    # Set up initial prior
    priorPose = Pose2(0, 0, 0)
    initial_0.insert(X(0), priorPose)
    initial_1.insert(X(0), priorPose)

    dataset = City10000Dataset(findExampleDataFile("T1_city10000_04.txt"))

    # Start main loop
    while index < max_loop_count:
        pose_array, keys, is_ambiguous_loop = dataset.next()
        print(index, pose_array)
        if pose_array is None:
            break
        key_s = keys[0]
        key_t = keys[1]

        if key_s == key_t - 1:
            # Insert next pose initial guess
            pose = pose_array[0]

            initial_0.insert(X(key_t), initial_0.atPose2(X(key_s)) * pose)

            if len(pose_array) > 1:
                print("Ambiguous odometry @", index)
                pose = pose_array[1]
            initial_1.insert(X(key_t), initial_1.atPose2(X(key_s)) * pose)

        index += 1

    num_poses = key_t + 1
    poses_0 = get_poses_from_values(initial_0, num_poses)
    poses_1 = get_poses_from_values(initial_1, num_poses)

    fig = plt.figure()
    ax = plt.gca()
    ax.axis('equal')
    ax.axis((-75.0, 100.0, -75.0, 75.0))
    ax.plot(poses_0[:, 0],
            poses_0[:, 1],
            '-',
            linewidth=1,
            color='red',
            label="Poses 0")
    ax.plot(poses_1[:, 0],
            poses_1[:, 1],
            '-',
            linewidth=1,
            color='blue',
            label="Poses 1")
    ax.legend()

    plt.show()


def plot_dataset():
    """Plot the dataset measurements for both modes"""
    index = 0
    max_loop_count = 10000
    dataset = City10000Dataset(findExampleDataFile("T1_city10000_04.txt"))
    x0, x1 = [], []
    y0, y1 = [], []
    theta0, theta1 = [], []
    t = []
    while index < max_loop_count:
        pose_array, _, _ = dataset.next()
        if len(pose_array) > 1:
            t.append(index)
            # x0.append(np.abs(pose_array[0].x()))
            # y0.append(np.abs(pose_array[0].y()))
            # x1.append(np.abs(pose_array[1].x()))
            # y1.append(np.abs(pose_array[1].y()))
            x0.append(pose_array[0].x())
            y0.append(pose_array[0].y())
            theta0.append(pose_array[0].theta())
            x1.append(pose_array[1].x())
            y1.append(pose_array[1].y())
            theta1.append(pose_array[1].theta())

        index += 1

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(t, x0, label="x0")
    axes[0].plot(t, x1, label="x1")
    axes[1].plot(t, y0, label="y0")
    axes[1].plot(t, y1, label="y1")
    axes[2].plot(t, theta0, label="theta0")
    axes[2].plot(t, theta1, label="theta1")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.show()


if __name__ == "__main__":
    main()
    # plot_dataset()
