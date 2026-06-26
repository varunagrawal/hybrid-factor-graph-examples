/**
 * @file   ISAM2_City10000.cpp
 * @brief  Example of using ISAM2 estimation
 *         with multiple odometry measurements.
 * @author Varun Agrawal
 * @date   January 22, 2025
 */

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <time.h>

#include <fstream>
#include <string>
#include <vector>

#include "City10000.h"

using namespace gtsam;

using symbol_shorthand::X;

// Experiment Class
class Experiment {
  /// The City10000 dataset
  City10000Dataset dataset_;

 public:
  // Parameters with default values
  size_t maxLoopCount = 20687;  // 200 //2000 //8000

  // false: run original iSAM2 without ambiguities
  // true: run original iSAM2 with ambiguities
  bool isWithAmbiguity;

 private:
  ISAM2 isam2_;
  NonlinearFactorGraph graph_;
  Values initial_;
  Values results;

 public:
  /// Construct with filename of experiment to run
  explicit Experiment(const std::string& filename, bool isWithAmbiguity = false)
      : dataset_(filename), isWithAmbiguity(isWithAmbiguity) {
    ISAM2Params parameters;
    parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams();
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 2;
    isam2_ = ISAM2(parameters);
  }

  clock_t smootherUpdate() {
    clock_t beforeUpdate = clock();
    isam2_.update(graph_, initial_);
    results = isam2_.calculateEstimate();
    clock_t afterUpdate = clock();
    graph_.resize(0);
    initial_.clear();
    return afterUpdate - beforeUpdate;
  }

  /// @brief Run the main experiment with a given maxLoopCount.
  void run() {
    // Initialize local variables
    size_t index = 0;

    std::vector<std::pair<size_t, double>> smootherUpdateTimes;

    std::list<double> cumulativeTimeList;

    // Set up initial prior
    Pose2 priorPose(0, 0, 0);
    initial_.insert(X(0), priorPose);
    graph_.addPrior<Pose2>(X(0), priorPose, kPriorNoiseModel);

    // Initial update
    clock_t timeDelta = smootherUpdate();
    smootherUpdateTimes.push_back(std::make_pair(index, timeDelta));
    index += 1;

    // Start main loop
    size_t keyS = 0;
    size_t keyT = 0;
    clock_t startTime = clock();

    std::vector<Pose2> poseArray;
    std::pair<size_t, size_t> keys;

    while (dataset_.next(&poseArray, &keys) && index < maxLoopCount) {
      keyS = keys.first;
      keyT = keys.second;
      size_t numMeasurements = poseArray.size();

      Pose2 odomPose;
      if (isWithAmbiguity) {
        // Get wrong intentionally
        int id = index % numMeasurements;
        odomPose = Pose2(poseArray[id]);
      } else {
        odomPose = poseArray[0];
      }

      if (keyS == keyT - 1) {  // new X(key)
        initial_.insert(X(keyT), results.at<Pose2>(X(keyS)) * odomPose);
        graph_.add(
            BetweenFactor<Pose2>(X(keyS), X(keyT), odomPose, kPoseNoiseModel));

      } else {  // Loop closure
        int id = index % numMeasurements;
        if (isWithAmbiguity && id % 2 == 0) {
          graph_.add(BetweenFactor<Pose2>(X(keyS), X(keyT), odomPose,
                                          kPoseNoiseModel));

        } else {
          graph_.add(BetweenFactor<Pose2>(
              X(keyS), X(keyT), odomPose,
              noiseModel::Diagonal::Sigmas(Vector3::Ones() * 10.0)));
        }
        index += 1;
      }

      // Perform update
      clock_t timeDelta = smootherUpdate();
      smootherUpdateTimes.push_back(std::make_pair(index, timeDelta));

      // Record timing for odometry edges only
      if (keyS == keyT - 1) {
        clock_t curTime = clock();
        cumulativeTimeList.push_back(curTime - startTime);
      }

      // Print loop index and time taken in processor clock ticks
      if (index % 100 == 0) {
        std::cout << "Index: " << index << std::endl;
        if (!cumulativeTimeList.empty()) {
          std::cout << "accTime:  "
                    << cumulativeTimeList.back() / CLOCKS_PER_SEC << " seconds"
                    << std::endl;
        }
      }
    }

    // Final update
    auto time_delta = smootherUpdate();
    smootherUpdateTimes.push_back({index, time_delta});

    // Final optimize
    results = isam2_.calculateBestEstimate();

    clock_t endTime = clock();
    clock_t totalTime = endTime - startTime;
    std::cout << "Total time: " << totalTime / CLOCKS_PER_SEC << " seconds"
              << std::endl;

    /// Write results to file
    writeResult(results, (keyT + 1), "ISAM2_City10000.txt");

    std::ofstream outfileTime;
    std::string timeFileName = "ISAM2_City10000_time.txt";
    outfileTime.open(timeFileName);
    for (auto accTime : cumulativeTimeList) {
      outfileTime << accTime / CLOCKS_PER_SEC << std::endl;
    }
    outfileTime.close();
    std::cout << "Written cumulative time to: " << timeFileName << " file."
              << std::endl;

    std::ofstream timingFile;
    std::string timingFileName = "ISAM2_City10000_timing.txt";
    timingFile.open(timingFileName);
    for (size_t i = 0; i < smootherUpdateTimes.size(); i++) {
      auto p = smootherUpdateTimes.at(i);
      timingFile << p.first << ", " << p.second / CLOCKS_PER_SEC << std::endl;
    }
    timingFile.close();
    std::cout << "Wrote timing information to " << timingFileName << std::endl;
  }
};

/* ************************************************************************* */
// Function to parse command-line arguments
void parseArguments(int argc, char* argv[], size_t& maxLoopCount,
                    bool& isWithAmbiguity) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--max-loop-count" && i + 1 < argc) {
      maxLoopCount = std::stoul(argv[++i]);
    } else if (arg == "--is-with-ambiguity" && i + 1 < argc) {
      isWithAmbiguity = bool(std::stoul(argv[++i]));
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --max-loop-count <value>       Set the maximum loop "
                   "count (default: 2000)\n"
                << "  --is-with-ambiguity <value=0/1>     Set whether to use "
                   "ambiguous measurements "
                   "(default: false)\n"
                << "  --help                         Show this help message\n";
      std::exit(0);
    }
  }
}

/* ************************************************************************* */
int main(int argc, char* argv[]) {
  Experiment experiment(findExampleDataFile("T1_City10000_04.txt"));
  // Experiment experiment("../data/mh_T1_City10000_04.txt"); //Type #1 only
  // Experiment experiment("../data/mh_T3b_City10000_10.txt"); //Type #3 only
  // Experiment experiment("../data/mh_T1_T3_City10000_04.txt"); //Type #1 +
  // Type #3

  // Parse command-line arguments
  parseArguments(argc, argv, experiment.maxLoopCount,
                 experiment.isWithAmbiguity);

  // Run the experiment
  experiment.run();

  return 0;
}
