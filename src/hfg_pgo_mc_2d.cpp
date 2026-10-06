#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/hybrid/DCSAM.h>
#include <gtsam/hybrid/HybridFactorGraph.h>
#include <gtsam/hybrid/HybridNonlinearFactor.h>
#include <gtsam/hybrid/HybridNonlinearFactorGraph.h>
#include <gtsam/hybrid/HybridSmoother.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/dataset.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "pgo_utils.h"
#include "robust_pgo.h"

const bool DEBUG = false;

/**
 * @brief Re-linearize, solve ALL, update the linearization point, and
 * re-initialize the smoother.
 *
 * The smoother's Bayes net must be linearized at the same point that new
 * factors are linearized at, so after moving the estimate we re-linearize and
 * eliminate again before handing the Bayes net back to the smoother.
 */
void reInitialize(gtsam::HybridSmoother& smoother,
                  gtsam::HybridNonlinearFactorGraph& allFactors,
                  gtsam::Values& estimate) {
  allFactors = allFactors.restrict(smoother.fixedValues());
  auto bayesNet = allFactors.linearize(estimate)->eliminateSequential();
  gtsam::HybridValues delta = bayesNet->optimize();
  estimate = estimate.retract(delta.continuous());
  bayesNet = allFactors.linearize(estimate)->eliminateSequential();
  smoother.reInitialize(std::move(*bayesNet));
}

void run_experiment(const gtsam::NonlinearFactorGraph& graph,
                    const gtsam::Values& initial_values, double outlier_pct,
                    size_t random_seed, std::string dataset_name) {
  /// 1. Build outlier graph.

  // Add prior on the pose having index (key) = 0
  gtsam::NonlinearFactorGraph graphWithOutliers;
  graphWithOutliers.add(graph);

  gtsam::Vector3 prior_sigmas;
  prior_sigmas << 1e-6, 1e-6, 1e-8;
  gtsam::noiseModel::Diagonal::shared_ptr priorModel =
      gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
  graphWithOutliers.add(
      gtsam::PriorFactor<gtsam::Pose2>(0, gtsam::Pose2(), priorModel));
  std::cout << "Adding prior on pose 0 " << std::endl;

  std::cout << graphWithOutliers.size() << std::endl;

  // Give a very coarse estimate for outlier sigmas. Derived from Olson and
  // Agarwal 2013 who recommend 10^7 for outlier variance, we conservatively
  // round up to 4000 for sigmas.
  gtsam::Vector3 outlier_sigmas;
  outlier_sigmas << 4000, 4000, 4000;
  gtsam::noiseModel::Diagonal::shared_ptr outlierModel =
      gtsam::noiseModel::Diagonal::Sigmas(outlier_sigmas);

  double outlier_prob = 0.5;
  std::vector<double> lc_probabilities{(1.0 - outlier_prob), outlier_prob};

  // We'll use this to save the inlier model so we can create the inlier
  // hypotheses.
  gtsam::SharedNoiseModel inlier_model;

  // Keep track of loop closure index.
  size_t k = 0;

  // Make a HybridFactorGraph to store the problem data.
  gtsam::HybridNonlinearFactorGraph hfg;
  gtsam::HybridNonlinearFactorGraph allFactors;

  size_t updateFrequency = 4;
  size_t maxNrHypotheses = 4;
  size_t reLinearizationFrequency = 16;
  size_t numberOfHybridFactors = 0;
  size_t updateCount = 0;
  double dmrThreshold = 0.99;

  // Create the smoother to optimize the HFG
  gtsam::HybridSmoother smoother(dmrThreshold);

  // Current linearization point, updated on every re-initialization.
  gtsam::Values estimate = initial_values;

  // Record the computation time for each smoother update.
  std::vector<double> hfg_compute_times;

  // Run a smoother update on the pending factors and re-initialize if needed.
  auto updateSmoother = [&]() {
    auto t1 = std::chrono::high_resolution_clock::now();
    smoother.update(hfg, estimate, maxNrHypotheses);
    numberOfHybridFactors = 0;
    updateCount++;

    allFactors.push_back(hfg);
    hfg.resize(0);

    if (updateCount % reLinearizationFrequency == 0) {
      reInitialize(smoother, allFactors, estimate);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    hfg_compute_times.push_back(
        std::chrono::duration<double>(t2 - t1).count());
  };

  // Pre-load the odometry and prior (but not the loop closures, which must stay
  // hybrid) so that every per-mode Gaussian system is well-conditioned enough
  // to pass GTSAM's normalized Cholesky pivot check. Without this, the first
  // smoother update throws an IndeterminateSystemException on CSAIL.
  // NOTE: this double-counts odometry and the prior, since they are also added
  // in the loop below.
  for (const auto& factor : graphWithOutliers) {
    auto bwFactor =
        std::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose2>>(factor);
    if (!(bwFactor && isLoopClosure(*bwFactor))) {
      hfg.push_back(factor);
    }
  }

  // Add all good measurements.
  for (const auto& factor : graphWithOutliers) {
    // Ensure that we correctly retrieve a BetweenFactor
    std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose2>> bwFactor =
        std::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose2>>(factor);
    if (bwFactor && isLoopClosure(*bwFactor)) {
      // If the meaasurement is a loop closure, we add inlier and outlier
      // hypotheses.
      auto keys = bwFactor->keys();

      // We'll save the last inlier model to use for all our random
      // measuremenets.
      inlier_model = bwFactor->noiseModel();
      // std::cout << inlier_model->sigmas() << std::endl;

      gtsam::Vector3 outlier_sigmas = 4000.0 * inlier_model->sigmas();

      gtsam::noiseModel::Diagonal::shared_ptr outlierModel =
          gtsam::noiseModel::Diagonal::Sigmas(outlier_sigmas);

      // Build a vector of components factors (inlier model, outlier model)
      auto components =
          get_factor_components_2d(bwFactor, inlier_model, outlierModel);

      // Create a discrete key to index into components. Cardinality is 2
      // since a loop closure is either an inlier (0) or outlier (1).
      gtsam::DiscreteKey dk(gtsam::Symbol('d', k), 2);
      gtsam::HybridNonlinearFactor dcmf(dk, components);

      // hfg.push_back(bwFactor);
      hfg.push_back(dcmf);
      numberOfHybridFactors += 1;
      gtsam::DecisionTreeFactor dpf(dk, lc_probabilities);
      hfg.push_back(dpf);

      k++;

      if (DEBUG) {
        std::cout << "num factors = " << hfg.size() << "  ||  "
                  << "numberOfHybridFactors = " << numberOfHybridFactors
                  << "  ||  " << std::endl;
      }

      if (numberOfHybridFactors >= updateFrequency) {
        updateSmoother();
      }

    } else {
      hfg.push_back(factor);
    }
  }

  updateSmoother();

  size_t num_original_lc = k;

  std::cout << "Processed " << num_original_lc << " loop closures."
            << std::endl;

  // We want num_outliers such that:
  //    outlier_pct = num_outliers / (num_outliers + num_original)
  //
  // A little algebra reveals:
  //    num_outliers = num_original_lc * (outlier_pct / (1 - outlier_pct))
  //
  // Cast is kosher since we know outlier_pct >= 0 and num_original_lc is
  // of type size_t.
  size_t num_outliers =
      (size_t)(num_original_lc * outlier_pct / (1.0 - outlier_pct));
  std::cout << "num original loop closures: " << num_original_lc << std::endl;
  std::cout << "num outliers: " << num_outliers << std::endl;

  gtsam::NonlinearFactorGraph outlierGraph = generateRandomLoopClosures(
      graph.keyVector(), num_outliers, inlier_model, false, random_seed);

  std::cout << "outlier graph size: " << outlierGraph.size() << std::endl;

  // Add all outlier measurements.
  for (const auto& factor : outlierGraph) {
    // Ensure that we correctly retrieve a BetweenFactor
    std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose2>> bwFactor =
        std::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose2>>(factor);
    if (bwFactor && isLoopClosure(*bwFactor)) {
      // If the meaasurement is a loop closure, we add inlier and outlier
      // hypotheses.
      auto keys = bwFactor->keys();

      inlier_model = bwFactor->noiseModel();

      gtsam::Vector3 outlier_sigmas = 4000.0 * inlier_model->sigmas();

      gtsam::noiseModel::Diagonal::shared_ptr outlierModel =
          gtsam::noiseModel::Diagonal::Sigmas(outlier_sigmas);

      // Build a vector of components factors (inlier model, outlier model)
      auto components =
          get_factor_components_2d(bwFactor, inlier_model, outlierModel);

      // Create a discrete key to index into components. Cardinality is 2
      // since a loop closure is either an inlier (0) or outlier (1).
      gtsam::DiscreteKey dk(gtsam::Symbol('d', k), 2);
      gtsam::HybridNonlinearFactor dcmf(dk, components);
      hfg.push_back(dcmf);

      // Set up the discrete weights for each factor.
      gtsam::DecisionTreeFactor dpf(dk, lc_probabilities);
      hfg.push_back(dpf);

      numberOfHybridFactors += 1;

      // Add outlier factor to graph so we can visualize later.
      graphWithOutliers.add(*bwFactor);
      k++;

      if (numberOfHybridFactors >= updateFrequency) {
        updateSmoother();
      }
    } else {
      hfg.push_back(factor);
    }
  }

  // Flush any remaining factors that didn't fill a full update batch.
  if (hfg.size() > 0) {
    updateSmoother();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  gtsam::HybridValues delta = smoother.optimize();
  gtsam::Values result = estimate.retract(delta.continuous());
  auto t2 = std::chrono::high_resolution_clock::now();
  hfg_compute_times.push_back(std::chrono::duration<double>(t2 - t1).count());
  std::cout << "Initial cost: " << graph.error(initial_values) << std::endl;
  std::cout << "Final cost [HFG]: " << graph.error(result) << std::endl;

  std::string path_prefix = "../../output/robust_pgo_vanilla/" + dataset_name +
                            "/" + std::to_string((int)(100 * outlier_pct)) +
                            "/" + std::to_string(random_seed) + "/";
  std::filesystem::create_directories(path_prefix);
  gtsam::writeG2o(graphWithOutliers, result,
                  path_prefix + "out_hfg_robust.g2o");

  // Write timing info
  double total_time_hfg = 0.0;
  for (size_t idx = 0; idx < hfg_compute_times.size(); idx++) {
    total_time_hfg += hfg_compute_times[idx];
  }
  std::ofstream time_file(path_prefix + "times.txt", std::ios_base::app);
  time_file << "HFG: " << std::to_string(total_time_hfg) << "\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " [.g2o file]"
              << " [outlier rate (int >= 0; default 0)]"
              << " [num trials (default 1)]"
              << " [optimizer type (default LM)]" << std::endl;
    exit(1);
  }

  // Defaults
  double outlier_pct = 0.0;
  size_t num_trials = 1;
  std::string optimizer_type = "HFG";

  // Begin parsing input
  std::string path = argv[1];

  std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
  std::string::size_type const p(base_filename.find_last_of('.'));
  std::string file_without_extension = base_filename.substr(0, p);

  if (argc >= 3) {
    std::cout << "outlier percent " << atoi(argv[2]) << std::endl;
    outlier_pct = (double)(atoi(argv[2])) / 100.0;
    std::cout << "outlier percent " << outlier_pct << std::endl;
  }
  if (argc >= 4) {
    assert(atoi(argv[3]) > 0);
    num_trials = atoi(argv[3]);
  }

  std::cout << "Running robust PGO experiment\n"
            << "=============================\n"
            << "dataset:\t" << file_without_extension << "\n"
            << "outlier_pct:\t" << outlier_pct << "\n"
            << "num_trials:\t" << num_trials << "\n"
            << "optimizer_type:\t" << optimizer_type << "\n"
            << std::endl;

  gtsam::NonlinearFactorGraph::shared_ptr graph;
  gtsam::Values::shared_ptr initial;

  std::tie(graph, initial) = readDataset(path);

  std::cout << "Loaded a graph of size: " << graph->size() << std::endl;

  for (size_t i = 1; i < num_trials + 1; i++) {
    std::cout << "Experiment: " << i << std::endl;
    run_experiment(*graph, *initial, outlier_pct, i, file_without_extension);
  }
}
