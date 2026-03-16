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
#include <iostream>
#include <random>

#include "pgo_utils.h"
#include "robust_pgo.h"

const bool DEBUG = false;

/// @brief Re-linearize, solve ALL, and re-initialize smoother.
clock_t reInitialize(gtsam::HybridSmoother& smoother,
                     gtsam::HybridNonlinearFactorGraph& allFactors,
                     gtsam::Values initial) {
  clock_t beforeUpdate = clock();
  allFactors = allFactors.restrict(smoother.fixedValues());
  auto linearized = allFactors.linearize(initial);
  auto bayesNet = linearized->eliminateSequential();
  gtsam::HybridValues delta = bayesNet->optimize();
  initial = initial.retract(delta.continuous());
  smoother.reInitialize(std::move(*bayesNet));
  clock_t afterUpdate = clock();
  // std::cout << "Took " << (afterUpdate - beforeUpdate) / CLOCKS_PER_SEC
  //           << " seconds." << std::endl;
  return afterUpdate - beforeUpdate;
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
  size_t reLinearizationFrequency = 16;  // best value: 6;
  size_t numberOfHybridFactors = 0;
  size_t updateCount = 0;

  // Create the smoother to optimize the HFG
  gtsam::HybridSmoother smoother;

  std::set<gtsam::Key> added_keys;

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

      hfg.push_back(bwFactor);
      // hfg.push_back(dcmf);
      // numberOfHybridFactors += 1;
      // gtsam::DecisionTreeFactor dpf(dk, lc_probabilities);
      // hfg.push_back(dpf);

      k++;

      if (DEBUG) {
        std::cout << "num factors = " << hfg.size() << "  ||  "
                  << "numberOfHybridFactors = " << numberOfHybridFactors
                  << "  ||  " << std::endl;
      }

      if (numberOfHybridFactors >= updateFrequency) {
        smoother.update(hfg, initial_values, maxNrHypotheses);
        numberOfHybridFactors = 0;
        updateCount++;

        allFactors.push_back(hfg);
        hfg.resize(0);

        if (updateCount % reLinearizationFrequency == 0) {
          reInitialize(smoother, allFactors, initial_values);
        }
      }

    } else {
      hfg.push_back(factor);
    }
  }

  smoother.update(hfg, initial_values, maxNrHypotheses);
  allFactors.push_back(hfg);
  hfg.resize(0);

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
        smoother.update(hfg, initial_values, maxNrHypotheses, 0.99);
        numberOfHybridFactors = 0;
        updateCount++;

        allFactors.push_back(hfg);
        hfg.resize(0);

        if (updateCount % reLinearizationFrequency == 0) {
          reInitialize(smoother, allFactors, initial_values);
        }
      }
    } else {
      hfg.push_back(factor);
    }
  }

  gtsam::Values result;
  gtsam::HybridValues delta = smoother.optimize();
  result.insert_or_assign(initial_values.retract(delta.continuous()));
  std::cout << "Initial cost: " << graph.error(initial_values) << std::endl;
  std::cout << "Final cost [HFG]: " << graph.error(result) << std::endl;
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

  std::tie(graph, initial) = gtsam::readG2o(path, false);

  std::cout << "Loaded a graph of size: " << graph->size() << std::endl;

  for (size_t i = 1; i < num_trials + 1; i++) {
    std::cout << "Experiment: " << i << std::endl;
    run_experiment(*graph, *initial, outlier_pct, i, file_without_extension);
  }
}
