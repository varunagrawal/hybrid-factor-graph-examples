#include <gtsam/hybrid/HybridNonlinearFactor.h>

/**
 * @brief Build a vector of components factors (inlier model, outlier model)
 * 2D version version.
 *
 * @param bwFactor
 * @param inlier_model
 * @param outlier_model
 * @return std::vector<gtsam::NonlinearFactorValuePair>
 */
std::vector<gtsam::NonlinearFactorValuePair> get_factor_components_2d(
    const std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose2>>& bwFactor,
    const gtsam::SharedNoiseModel inlier_model,
    const gtsam::noiseModel::Gaussian::shared_ptr& outlier_model) {
  auto keys = bwFactor->keys();

  std::vector<gtsam::NonlinearFactorValuePair> components;

  double negLogConstant = 0.0;
  if (auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
          inlier_model)) {
    negLogConstant = gaussian->negLogConstant();
  }
  components.push_back({bwFactor, negLogConstant});

  auto outlier_factor = std::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      keys[0], keys[1], bwFactor->measured(), outlier_model);
  components.push_back({outlier_factor, outlier_model->negLogConstant()});

  return components;
}

/// Build a vector of components factors (inlier model, outlier model)
std::vector<gtsam::NonlinearFactorValuePair> get_factor_components_3d(
    const std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose3>>& bwFactor,
    const gtsam::SharedNoiseModel inlier_model,
    const gtsam::noiseModel::Diagonal::shared_ptr& outlier_model) {
  auto keys = bwFactor->keys();

  std::vector<gtsam::NonlinearFactorValuePair> components;

  double negLogConstant = 0.0;
  if (auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
          inlier_model)) {
    negLogConstant = gaussian->negLogConstant();
  }
  components.push_back({bwFactor, negLogConstant});

  auto outlier_factor = std::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      keys[0], keys[1], bwFactor->measured(), outlier_model);
  components.push_back({outlier_factor, outlier_model->negLogConstant()});

  return components;
}
