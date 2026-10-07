#include <gtsam/hybrid/HybridNonlinearFactor.h>

#include <cmath>
#include <optional>

/**
 * Loop closures are modeled as a Gaussian-uniform mixture (as in MLESAC):
 *
 *   p(z | inlier)  = N(z; h(x), Sigma_in)
 *   p(z | outlier) = 1 / V            (uniform over a measurement volume V)
 *
 * V is specified relative to the inlier Gaussian's normalizer
 * Z_in = sqrt((2 pi)^n |Sigma_in|) as `log_volume_ratio` = ln(V / Z_in), so
 * the outlier mode costs ln V = (inlier negLogConstant) + log_volume_ratio.
 * With equal discrete priors, a loop closure is kept as an inlier iff
 * 0.5 * chi^2 < log_volume_ratio, i.e. this is a chi^2 gate at
 * 2 * log_volume_ratio.
 *
 * For 3-DOF (Pose2) loop closures, 3 ln(10) (V = 10^3 Z_in) gates at
 * chi^2 ~= 13.8, roughly the 99.7th percentile.
 */
const double DEFAULT_OUTLIER_LOG_VOLUME_RATIO_2D = 3.0 * std::log(10.0);

/// Negative log density of the uniform outlier model, see above.
inline double uniform_outlier_neg_log_density(double inlier_neg_log_constant,
                                              double log_volume_ratio) {
  return inlier_neg_log_constant + log_volume_ratio;
}

/**
 * @brief Build a vector of components factors (inlier model, outlier model)
 * 2D version version.
 *
 * @param bwFactor
 * @param inlier_model
 * @param outlier_model Noise model of the outlier factor. With
 * `log_volume_ratio`, this factor only stands in for a zero-information
 * (uniform) factor on the same keys, so make it wide enough that its residual
 * is negligible and rejected loop closures don't pull on the trajectory.
 * @param log_volume_ratio If given, use a uniform outlier model with this
 * ln(V / Z_in). Otherwise the outlier mode is the Gaussian `outlier_model`.
 * @return std::vector<gtsam::NonlinearFactorValuePair>
 */
std::vector<gtsam::NonlinearFactorValuePair> get_factor_components_2d(
    const std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose2>>& bwFactor,
    const gtsam::SharedNoiseModel inlier_model,
    const gtsam::noiseModel::Gaussian::shared_ptr& outlier_model,
    std::optional<double> log_volume_ratio = {}) {
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
  double outlierNegLogConstant =
      log_volume_ratio ? uniform_outlier_neg_log_density(negLogConstant,
                                                         *log_volume_ratio)
                       : outlier_model->negLogConstant();
  components.push_back({outlier_factor, outlierNegLogConstant});

  return components;
}

/// Build a vector of components factors (inlier model, outlier model).
/// See get_factor_components_2d for `log_volume_ratio`.
std::vector<gtsam::NonlinearFactorValuePair> get_factor_components_3d(
    const std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose3>>& bwFactor,
    const gtsam::SharedNoiseModel inlier_model,
    const gtsam::noiseModel::Diagonal::shared_ptr& outlier_model,
    std::optional<double> log_volume_ratio = {}) {
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
  double outlierNegLogConstant =
      log_volume_ratio ? uniform_outlier_neg_log_density(negLogConstant,
                                                         *log_volume_ratio)
                       : outlier_model->negLogConstant();
  components.push_back({outlier_factor, outlierNegLogConstant});

  return components;
}
