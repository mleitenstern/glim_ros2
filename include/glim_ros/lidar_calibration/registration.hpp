#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace glim {

struct FineRegistrationParams {
  double max_correspondence_distance = 1.0;
  int max_iterations = 30;
  bool rotation_only = false;  ///< Anchor the translation of the source pose to its initial value
  int num_threads = 1;
};

struct FineRegistrationResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Isometry3d T_world_source;  ///< Estimated source pose
  double error;                      ///< Final GICP error
  double inlier_fraction;            ///< Fraction of source points with a correspondence at the final pose
};

/// @brief GICP registration of source to target, both expressed in a common "world" frame.
///        The target pose is held fixed at T_world_target; the source pose is estimated.
///        With T_world_target = Identity this is a plain target <- source registration.
/// @param target_tree  Optional prebuilt nearest neighbor search on target (built internally if null)
/// @param callback     Optional per-iteration callback (iteration, error)
FineRegistrationResult fine_registration(
  const gtsam_points::PointCloud::ConstPtr& target,
  const gtsam_points::NearestNeighborSearch::ConstPtr& target_tree,
  const gtsam_points::PointCloud::ConstPtr& source,
  const Eigen::Isometry3d& T_world_target,
  const Eigen::Isometry3d& init_T_world_source,
  const FineRegistrationParams& params,
  const std::function<void(int, double)>& callback = nullptr);

struct TransformStatistics {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Isometry3d mean = Eigen::Isometry3d::Identity();
  Eigen::Vector3d translation_std = Eigen::Vector3d::Zero();  ///< Per-axis std of the translations [m]
  Eigen::Vector3d rotation_std = Eigen::Vector3d::Zero();     ///< Per-axis std of the rotation deviations (so3 log in the mean frame) [rad]
  std::vector<double> translation_deviations;                 ///< |t_i - t_mean| [m]
  std::vector<double> rotation_deviations;                    ///< angle(R_mean^T R_i) [rad]
};

/// @brief Average of rigid transforms: arithmetic mean translation, quaternion (Markley) mean rotation.
TransformStatistics average_transforms(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& transforms);

}  // namespace glim
