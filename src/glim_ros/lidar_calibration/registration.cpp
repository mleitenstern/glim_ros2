#include <glim_ros/lidar_calibration/registration.hpp>

#include <cmath>

#include <Eigen/Eigenvalues>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

FineRegistrationResult fine_registration(
  const gtsam_points::PointCloud::ConstPtr& target,
  const gtsam_points::NearestNeighborSearch::ConstPtr& target_tree,
  const gtsam_points::PointCloud::ConstPtr& source,
  const Eigen::Isometry3d& T_world_target,
  const Eigen::Isometry3d& init_T_world_source,
  const FineRegistrationParams& params,
  const std::function<void(int, double)>& callback) {
  //
  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(T_world_target.matrix()), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

  if (params.rotation_only) {
    // Anchor the translation of the source pose to the initial guess (rotation stays free)
    gtsam::Vector6 sigmas;
    sigmas << 1e6, 1e6, 1e6, 1e-6, 1e-6, 1e-6;
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1, gtsam::Pose3(init_T_world_source.matrix()), gtsam::noiseModel::Diagonal::Sigmas(sigmas));
  }

  const auto factor = target_tree ? gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source, target_tree)
                                  : gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source);
  factor->set_num_threads(params.num_threads);
  factor->set_max_correspondence_distance(params.max_correspondence_distance);
  graph.add(factor);

  gtsam::Values values;
  values.insert(0, gtsam::Pose3(T_world_target.matrix()));
  values.insert(1, gtsam::Pose3(init_T_world_source.matrix()));

  double final_error = -1.0;

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(params.max_iterations);
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    final_error = status.error;
    if (callback) {
      callback(status.iterations, status.error);
    }
  };

  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

#ifdef GTSAM_USE_TBB
  tbb::task_arena arena(1);
  arena.execute([&] {
#endif
    values = optimizer.optimize();
#ifdef GTSAM_USE_TBB
  });
#endif

  // Update the correspondences at the final estimate to get a meaningful inlier fraction
  factor->error(values);

  FineRegistrationResult result;
  result.T_world_source = Eigen::Isometry3d(values.at<gtsam::Pose3>(1).matrix());
  if (params.rotation_only) {
    result.T_world_source.translation() = init_T_world_source.translation();
  }
  result.error = final_error;
  result.inlier_fraction = factor->inlier_fraction();
  return result;
}

TransformStatistics average_transforms(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& transforms) {
  TransformStatistics stats;
  if (transforms.empty()) {
    return stats;
  }

  // Translation: arithmetic mean
  Eigen::Vector3d t_mean = Eigen::Vector3d::Zero();
  for (const auto& T : transforms) {
    t_mean += T.translation();
  }
  t_mean /= transforms.size();

  // Rotation: eigenvector of the largest eigenvalue of sum(q q^T) (sign invariant, Markley et al. 2007)
  Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
  for (const auto& T : transforms) {
    const Eigen::Vector4d q = Eigen::Quaterniond(T.linear()).normalized().coeffs();
    M += q * q.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(M);
  const Eigen::Vector4d q_mean_coeffs = eig.eigenvectors().col(3);  // eigenvalues sorted ascending
  const Eigen::Quaterniond q_mean(q_mean_coeffs[3], q_mean_coeffs[0], q_mean_coeffs[1], q_mean_coeffs[2]);

  stats.mean.setIdentity();
  stats.mean.linear() = q_mean.normalized().toRotationMatrix();
  stats.mean.translation() = t_mean;

  Eigen::Vector3d t_var = Eigen::Vector3d::Zero();
  Eigen::Vector3d r_var = Eigen::Vector3d::Zero();
  for (const auto& T : transforms) {
    const Eigen::Vector3d dt = T.translation() - t_mean;
    const Eigen::AngleAxisd aa(stats.mean.linear().transpose() * T.linear());
    const Eigen::Vector3d dr = aa.axis() * aa.angle();

    t_var += dt.cwiseProduct(dt);
    r_var += dr.cwiseProduct(dr);
    stats.translation_deviations.push_back(dt.norm());
    stats.rotation_deviations.push_back(std::abs(aa.angle()));
  }

  if (transforms.size() > 1) {
    stats.translation_std = (t_var / (transforms.size() - 1)).cwiseSqrt();
    stats.rotation_std = (r_var / (transforms.size() - 1)).cwiseSqrt();
  }

  return stats;
}

}  // namespace glim
