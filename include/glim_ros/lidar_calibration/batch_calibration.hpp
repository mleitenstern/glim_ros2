#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

#include <glim_ros/lidar_calibration/calibration_bag.hpp>
#include <glim_ros/lidar_calibration/registration.hpp>

namespace guik {
class ProgressInterface;
}

namespace glim {

/// @brief Registration of one partner scan within a frame
struct PairResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Isometry3d T_ref_lidar = Eigen::Isometry3d::Identity();
  double error = -1.0;
  double inlier_fraction = 0.0;
  bool used = false;  ///< Contributed to the average (after inlier / outlier filtering)
};

/// @brief Reference scan with the closest scans of the calibrated topics
struct CalibrationFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BagStamp ref;
  std::map<std::string, BagStamp> partners;  ///< calib topic -> partner scan

  bool processed = false;
  Eigen::Isometry3d T_ref_map = Eigen::Isometry3d::Identity();
  double ref_inlier_fraction = 0.0;
  std::map<std::string, PairResult> results;
};
using CalibrationFrames = std::vector<CalibrationFrame, Eigen::aligned_allocator<CalibrationFrame>>;

/// @brief Initial guess of a LiDAR to calibrate
struct LidarInitialGuess {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Isometry3d T_ref_lidar = Eigen::Isometry3d::Identity();
  bool rotation_only = false;  ///< Keep the translation fixed to the initial guess
};

struct ScanParams {
  double min_range = 1.0;
  double max_range = 200.0;
  double downsample_resolution = 0.1;
  int num_threads = 1;
};

/// @brief Convert and preprocess (downsampling, normals, covariances) a scan from the bag
/// @return nullptr if the scan has too few valid points
gtsam_points::PointCloudCPU::Ptr preprocess_scan(const sensor_msgs::msg::PointCloud2& msg, const ScanParams& params);

/// @brief Pair every frame_stride-th reference scan with the closest scan of each calibrated topic
///        (|stamp difference| <= max_time_offset). Reference scans without any partner are dropped.
CalibrationFrames pair_scans(
  const std::unordered_map<std::string, std::vector<BagStamp>>& index,
  const std::string& ref_topic,
  const std::vector<std::string>& calib_topics,
  double max_time_offset,
  int frame_stride);

/// @brief Register all frames from first_frame on: the reference scan is tracked in the map starting from
///        T_ref_map_first. The initial guess of each following frame is the odometry motion (T_map_ref of the
///        odometry) since the last tracked frame applied to its registered pose; without odometry pose,
///        constant velocity prediction is used. The partner scans are registered into the map with the
///        map pose fixed, yielding T_ref_lidar per pair. Scans are streamed from the bag.
/// @param odometry  Odometry trajectory (may be null or without pose)
/// @return Number of processed frames
int run_batch_calibration(
  const CalibrationBag& bag,
  const std::string& ref_topic,
  CalibrationFrames& frames,
  int first_frame,
  const Eigen::Isometry3d& T_ref_map_first,
  const OdometryTrajectory::ConstPtr& odometry,
  const gtsam_points::PointCloud::ConstPtr& map,
  const gtsam_points::NearestNeighborSearch::ConstPtr& map_tree,
  const std::map<std::string, LidarInitialGuess>& initial_guesses,
  const ScanParams& scan_params,
  const FineRegistrationParams& registration_params,
  double min_inlier_fraction,
  guik::ProgressInterface& progress);

/// @brief Average the per-pair results of one topic. Sets PairResult::used.
/// @param num_pairs  [out] Number of registered pairs of the topic
/// @param num_used   [out] Number of pairs passing the inlier fraction / outlier checks
TransformStatistics compute_lidar_statistics(
  CalibrationFrames& frames,
  const std::string& topic,
  double min_inlier_fraction,
  bool reject_outliers,
  int& num_pairs,
  int& num_used);

}  // namespace glim
