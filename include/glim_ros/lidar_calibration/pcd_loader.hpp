#pragma once

#include <string>
#include <memory>

#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace guik {
class ProgressInterface;
}

namespace glim {

/// @brief Load a point cloud from a .pcd file (XYZ + optional intensity).
/// @return Loaded point cloud or nullptr on failure
gtsam_points::PointCloudCPU::Ptr load_pcd(const std::string& path);

/// @brief Downsample a point cloud and estimate per-point normals and covariances (required for FPFH and GICP).
gtsam_points::PointCloudCPU::Ptr preprocess_cloud(
  const gtsam_points::PointCloud::ConstPtr& points,
  double downsample_resolution,
  int num_threads,
  guik::ProgressInterface& progress);

}  // namespace glim
