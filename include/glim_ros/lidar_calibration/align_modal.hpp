#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <glk/drawable.hpp>

namespace spdlog {
class logger;
}

namespace guik {
class GLCanvas;
class ModelControl;
class ProgressModal;
class ProgressInterface;
}  // namespace guik

namespace glim {

/// @brief Result of an accepted alignment
struct AlignResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Isometry3d T_target_source;  ///< Estimated source pose in the target frame
  double error;                       ///< Final fine registration error (negative if fine registration was not run)
};

/// @brief ImGui modal for aligning a source point cloud to a target point cloud.
///        The initial guess can be given manually (gizmo or numeric XYZ/RPY fields) or estimated
///        via FPFH-based global registration (RANSAC / GNC), and is then refined with GICP.
///        The estimation can be constrained to rotation only (translation fixed to the given values).
class AlignModal {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AlignModal(const std::shared_ptr<spdlog::logger>& logger, int num_threads);
  ~AlignModal();

  /// @brief Set the clouds to align and open the modal.
  ///        Both clouds are expected to have covariances (and normals) estimated.
  void set_clouds(
    const std::string& description,
    const gtsam_points::PointCloudCPU::Ptr& target,
    const gtsam_points::PointCloudCPU::Ptr& source,
    const Eigen::Isometry3d& init_T_target_source);

  /// @brief True while the modal is open or about to open
  bool is_active() const;

  /// @brief Draw the modal (call every frame from a UI callback).
  /// @return Alignment result once the user accepts, std::nullopt otherwise
  std::optional<AlignResult> run();

private:
  std::shared_ptr<Eigen::Isometry3d> align_global(guik::ProgressInterface& progress);
  std::shared_ptr<Eigen::Isometry3d> align(guik::ProgressInterface& progress);
  void draw_canvas();
  void draw_guess_ui();
  void clear();
  bool show_note(const std::string& note);

private:
  const int num_threads;

  bool request_to_open;
  bool active;
  std::string description;

  std::unique_ptr<guik::GLCanvas> canvas;
  std::unique_ptr<guik::ProgressModal> progress_modal;
  std::unique_ptr<guik::ModelControl> model_control;

  int seed;

  // Global registration params
  float fpfh_radius;
  int global_registration_type;
  bool global_registration_4dof;

  int ransac_max_iterations;
  float ransac_early_stop_rate;
  float ransac_inlier_voxel_resolution;

  int gnc_max_samples;

  // Fine registration params
  float max_correspondence_distance;
  bool rotation_only;
  double last_error;

  Eigen::Isometry3d init_T_target_source;

  gtsam_points::PointCloudCPU::Ptr target;
  gtsam_points::PointCloudCPU::Ptr source;

  glk::Drawable::ConstPtr target_drawable;
  glk::Drawable::ConstPtr source_drawable;

  gtsam_points::NearestNeighborSearch::Ptr target_fpfh_tree;
  gtsam_points::NearestNeighborSearch::Ptr source_fpfh_tree;

  std::shared_ptr<void> tbb_task_arena;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim
