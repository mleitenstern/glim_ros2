#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace spdlog {
class logger;
}

namespace glk {
class PointCloudBuffer;
}

namespace guik {
class ProgressModal;
class ProgressInterface;
}  // namespace guik

namespace glim {

class AlignModal;
struct AlignResult;

/// @brief Interactive extrinsic calibration tool for multi-LiDAR setups with non-overlapping FOVs.
///        A prebuilt pointcloud map is first aligned to the scan of a reference LiDAR. The scans of
///        the remaining LiDARs are then registered into the map, yielding their extrinsic calibration
///        w.r.t. the reference LiDAR (T_ref_lidar).
class CalibrationViewer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit CalibrationViewer(int num_threads);
  ~CalibrationViewer();

  /// @brief Queue files to be loaded on startup (any path may be empty)
  void preload(const std::string& ref_scan_path, const std::string& map_path, const std::vector<std::string>& scan_paths);

  /// @brief Run the viewer (blocks until the window is closed)
  void run();

private:
  enum CloudKind { KIND_REF = 0, KIND_MAP = 1, KIND_SCAN = 2 };
  static constexpr int PENDING_NONE = -2;
  static constexpr int PENDING_MAP = -1;

  struct LidarCloud {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::string name;
    std::string path;
    gtsam_points::PointCloudCPU::Ptr raw;
    gtsam_points::PointCloudCPU::Ptr preprocessed;
    std::shared_ptr<glk::PointCloudBuffer> buffer;
    Eigen::Isometry3d T_ref = Eigen::Isometry3d::Identity();  ///< Pose in the reference LiDAR frame
    bool aligned = false;
    double error = -1.0;
  };
  using LidarCloudPtr = std::shared_ptr<LidarCloud>;

  void ui_callback();
  void main_menu();
  void draw_panel();

  void start_load(int kind, const std::string& path_hint);
  LidarCloudPtr load_cloud(guik::ProgressInterface& progress, const std::string& path, double resolution) const;
  void on_loaded(const LidarCloudPtr& cloud);
  void on_aligned(const AlignResult& result);

  void open_align_map();
  void open_align_scan(int index);

  void update_drawables();
  void reset_session();

  void log_extrinsics(const std::string& child_name, const Eigen::Isometry3d& T, double error) const;
  void save_calibration();

private:
  const int num_threads;
  std::atomic_bool request_to_terminate;

  float map_downsample;
  float scan_downsample;

  std::unique_ptr<guik::ProgressModal> progress_modal;
  std::unique_ptr<AlignModal> align_modal;

  LidarCloudPtr ref_scan;
  LidarCloudPtr map;
  std::vector<LidarCloudPtr> scans;

  Eigen::Isometry3d T_ref_map;
  gtsam_points::PointCloudCPU::Ptr map_in_ref;  ///< Preprocessed map transformed into the reference LiDAR frame

  int loading_kind;   // Cloud kind currently being loaded (-1: none)
  int pending_align;  // What the currently open align modal refers to (PENDING_NONE / PENDING_MAP / scan index)

  std::deque<std::pair<int, std::string>> pending_loads;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim
