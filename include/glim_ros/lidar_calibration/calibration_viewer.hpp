#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

#include <glim_ros/lidar_calibration/batch_calibration.hpp>

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
///        The scans are read from a rosbag, the map of the scene from a .pcd file.
///        1. A time window is selected on the speed profile (odometry topic) of the bag.
///        2. Every scan of the reference LiDAR topic in the window is paired with the closest scan
///           (max. time offset) of each LiDAR topic to calibrate.
///        3. On one pair, the map is aligned to the reference scan and the other scans to the map interactively.
///        4. Starting from that pair, all following pairs are registered automatically: the reference scan
///           is tracked in the map, the other scans are registered into the map, yielding T_ref_lidar per pair.
///        5. The per-pair extrinsics are averaged.
class CalibrationViewer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    std::string bag_path;
    std::string map_path;
    std::string odom_topic = "/core/state/odometry";
    std::string ref_topic;
    std::vector<std::string> calib_topics;
    double max_time_offset = 0.05;
  };

  CalibrationViewer(int num_threads, const Options& options);
  ~CalibrationViewer();

  /// @brief Run the viewer (blocks until the window is closed)
  void run();

private:
  /// @brief Point cloud with its preprocessed (downsampled + covariances) version and drawable
  struct Cloud {
    gtsam_points::PointCloudCPU::Ptr raw;
    gtsam_points::PointCloudCPU::Ptr preprocessed;
    std::shared_ptr<glk::PointCloudBuffer> buffer;
  };
  using CloudPtr = std::shared_ptr<Cloud>;

  using Frame = CalibrationFrame;

  /// @brief Calibration state of one LiDAR topic
  struct Lidar {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool initialized = false;  ///< Initial guess accepted on the anchor frame
    bool rotation_only = false;
    Eigen::Isometry3d T_init = Eigen::Isometry3d::Identity();
    double init_error = -1.0;

    int num_pairs = 0;  ///< Registered pairs
    int num_used = 0;   ///< Pairs used for the average
    TransformStatistics stats;
  };

  /// @brief Clouds of one frame loaded for display / interactive alignment
  struct LoadedFrame {
    int index = -1;
    CloudPtr ref;
    std::map<std::string, CloudPtr> partners;
  };
  using LoadedFramePtr = std::shared_ptr<LoadedFrame>;

  struct LoadedMap {
    std::string path;
    CloudPtr cloud;
    gtsam_points::NearestNeighborSearch::ConstPtr tree;
  };
  using LoadedMapPtr = std::shared_ptr<LoadedMap>;

  struct BatchResult {
    CalibrationFrames frames;
    int num_processed = 0;
  };
  using BatchResultPtr = std::shared_ptr<BatchResult>;

  void ui_callback();
  void main_menu();
  void draw_bag_panel();
  void draw_speed_plot();
  void draw_calibration_panel();
  void dispatch_tasks();
  bool busy() const;

  // Bag / map loading
  void open_bag(const std::string& path);
  void start_load_odometry();
  void start_load_map(const std::string& path_hint);
  void start_index_window();
  void start_load_frame(int index);
  void start_batch();

  ScanParams scan_params() const;

  // Pairing / results
  std::string ref_topic() const;
  std::vector<std::string> calib_topics() const;
  void rebuild_frames();
  void clear_results();
  void recompute_statistics();
  double speed_at(double stamp) const;

  // Interactive alignment
  void open_align_map();
  void open_align_lidar(const std::string& topic);
  void on_aligned(const AlignResult& result);

  void update_drawables();
  void log_extrinsics(const std::string& child_name, const Eigen::Isometry3d& T) const;
  void save_calibration();

private:
  const int num_threads;
  Options options;
  std::atomic_bool request_to_terminate;

  // Parameters
  float map_downsample;
  float scan_downsample;
  float min_range;
  float max_range;
  bool use_header_stamp;
  float max_time_offset_ms;
  int frame_stride;
  float batch_max_corr_dist;
  float min_inlier_fraction;
  bool reject_outliers;

  std::unique_ptr<guik::ProgressModal> progress_modal;
  std::unique_ptr<AlignModal> align_modal;
  std::string active_task;                         ///< Name of the running progress modal task ("" if idle)
  std::deque<std::function<void()>> task_queue;  ///< Tasks started once idle (startup loading)

  // Bag
  CalibrationBag::Ptr bag;
  std::vector<BagTopic> cloud_topics;
  std::vector<BagTopic> odom_topics;
  std::map<std::string, int> topic_roles;  ///< 0: unused, 1: reference, 2: calibrate
  int odom_topic_index;
  OdometryTrajectory::ConstPtr odometry;
  std::vector<double> speed_t;  ///< relative to bag start [s]
  std::vector<double> speed_v;  ///< [m/s]

  // Time window (relative to bag start) [s]
  double window_begin;
  double window_end;
  double plot_range_begin;  ///< Visible x range of the speed plot
  double plot_range_end;

  // Indexed window
  double indexed_begin;
  double indexed_end;
  bool indexed_header_stamp;
  std::unordered_map<std::string, std::vector<BagStamp>> index;
  CalibrationFrames frames;
  std::string frames_ref_topic;

  // Map
  LoadedMapPtr map;

  // Anchor frame for the interactive initial alignment
  int anchor_frame;
  bool map_aligned;
  Eigen::Isometry3d T_ref_map_anchor;
  gtsam_points::PointCloudCPU::Ptr map_in_ref;  ///< Preprocessed map in the reference frame of the anchor frame

  std::map<std::string, Lidar> lidars;

  // Display
  int display_frame;
  LoadedFramePtr loaded_frame;
  bool show_map;

  bool pending_align_map;      ///< The open align modal refers to the map
  std::string pending_align;  ///< The open align modal refers to this LiDAR topic

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim
