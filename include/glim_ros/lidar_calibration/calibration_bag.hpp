#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace guik {
class ProgressInterface;
}

namespace glim {

struct BagTopic {
  std::string name;
  std::string type;
  size_t message_count = 0;
};

/// @brief Timestamp entry of a message in the bag
struct BagStamp {
  int64_t recv_ns;  ///< Bag receive timestamp (unique message key together with the topic)
  double stamp;     ///< Header stamp (or receive time if header stamps are disabled) [s]
};

/// @brief Sample of the odometry topic
struct OdometrySample {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp;            ///< [s]
  double speed;            ///< |twist.linear| [m/s]
  Eigen::Isometry3d pose;  ///< Pose of the reference LiDAR in the map (T_map_ref)
};

/// @brief Odometry samples sorted by stamp: speed profile and (if the topic carries one) the pose of the reference LiDAR in the map
class OdometryTrajectory {
public:
  using Ptr = std::shared_ptr<OdometryTrajectory>;
  using ConstPtr = std::shared_ptr<const OdometryTrajectory>;

  explicit OdometryTrajectory(std::vector<OdometrySample, Eigen::aligned_allocator<OdometrySample>> samples);

  const std::vector<OdometrySample, Eigen::aligned_allocator<OdometrySample>>& samples() const { return samples_; }

  /// @brief False if the topic carries no pose (all positions zero, e.g., twist / attitude-only odometry)
  bool has_pose() const { return has_pose_; }

  /// @brief Interpolated T_map_ref at the given stamp (nullopt without pose or outside the trajectory)
  std::optional<Eigen::Isometry3d> pose_at(double stamp) const;

private:
  std::vector<OdometrySample, Eigen::aligned_allocator<OdometrySample>> samples_;
  bool has_pose_;
};

/// @brief Read-only access to a rosbag2 (sqlite3 / mcap, optionally compressed) for the calibration tool.
///        Every read opens a fresh reader, so the methods can be called from worker threads.
class CalibrationBag {
public:
  using Ptr = std::shared_ptr<CalibrationBag>;
  using MessageKey = std::pair<std::string, int64_t>;  ///< (topic, recv_ns)

  /// @brief Open the bag and read its metadata (throws on failure)
  explicit CalibrationBag(const std::string& path);

  const std::string& path() const { return bag_path; }
  double start_time() const { return start_sec; }  ///< Bag start (receive clock) [s]
  double duration() const { return duration_sec; }

  /// @brief Topics of the given type (e.g., "sensor_msgs/msg/PointCloud2"), sorted by name
  std::vector<BagTopic> topics_of_type(const std::string& type) const;

  /// @brief Read a nav_msgs/msg/Odometry topic (speed profile and pose)
  OdometryTrajectory::Ptr read_odometry(const std::string& topic, bool use_header_stamp, guik::ProgressInterface& progress) const;

  /// @brief Collect the stamps of all messages of the given topics with stamp in [t_begin, t_end] (absolute [s]).
  ///        Only the message headers are parsed (no full deserialization).
  std::unordered_map<std::string, std::vector<BagStamp>> index_messages(
    const std::vector<std::string>& topics,
    double t_begin,
    double t_end,
    bool use_header_stamp,
    guik::ProgressInterface& progress) const;

  /// @brief Stream the requested PointCloud2 messages in bag order.
  ///        The callback may return false to abort reading.
  void read_clouds(
    const std::set<MessageKey>& keys,
    const std::function<bool(const MessageKey& key, const sensor_msgs::msg::PointCloud2& msg)>& callback,
    guik::ProgressInterface* progress = nullptr) const;

private:
  std::string bag_path;
  double start_sec;
  double duration_sec;
  std::vector<BagTopic> topics;
};

/// @brief Convert a PointCloud2 into a point cloud with intensities, dropping invalid points and points outside [min_range, max_range].
/// @return nullptr if the message has no xyz fields
gtsam_points::PointCloudCPU::Ptr convert_cloud(const sensor_msgs::msg::PointCloud2& msg, double min_range, double max_range);

}  // namespace glim
