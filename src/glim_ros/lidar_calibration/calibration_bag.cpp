#define GLIM_ROS2
#include <glim_ros/lidar_calibration/calibration_bag.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_compression/sequential_compression_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/metadata_io.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <glim/util/ros_cloud_converter.hpp>
#include <glim_ros/ros_compatibility.hpp>
#include <guik/progress_interface.hpp>

namespace glim {

namespace {

using ReaderPtr = std::unique_ptr<rosbag2_cpp::reader_interfaces::BaseReaderInterface>;

/// @brief Open a bag (storage id detection and compression handling as in glim_rosbag)
ReaderPtr open_reader(const std::string& path) {
  rosbag2_storage::StorageOptions options;
  options.uri = path;

  const bool is_mcap = path.size() > 5 && path.rfind(".mcap") == (path.size() - 5);
  if (is_mcap) {
    options.storage_id = "mcap";
  } else if (std::filesystem::is_directory(path)) {
    try {
      rosbag2_storage::MetadataIo metadata_io;
      options.storage_id = metadata_io.read_metadata(path).storage_identifier;
    } catch (const std::exception& e) {
      spdlog::warn("failed to read metadata.yaml (uri={}): {}", path, e.what());
    }
    if (options.storage_id.empty()) {
      options.storage_id = "sqlite3";
    }
  } else {
    options.storage_id = "sqlite3";
  }

  rosbag2_cpp::ConverterOptions converter_options;
  ReaderPtr reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
  reader->open(options, converter_options);

  if (!reader->get_metadata().compression_format.empty()) {
    reader = std::make_unique<rosbag2_compression::SequentialCompressionReader>();
    reader->open(options, converter_options);
  }

  return reader;
}

/// @brief Read header.stamp directly from a CDR serialized message whose first member is a std_msgs/Header
///        (PointCloud2, Odometry, ...). Avoids deserializing the whole message.
bool read_cdr_header_stamp(const rcutils_uint8_array_t& data, double& stamp) {
  if (data.buffer_length < 12) {
    return false;
  }

  // Encapsulation header: 0x00 0x00 (CDR_BE) or 0x00 0x01 (CDR_LE)
  const bool little_endian = data.buffer[1] == 0x01;
  uint8_t bytes[8];
  std::memcpy(bytes, data.buffer + 4, 8);
  if (!little_endian) {
    std::reverse(bytes, bytes + 4);
    std::reverse(bytes + 4, bytes + 8);
  }

  int32_t sec;
  uint32_t nanosec;
  std::memcpy(&sec, bytes, 4);
  std::memcpy(&nanosec, bytes + 4, 4);
  stamp = sec + nanosec / 1e9;
  return true;
}

void set_progress(guik::ProgressInterface* progress, double fraction, int resolution) {
  if (progress) {
    progress->set_current(std::clamp(static_cast<int>(fraction * resolution), 0, resolution));
  }
}

}  // namespace

CalibrationBag::CalibrationBag(const std::string& path) : bag_path(path) {
  auto reader = open_reader(path);
  const auto metadata = reader->get_metadata();

  start_sec = std::chrono::duration<double>(metadata.starting_time.time_since_epoch()).count();
  duration_sec = std::chrono::duration<double>(metadata.duration).count();

  for (const auto& info : metadata.topics_with_message_count) {
    topics.push_back(BagTopic{info.topic_metadata.name, info.topic_metadata.type, info.message_count});
  }
  std::sort(topics.begin(), topics.end(), [](const BagTopic& a, const BagTopic& b) { return a.name < b.name; });
}

std::vector<BagTopic> CalibrationBag::topics_of_type(const std::string& type) const {
  std::vector<BagTopic> result;
  std::copy_if(topics.begin(), topics.end(), std::back_inserter(result), [&](const BagTopic& t) { return t.type == type; });
  return result;
}

OdometryTrajectory::OdometryTrajectory(std::vector<OdometrySample, Eigen::aligned_allocator<OdometrySample>> samples) : samples_(std::move(samples)) {
  std::sort(samples_.begin(), samples_.end(), [](const OdometrySample& a, const OdometrySample& b) { return a.stamp < b.stamp; });
  // Attitude-only odometry (e.g., INS twist + orientation with zero position) is not a pose in the map
  has_pose_ = std::any_of(samples_.begin(), samples_.end(), [](const OdometrySample& s) { return s.pose.translation().norm() > 1e-3; });
}

std::optional<Eigen::Isometry3d> OdometryTrajectory::pose_at(double stamp) const {
  // Allow a small extrapolation at the ends (stamp jitter)
  const double tolerance = 0.1;
  if (!has_pose_ || samples_.empty() || stamp < samples_.front().stamp - tolerance || stamp > samples_.back().stamp + tolerance) {
    return std::nullopt;
  }

  const auto it = std::lower_bound(samples_.begin(), samples_.end(), stamp, [](const OdometrySample& s, double t) { return s.stamp < t; });
  if (it == samples_.begin()) {
    return samples_.front().pose;
  }
  if (it == samples_.end()) {
    return samples_.back().pose;
  }

  const OdometrySample& s0 = *std::prev(it);
  const OdometrySample& s1 = *it;
  const double dt = s1.stamp - s0.stamp;
  const double p = dt > 1e-9 ? (stamp - s0.stamp) / dt : 0.0;

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = (1.0 - p) * s0.pose.translation() + p * s1.pose.translation();
  pose.linear() = Eigen::Quaterniond(s0.pose.linear()).slerp(p, Eigen::Quaterniond(s1.pose.linear())).normalized().toRotationMatrix();
  return pose;
}

OdometryTrajectory::Ptr CalibrationBag::read_odometry(const std::string& topic, bool use_header_stamp, guik::ProgressInterface& progress) const {
  progress.set_title("Reading odometry");
  progress.set_text("Reading " + topic);
  const int resolution = 100;
  progress.set_maximum(resolution);

  auto reader = open_reader(bag_path);
  rosbag2_storage::StorageFilter filter;
  filter.topics.push_back(topic);
  reader->set_filter(filter);

  rclcpp::Serialization<nav_msgs::msg::Odometry> serialization;
  nav_msgs::msg::Odometry odom;

  std::vector<OdometrySample, Eigen::aligned_allocator<OdometrySample>> samples;
  while (reader->has_next()) {
    const auto msg = reader->read_next();
    const rclcpp::SerializedMessage serialized(*msg->serialized_data);
    serialization.deserialize_message(&serialized, &odom);

    const double recv = get_msg_recv_timestamp(*msg) / 1e9;
    const double stamp = use_header_stamp ? to_sec(odom.header.stamp) : recv;
    const auto& v = odom.twist.twist.linear;
    const auto& p = odom.pose.pose.position;
    const auto& q = odom.pose.pose.orientation;

    OdometrySample sample;
    sample.stamp = stamp;
    sample.speed = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    sample.pose.setIdentity();
    sample.pose.translation() << p.x, p.y, p.z;
    const Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
    if (quat.norm() > 0.5) {  // all-zero quaternion: no orientation
      sample.pose.linear() = quat.normalized().toRotationMatrix();
    }
    samples.push_back(sample);

    set_progress(&progress, (recv - start_sec) / std::max(1e-3, duration_sec), resolution);
  }

  return std::make_shared<OdometryTrajectory>(std::move(samples));
}

std::unordered_map<std::string, std::vector<BagStamp>> CalibrationBag::index_messages(
  const std::vector<std::string>& topics,
  double t_begin,
  double t_end,
  bool use_header_stamp,
  guik::ProgressInterface& progress) const {
  //
  progress.set_title("Indexing point clouds");
  progress.set_text("Reading message headers");
  const int resolution = 100;
  progress.set_maximum(resolution);

  // Header stamps are usually slightly older than the receive time; read with some margin
  const double margin = 2.0;

  auto reader = open_reader(bag_path);
  rosbag2_storage::StorageFilter filter;
  filter.topics = topics;
  reader->set_filter(filter);
  reader->seek(static_cast<int64_t>(std::max(start_sec, t_begin - margin) * 1e9));

  std::unordered_map<std::string, std::vector<BagStamp>> index;
  for (const auto& topic : topics) {
    index[topic];
  }

  while (reader->has_next()) {
    const auto msg = reader->read_next();
    const int64_t recv_ns = get_msg_recv_timestamp(*msg);
    const double recv = recv_ns / 1e9;
    if (recv > t_end + margin) {
      break;
    }

    double stamp = recv;
    if (use_header_stamp && !read_cdr_header_stamp(*msg->serialized_data, stamp)) {
      spdlog::warn("failed to read header stamp of a message on {}", msg->topic_name);
      continue;
    }

    if (stamp >= t_begin && stamp <= t_end) {
      index[msg->topic_name].push_back(BagStamp{recv_ns, stamp});
    }

    set_progress(&progress, (recv - t_begin + margin) / std::max(1e-3, t_end - t_begin + 2.0 * margin), resolution);
  }

  for (auto& [topic, stamps] : index) {
    std::sort(stamps.begin(), stamps.end(), [](const BagStamp& a, const BagStamp& b) { return a.stamp < b.stamp; });
  }

  return index;
}

void CalibrationBag::read_clouds(
  const std::set<MessageKey>& keys,
  const std::function<bool(const MessageKey& key, const sensor_msgs::msg::PointCloud2& msg)>& callback,
  guik::ProgressInterface* progress) const {
  //
  if (keys.empty()) {
    return;
  }

  std::set<std::string> topic_set;
  int64_t first_ns = std::numeric_limits<int64_t>::max();
  int64_t last_ns = std::numeric_limits<int64_t>::min();
  for (const auto& [topic, recv_ns] : keys) {
    topic_set.insert(topic);
    first_ns = std::min(first_ns, recv_ns);
    last_ns = std::max(last_ns, recv_ns);
  }

  const int resolution = 1000;
  if (progress) {
    progress->set_maximum(resolution);
  }

  auto reader = open_reader(bag_path);
  rosbag2_storage::StorageFilter filter;
  filter.topics.assign(topic_set.begin(), topic_set.end());
  reader->set_filter(filter);
  reader->seek(first_ns);

  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
  sensor_msgs::msg::PointCloud2 cloud_msg;

  size_t num_found = 0;
  while (reader->has_next() && num_found < keys.size()) {
    const auto msg = reader->read_next();
    const int64_t recv_ns = get_msg_recv_timestamp(*msg);
    if (recv_ns > last_ns) {
      break;
    }

    const MessageKey key(msg->topic_name, recv_ns);
    if (!keys.count(key)) {
      continue;
    }
    num_found++;

    const rclcpp::SerializedMessage serialized(*msg->serialized_data);
    serialization.deserialize_message(&serialized, &cloud_msg);
    if (!callback(key, cloud_msg)) {
      return;
    }

    set_progress(progress, static_cast<double>(recv_ns - first_ns) / std::max<int64_t>(1, last_ns - first_ns), resolution);
  }

  if (num_found < keys.size()) {
    spdlog::warn("only found {} of {} requested point clouds in the bag", num_found, keys.size());
  }
}

gtsam_points::PointCloudCPU::Ptr convert_cloud(const sensor_msgs::msg::PointCloud2& msg, double min_range, double max_range) {
  const auto raw = extract_raw_points(msg, "intensity");
  if (!raw) {
    return nullptr;
  }

  const bool with_intensity = raw->intensities.size() == raw->points.size();
  const double min_sq = min_range * min_range;
  const double max_sq = max_range * max_range;

  std::vector<Eigen::Vector4d> points;
  std::vector<double> intensities;
  points.reserve(raw->points.size());
  intensities.reserve(with_intensity ? raw->points.size() : 0);

  for (int i = 0; i < raw->size(); i++) {
    const Eigen::Vector4d& p = raw->points[i];
    const double sq = p.head<3>().squaredNorm();
    if (!p.allFinite() || sq < min_sq || sq > max_sq) {
      continue;
    }
    points.emplace_back(p.x(), p.y(), p.z(), 1.0);
    if (with_intensity) {
      intensities.push_back(raw->intensities[i]);
    }
  }

  auto cloud = std::make_shared<gtsam_points::PointCloudCPU>(points);
  if (with_intensity) {
    cloud->add_intensities(intensities);
  }
  return cloud;
}

}  // namespace glim
