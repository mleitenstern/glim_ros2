#include <glim_ros/lidar_calibration/batch_calibration.hpp>

#include <cmath>
#include <algorithm>

#include <spdlog/spdlog.h>

#include <gtsam/geometry/Pose3.h>
#include <guik/progress_interface.hpp>

#include <glim_ros/lidar_calibration/pcd_loader.hpp>

namespace glim {

namespace {

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t n = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + n, values.end());
  return values[n];
}

/// @brief Robust upper bound for non-negative deviations: median + 3 * 1.4826 * MAD
double robust_threshold(const std::vector<double>& deviations, double floor) {
  const double med = median(deviations);
  std::vector<double> abs_dev(deviations.size());
  std::transform(deviations.begin(), deviations.end(), abs_dev.begin(), [med](double d) { return std::abs(d - med); });
  return std::max(floor, med + 3.0 * 1.4826 * median(abs_dev));
}

}  // namespace

gtsam_points::PointCloudCPU::Ptr preprocess_scan(const sensor_msgs::msg::PointCloud2& msg, const ScanParams& params) {
  const auto raw = convert_cloud(msg, params.min_range, params.max_range);
  if (!raw || raw->size() < 10) {
    spdlog::warn("too few valid points in a scan of {}", msg.header.frame_id);
    return nullptr;
  }

  guik::ProgressInterface no_progress;
  return preprocess_cloud(raw, params.downsample_resolution, params.num_threads, no_progress);
}

CalibrationFrames pair_scans(
  const std::unordered_map<std::string, std::vector<BagStamp>>& index,
  const std::string& ref_topic,
  const std::vector<std::string>& calib_topics,
  double max_time_offset,
  int frame_stride) {
  //
  CalibrationFrames frames;
  const auto ref_found = index.find(ref_topic);
  if (ref_found == index.end()) {
    return frames;
  }

  const auto& ref_stamps = ref_found->second;
  for (size_t i = 0; i < ref_stamps.size(); i += std::max(1, frame_stride)) {
    CalibrationFrame frame;
    frame.ref = ref_stamps[i];

    for (const auto& topic : calib_topics) {
      const auto found = index.find(topic);
      if (found == index.end() || topic == ref_topic) {
        continue;
      }

      // Closest stamp (stamps are sorted)
      const auto& stamps = found->second;
      const auto it = std::lower_bound(stamps.begin(), stamps.end(), frame.ref.stamp, [](const BagStamp& s, double t) { return s.stamp < t; });
      const BagStamp* best = it != stamps.end() ? &*it : nullptr;
      if (it != stamps.begin() && (!best || frame.ref.stamp - std::prev(it)->stamp < best->stamp - frame.ref.stamp)) {
        best = &*std::prev(it);
      }

      if (best && std::abs(best->stamp - frame.ref.stamp) <= max_time_offset) {
        frame.partners[topic] = *best;
      }
    }

    if (!frame.partners.empty()) {
      frames.push_back(frame);
    }
  }

  return frames;
}

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
  guik::ProgressInterface& progress) {
  //
  progress.set_title("Batch calibration");
  for (auto& frame : frames) {
    frame.processed = false;
    frame.results.clear();
  }

  const int num_frames = frames.size();
  using MessageKey = CalibrationBag::MessageKey;

  // Messages to read and the number of frames that need each of them (a scan may be the partner of several frames)
  std::map<MessageKey, int> ref_counts;
  for (int i = first_frame; i < num_frames; i++) {
    ref_counts[{ref_topic, frames[i].ref.recv_ns}]++;
    for (const auto& [topic, partner] : frames[i].partners) {
      if (initial_guesses.count(topic)) {
        ref_counts[{topic, partner.recv_ns}]++;
      }
    }
  }

  std::set<MessageKey> keys;
  for (const auto& kv : ref_counts) {
    keys.insert(kv.first);
  }

  // Preprocessed scans that are read but not yet consumed by all of their frames
  std::map<MessageKey, gtsam_points::PointCloudCPU::Ptr> cache;

  const auto release = [&](const MessageKey& key) {
    if (--ref_counts[key] <= 0) {
      cache.erase(key);
    }
  };

  // Track of the reference LiDAR pose in the map: (stamp, T_map_ref)
  std::vector<std::pair<double, Eigen::Isometry3d>, Eigen::aligned_allocator<std::pair<double, Eigen::Isometry3d>>> track;

  const auto predict = [&](double stamp) -> Eigen::Isometry3d {
    if (track.empty()) {
      return T_ref_map_first.inverse();
    }

    // Odometry motion since the last tracked frame (robust to a constant offset between odometry and map)
    if (odometry && odometry->has_pose()) {
      const auto T_odom_prev = odometry->pose_at(track.back().first);
      const auto T_odom_curr = odometry->pose_at(stamp);
      if (T_odom_prev && T_odom_curr) {
        return track.back().second * T_odom_prev->inverse() * (*T_odom_curr);
      }
    }

    if (track.size() < 2) {
      return track.back().second;
    }

    // Constant velocity extrapolation of the last two tracked poses
    const auto& [t0, T0] = track[track.size() - 2];
    const auto& [t1, T1] = track.back();
    if (t1 - t0 < 1e-6) {
      return T1;
    }
    const gtsam::Vector6 delta = gtsam::Pose3::Logmap(gtsam::Pose3((T0.inverse() * T1).matrix()));
    const double scale = std::clamp((stamp - t1) / (t1 - t0), 0.0, 10.0);
    return T1 * Eigen::Isometry3d(gtsam::Pose3::Expmap(delta * scale).matrix());
  };

  const auto process = [&](CalibrationFrame& frame) {
    const MessageKey ref_key(ref_topic, frame.ref.recv_ns);
    const auto ref_cloud = cache[ref_key];

    if (ref_cloud) {
      // World = map frame, the map is fixed at identity
      const auto r = fine_registration(map, map_tree, ref_cloud, Eigen::Isometry3d::Identity(), predict(frame.ref.stamp), registration_params);
      frame.T_ref_map = r.T_world_source.inverse();
      frame.ref_inlier_fraction = r.inlier_fraction;
      frame.processed = true;

      if (r.inlier_fraction >= min_inlier_fraction) {
        track.emplace_back(frame.ref.stamp, r.T_world_source);
      }
    }

    for (const auto& [topic, partner] : frame.partners) {
      const auto guess = initial_guesses.find(topic);
      if (guess == initial_guesses.end()) {
        continue;
      }

      const MessageKey key(topic, partner.recv_ns);
      const auto cloud = cache[key];
      if (frame.processed && cloud) {
        FineRegistrationParams params = registration_params;
        params.rotation_only = guess->second.rotation_only;

        // World = reference LiDAR frame, the map is fixed at T_ref_map
        const auto r = fine_registration(map, map_tree, cloud, frame.T_ref_map, guess->second.T_ref_lidar, params);
        PairResult& pair = frame.results[topic];
        pair.T_ref_lidar = r.T_world_source;
        pair.error = r.error;
        pair.inlier_fraction = r.inlier_fraction;
      }
      release(key);
    }
    release(ref_key);
  };

  const auto frame_ready = [&](const CalibrationFrame& frame) {
    if (!cache.count({ref_topic, frame.ref.recv_ns})) {
      return false;
    }
    for (const auto& [topic, partner] : frame.partners) {
      if (initial_guesses.count(topic) && !cache.count({topic, partner.recv_ns})) {
        return false;
      }
    }
    return true;
  };

  int next_frame = first_frame;
  int num_processed = 0;

  bag.read_clouds(
    keys,
    [&](const MessageKey& key, const sensor_msgs::msg::PointCloud2& msg) {
      cache[key] = preprocess_scan(msg, scan_params);

      // Frames are processed in order as soon as all of their scans are available
      while (next_frame < num_frames && frame_ready(frames[next_frame])) {
        progress.set_text("frame " + std::to_string(next_frame - first_frame + 1) + " / " + std::to_string(num_frames - first_frame));
        process(frames[next_frame++]);
        num_processed++;
      }
      return true;
    },
    &progress);

  if (next_frame < num_frames) {
    spdlog::warn("{} frames could not be processed (scans missing in the bag)", num_frames - next_frame);
  }

  return num_processed;
}

TransformStatistics compute_lidar_statistics(
  CalibrationFrames& frames,
  const std::string& topic,
  double min_inlier_fraction,
  bool reject_outliers,
  int& num_pairs,
  int& num_used) {
  //
  num_pairs = 0;
  num_used = 0;

  std::vector<PairResult*> pairs;
  for (auto& frame : frames) {
    const auto found = frame.results.find(topic);
    if (!frame.processed || found == frame.results.end()) {
      continue;
    }

    num_pairs++;
    found->second.used = false;
    if (frame.ref_inlier_fraction >= min_inlier_fraction && found->second.inlier_fraction >= min_inlier_fraction) {
      pairs.push_back(&found->second);
    }
  }

  if (pairs.empty()) {
    return TransformStatistics();
  }

  const auto average = [&] {
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> transforms;
    for (const auto* p : pairs) {
      transforms.push_back(p->T_ref_lidar);
    }
    return average_transforms(transforms);
  };

  TransformStatistics stats = average();

  if (reject_outliers && pairs.size() >= 5) {
    const double rot_thresh = robust_threshold(stats.rotation_deviations, 1e-4);
    const double trans_thresh = robust_threshold(stats.translation_deviations, 1e-3);

    std::vector<PairResult*> inliers;
    for (size_t i = 0; i < pairs.size(); i++) {
      if (stats.rotation_deviations[i] <= rot_thresh && stats.translation_deviations[i] <= trans_thresh) {
        inliers.push_back(pairs[i]);
      }
    }

    if (!inliers.empty()) {
      pairs = inliers;
      stats = average();
    }
  }

  for (auto* p : pairs) {
    p->used = true;
  }
  num_used = pairs.size();

  return stats;
}

}  // namespace glim
