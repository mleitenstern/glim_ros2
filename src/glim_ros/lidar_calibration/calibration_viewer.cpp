#include <glim_ros/lidar_calibration/calibration_viewer.hpp>

#include <cmath>
#include <optional>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <portable-file-dialogs.h>
#include <implot.h>

#include <gtsam_points/ann/kdtree2.hpp>

#include <glk/colormap.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/recent_files.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <glim_ros/lidar_calibration/align_modal.hpp>
#include <glim_ros/lidar_calibration/pcd_loader.hpp>

namespace glim {

namespace {

enum TopicRole { ROLE_NONE = 0, ROLE_REF = 1, ROLE_CALIB = 2 };

void to_rpy(const Eigen::Isometry3d& T, Eigen::Vector3d& rpy) {
  const Eigen::Matrix3d R = T.linear();
  rpy.x() = std::atan2(R(2, 1), R(2, 2));
  rpy.y() = std::asin(std::max(-1.0, std::min(1.0, -R(2, 0))));
  rpy.z() = std::atan2(R(1, 0), R(0, 0));
}

bool show_note(const char* note) {
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%s", note);
  }
  return false;
}

}  // namespace

CalibrationViewer::CalibrationViewer(int num_threads, const Options& options) : num_threads(num_threads), options(options), request_to_terminate(false) {
  map_downsample = 0.25f;
  scan_downsample = 0.1f;
  min_range = 1.0f;
  max_range = 200.0f;
  use_header_stamp = true;
  max_time_offset_ms = options.max_time_offset * 1e3;
  frame_stride = 1;
  batch_max_corr_dist = 1.0f;
  min_inlier_fraction = 0.5f;
  reject_outliers = true;

  odom_topic_index = -1;
  window_begin = window_end = 0.0;
  plot_range_begin = plot_range_end = 0.0;
  indexed_begin = indexed_end = 0.0;
  indexed_header_stamp = true;

  anchor_frame = -1;
  map_aligned = false;
  T_ref_map_anchor.setIdentity();

  display_frame = 0;
  show_map = true;

  pending_align_map = false;

  logger = spdlog::default_logger();
}

CalibrationViewer::~CalibrationViewer() {}

void CalibrationViewer::run() {
  auto viewer = guik::LightViewer::instance(Eigen::Vector2i(1920, 1080));

  progress_modal.reset(new guik::ProgressModal("calibration_progress"));
  align_modal.reset(new AlignModal(logger, num_threads));

  if (!options.bag_path.empty()) {
    open_bag(options.bag_path);
  }
  if (bag && odom_topic_index >= 0) {
    task_queue.emplace_back([this] { start_load_odometry(); });
  }
  if (!options.map_path.empty()) {
    task_queue.emplace_back([this] { start_load_map(options.map_path); });
  }

  viewer->register_ui_callback("calibration_ui", [this] { ui_callback(); });

  while (!request_to_terminate && viewer->spin_once()) {
  }

  viewer->register_ui_callback("calibration_ui", nullptr);
  align_modal.reset();
  progress_modal.reset();
}

void CalibrationViewer::ui_callback() {
  main_menu();
  draw_bag_panel();
  draw_calibration_panel();

  dispatch_tasks();

  // Alignment modal
  auto align_result = align_modal->run();
  if (align_result) {
    on_aligned(*align_result);
  } else if (!align_modal->is_active()) {
    pending_align_map = false;
    pending_align.clear();
  }

  // Start queued tasks once idle
  if (!task_queue.empty() && active_task.empty() && !align_modal->is_active()) {
    auto task = task_queue.front();
    task_queue.pop_front();
    task();
  }
}

bool CalibrationViewer::busy() const {
  return !active_task.empty() || align_modal->is_active();
}

void CalibrationViewer::main_menu() {
  // Only set intent flags inside the menu scope; dialogs and ProgressModal::open must run at the
  // top level of the UI callback (same ImGui scope as ProgressModal::run).
  bool start_open_bag = false;
  bool start_map = false;
  bool start_save = false;
  bool start_quit = false;

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Open bag", nullptr, false, !busy())) {
        start_open_bag = true;
      }
      if (ImGui::MenuItem("Load map (PCD)", nullptr, false, !busy())) {
        start_map = true;
      }

      ImGui::Separator();
      const bool any_result = std::any_of(lidars.begin(), lidars.end(), [](const auto& l) { return l.second.num_used > 0; });
      if (ImGui::MenuItem("Save calibration (YAML)", nullptr, false, any_result)) {
        start_save = true;
      }

      ImGui::Separator();
      if (ImGui::MenuItem("Quit")) {
        start_quit = true;
      }

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  if (start_open_bag) {
    guik::RecentFiles recent_files("lidar_calibration_bag");
    const std::string path = pfd::select_folder("Select a rosbag2 directory", recent_files.most_recent()).result();
    if (!path.empty()) {
      recent_files.push(path);
      open_bag(path);
      if (bag && odom_topic_index >= 0) {
        start_load_odometry();
      }
    }
  }
  if (start_map) {
    start_load_map("");
  }
  if (start_save) {
    save_calibration();
  }
  if (start_quit && pfd::message("Warning", "Quit?").result() == pfd::button::ok) {
    request_to_terminate = true;
  }
}

/*** Bag panel: topics, speed profile, time window ***/

void CalibrationViewer::draw_bag_panel() {
  ImGui::SetNextWindowSize(ImVec2(760, 0), ImGuiCond_FirstUseEver);
  ImGui::Begin("bag");

  if (!bag) {
    ImGui::TextUnformatted("No bag loaded (File > Open bag)");
    ImGui::End();
    return;
  }

  const bool is_busy = busy();
  if (is_busy) {
    ImGui::BeginDisabled();
  }

  ImGui::Text("Bag: %s (%.1f s)", bag->path().c_str(), bag->duration());

  /*** Odometry / speed profile ***/
  ImGui::Separator();
  const char* odom_preview = odom_topic_index >= 0 ? odom_topics[odom_topic_index].name.c_str() : "(none)";
  ImGui::SetNextItemWidth(320);
  if (ImGui::BeginCombo("odometry topic", odom_preview)) {
    for (int i = 0; i < static_cast<int>(odom_topics.size()); i++) {
      if (ImGui::Selectable(odom_topics[i].name.c_str(), i == odom_topic_index) && i != odom_topic_index) {
        odom_topic_index = i;
        task_queue.emplace_back([this] { start_load_odometry(); });
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("header stamps", &use_header_stamp)) {
    task_queue.emplace_back([this] { start_load_odometry(); });
  }
  show_note("Use header.stamp of the messages (otherwise the bag receive time) for the speed profile and the scan pairing.");

  if (odometry) {
    if (odometry->has_pose()) {
      ImGui::TextUnformatted("odometry pose: available (initial guess of the map alignment)");
    } else {
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "odometry pose: not available (twist only), map alignment needs a manual guess");
    }
  }

  draw_speed_plot();

  ImGui::SetNextItemWidth(320);
  float window[2] = {static_cast<float>(window_begin), static_cast<float>(window_end)};
  if (ImGui::DragFloat2("time window [s]", window, 0.1f, 0.0f, static_cast<float>(bag->duration()))) {
    window_begin = window[0];
    window_end = window[1];
  }
  show_note("Also adjustable by dragging the two vertical lines in the plot.");
  ImGui::SameLine();
  if (ImGui::Button("Use plot range")) {
    window_begin = plot_range_begin;
    window_end = plot_range_end;
  }
  window_begin = std::clamp(window_begin, 0.0, bag->duration());
  window_end = std::clamp(window_end, window_begin, bag->duration());

  /*** PointCloud2 topics ***/
  ImGui::Separator();
  bool roles_changed = false;
  if (ImGui::BeginTable("cloud_topics", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
    ImGui::TableSetupColumn("PointCloud2 topic", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("msgs");
    ImGui::TableSetupColumn("in window");
    ImGui::TableSetupColumn("reference");
    ImGui::TableSetupColumn("calibrate");
    ImGui::TableHeadersRow();

    for (const auto& topic : cloud_topics) {
      ImGui::PushID(topic.name.c_str());
      int& role = topic_roles[topic.name];

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(topic.name.c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%zu", topic.message_count);
      ImGui::TableNextColumn();
      const auto found = index.find(topic.name);
      if (found != index.end()) {
        ImGui::Text("%zu", found->second.size());
      } else {
        ImGui::TextUnformatted("-");
      }

      ImGui::TableNextColumn();
      bool is_ref = role == ROLE_REF;
      if (ImGui::Checkbox("##ref", &is_ref)) {
        for (auto& r : topic_roles) {
          if (r.second == ROLE_REF) {
            r.second = ROLE_NONE;
          }
        }
        role = is_ref ? ROLE_REF : ROLE_NONE;
        roles_changed = true;
      }

      ImGui::TableNextColumn();
      bool is_calib = role == ROLE_CALIB;
      if (ImGui::Checkbox("##calib", &is_calib)) {
        role = is_calib ? ROLE_CALIB : ROLE_NONE;
        roles_changed = true;
      }

      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  /*** Pairing ***/
  ImGui::Separator();
  bool pairing_changed = roles_changed;
  ImGui::SetNextItemWidth(120);
  pairing_changed |= ImGui::DragFloat("max time offset [ms]", &max_time_offset_ms, 1.0f, 1.0f, 500.0f);
  show_note("Maximum stamp difference between a reference scan and the scan of a calibrated LiDAR to form a pair.");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  pairing_changed |= ImGui::DragInt("frame stride", &frame_stride, 0.1f, 1, 1000);
  show_note("Use only every n-th reference scan in the window.");
  frame_stride = std::max(1, frame_stride);

  ImGui::SetNextItemWidth(120);
  ImGui::DragFloatRange2("range crop [m]", &min_range, &max_range, 0.1f, 0.0f, 1000.0f);
  show_note("Points outside this range are dropped from the bag scans (e.g., to remove the ego vehicle).\nApplied when scans are read.");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  ImGui::DragFloat("scan downsample [m]", &scan_downsample, 0.01f, 0.01f, 5.0f);

  const bool window_dirty = index.empty() || indexed_begin != window_begin || indexed_end != window_end || indexed_header_stamp != use_header_stamp;
  if (ImGui::Button("Index scans in window")) {
    task_queue.emplace_back([this] { start_index_window(); });
  }
  show_note("Read the stamps of all PointCloud2 messages in the time window and pair the scans.");
  if (window_dirty && !index.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "window / stamp source changed, re-index");
  }

  if (pairing_changed && !index.empty()) {
    rebuild_frames();
  }

  if (!frames.empty()) {
    ImGui::Text("%zu frames (reference scans with at least one partner)", frames.size());
    for (const auto& topic : calib_topics()) {
      const int num = std::count_if(frames.begin(), frames.end(), [&](const Frame& f) { return f.partners.count(topic); });
      ImGui::Text("  %s: %d pairs", topic.c_str(), num);
    }
  } else if (!index.empty()) {
    ImGui::TextUnformatted("No pairs: select a reference topic and topics to calibrate");
  }

  if (is_busy) {
    ImGui::EndDisabled();
  }
  ImGui::End();
}

void CalibrationViewer::draw_speed_plot() {
  if (speed_t.empty()) {
    ImGui::TextUnformatted(odom_topic_index >= 0 ? "Speed profile not loaded" : "No nav_msgs/msg/Odometry topic selected");
    return;
  }

  if (!ImPlot::BeginPlot("##speed_profile", ImVec2(-1, 260))) {
    return;
  }

  ImPlot::SetupAxes("time since bag start [s]", "speed [m/s]");
  ImPlot::SetupAxisLimits(ImAxis_X1, speed_t.front(), speed_t.back(), ImPlotCond_Once);

  const double v_max = std::max(1.0, *std::max_element(speed_v.begin(), speed_v.end()) * 1.05);
  const double xs[2] = {window_begin, window_end};
  const double y0[2] = {0.0, 0.0};
  const double y1[2] = {v_max, v_max};
  ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), 0.2f);
  ImPlot::PlotShaded("window", xs, y0, y1, 2);

  ImPlot::PlotLine("speed", speed_t.data(), speed_v.data(), speed_t.size());

  // Frames: not processed / used / rejected
  std::vector<double> xs_pending, ys_pending, xs_used, ys_used, xs_rejected, ys_rejected;
  for (const auto& frame : frames) {
    const double t = frame.ref.stamp - bag->start_time();
    const double v = speed_at(t);
    if (!frame.processed) {
      xs_pending.push_back(t);
      ys_pending.push_back(v);
    } else if (std::any_of(frame.results.begin(), frame.results.end(), [](const auto& r) { return r.second.used; })) {
      xs_used.push_back(t);
      ys_used.push_back(v);
    } else {
      xs_rejected.push_back(t);
      ys_rejected.push_back(v);
    }
  }
  ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 0.0f);
  ImPlot::PlotScatter("frames", xs_pending.data(), ys_pending.data(), xs_pending.size());
  ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ImVec4(0.2f, 0.85f, 0.2f, 1.0f), 0.0f);
  ImPlot::PlotScatter("used", xs_used.data(), ys_used.data(), xs_used.size());
  ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ImVec4(0.95f, 0.2f, 0.2f, 1.0f), 0.0f);
  ImPlot::PlotScatter("rejected", xs_rejected.data(), ys_rejected.data(), xs_rejected.size());

  if (loaded_frame && loaded_frame->index >= 0 && loaded_frame->index < static_cast<int>(frames.size())) {
    const double t = frames[loaded_frame->index].ref.stamp - bag->start_time();
    ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.35f, 0.35f, 1.0f));
    ImPlot::PlotInfLines("displayed frame", &t, 1);
  }

  const bool is_busy = busy();
  const ImPlotDragToolFlags drag_flags = is_busy ? ImPlotDragToolFlags_NoInputs : ImPlotDragToolFlags_None;
  ImPlot::DragLineX(0, &window_begin, ImVec4(0.3f, 0.6f, 1.0f, 1.0f), 2.0f, drag_flags);
  ImPlot::DragLineX(1, &window_end, ImVec4(0.3f, 0.6f, 1.0f, 1.0f), 2.0f, drag_flags);
  if (window_end < window_begin) {
    std::swap(window_begin, window_end);
  }

  const ImPlotRect limits = ImPlot::GetPlotLimits();
  plot_range_begin = std::max(0.0, limits.X.Min);
  plot_range_end = std::min(bag->duration(), limits.X.Max);

  ImPlot::EndPlot();
}

/*** Calibration panel: map, interactive alignment, batch, results ***/

void CalibrationViewer::draw_calibration_panel() {
  ImGui::SetNextWindowSize(ImVec2(560, 0), ImGuiCond_FirstUseEver);
  ImGui::Begin("calibration");

  const bool is_busy = busy();
  if (is_busy) {
    ImGui::BeginDisabled();
  }

  /*** Map ***/
  ImGui::SetNextItemWidth(80);
  ImGui::DragFloat("map downsample [m]", &map_downsample, 0.01f, 0.05f, 5.0f);
  show_note("Applied when the map is loaded.");
  if (map) {
    ImGui::Text("Map: %s (%d pts, %d downsampled)", map->path.c_str(), static_cast<int>(map->cloud->raw->size()), static_cast<int>(map->cloud->preprocessed->size()));
  } else {
    ImGui::TextUnformatted("Map: (not loaded)");
  }
  if (ImGui::Button("Load map (PCD)")) {
    task_queue.emplace_back([this] { start_load_map(""); });
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("show map", &show_map)) {
    update_drawables();
  }

  /*** Frame selection ***/
  ImGui::Separator();
  if (frames.empty()) {
    ImGui::TextUnformatted("No frames: index the scans in a time window first");
  } else {
    display_frame = std::clamp(display_frame, 0, static_cast<int>(frames.size()) - 1);
    ImGui::SetNextItemWidth(300);
    ImGui::SliderInt("frame", &display_frame, 0, frames.size() - 1);
    const bool released = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SameLine();
    const bool load = ImGui::Button("Load##frame") || released;
    if (load) {
      const int index = display_frame;
      task_queue.emplace_back([this, index] { start_load_frame(index); });
    }

    const Frame& frame = frames[display_frame];
    ImGui::Text("t = %.3f s, speed = %.2f m/s, %zu partners%s", frame.ref.stamp - bag->start_time(), speed_at(frame.ref.stamp - bag->start_time()), frame.partners.size(), display_frame == anchor_frame ? "  [anchor]" : "");
    for (const auto& [topic, partner] : frame.partners) {
      ImGui::Text("  %s: dt = %+.1f ms", topic.c_str(), (partner.stamp - frame.ref.stamp) * 1e3);
    }
  }

  /*** Interactive initial alignment ***/
  ImGui::Separator();
  ImGui::TextUnformatted("Initial alignment (on the loaded frame)");
  const bool frame_loaded = loaded_frame && loaded_frame->index >= 0 && loaded_frame->index < static_cast<int>(frames.size());

  if (!map || !frame_loaded) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("1. Align map to reference scan")) {
    open_align_map();
  }
  if (!map || !frame_loaded) {
    ImGui::EndDisabled();
  }
  ImGui::SameLine();
  if (map_aligned) {
    ImGui::Text("anchor frame %d", anchor_frame);
  } else {
    ImGui::TextUnformatted("[NOT aligned]");
  }

  const bool can_align_lidars = map_aligned && frame_loaded && loaded_frame->index == anchor_frame;
  for (const auto& topic : calib_topics()) {
    ImGui::PushID(topic.c_str());
    const Lidar& lidar = lidars[topic];
    const bool has_partner = can_align_lidars && loaded_frame->partners.count(topic);
    if (!has_partner) {
      ImGui::BeginDisabled();
    }
    if (ImGui::Button("2. Align")) {
      open_align_lidar(topic);
    }
    if (!has_partner) {
      ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::Text("%s %s%s", topic.c_str(), lidar.initialized ? "[initialized]" : "[NOT initialized]", lidar.rotation_only ? " (rotation only)" : "");
    ImGui::PopID();
  }
  if (map_aligned && frame_loaded && loaded_frame->index != anchor_frame) {
    ImGui::TextUnformatted("Load the anchor frame to align the LiDARs");
  }

  /*** Batch ***/
  ImGui::Separator();
  ImGui::TextUnformatted("Batch calibration (anchor frame to end of window)");
  ImGui::SetNextItemWidth(80);
  ImGui::DragFloat("max corr dist [m]", &batch_max_corr_dist, 0.01f, 0.05f, 10.0f);
  show_note("Maximum correspondence distance of the GICP registrations in the batch.");

  ImGui::SetNextItemWidth(80);
  bool filter_changed = ImGui::DragFloat("min inlier fraction", &min_inlier_fraction, 0.01f, 0.0f, 1.0f);
  show_note("Registrations (reference scan to map and scan to map) with a smaller inlier fraction are discarded.");
  ImGui::SameLine();
  filter_changed |= ImGui::Checkbox("reject outliers", &reject_outliers);
  show_note("Discard pairs whose rotation or translation deviates from the median by more than 3 sigma (MAD).");
  if (filter_changed) {
    recompute_statistics();
    update_drawables();
  }

  const bool any_initialized = std::any_of(lidars.begin(), lidars.end(), [this](const auto& l) {
    const auto topics = calib_topics();
    return l.second.initialized && std::find(topics.begin(), topics.end(), l.first) != topics.end();
  });
  const bool can_run = map && map_aligned && anchor_frame >= 0 && any_initialized;
  if (!can_run) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("3. Run batch calibration")) {
    task_queue.emplace_back([this] { start_batch(); });
  }
  if (!can_run) {
    ImGui::EndDisabled();
  }

  /*** Results ***/
  const int num_processed = std::count_if(frames.begin(), frames.end(), [](const Frame& f) { return f.processed; });
  if (num_processed) {
    ImGui::Separator();
    ImGui::Text("Results (%d frames processed)", num_processed);

    const std::string ref_name = ref_topic();
    for (const auto& topic : calib_topics()) {
      const Lidar& lidar = lidars[topic];
      if (!lidar.num_pairs) {
        continue;
      }

      const Eigen::Isometry3d& T = lidar.stats.mean;
      const Eigen::Vector3d t = T.translation();
      const Eigen::Quaterniond q = Eigen::Quaterniond(T.linear()).normalized();
      Eigen::Vector3d rpy;
      to_rpy(T, rpy);
      rpy *= 180.0 / M_PI;
      const Eigen::Vector3d t_std = lidar.stats.translation_std;
      const Eigen::Vector3d r_std = lidar.stats.rotation_std * 180.0 / M_PI;

      ImGui::Text("T_ref_lidar  %s  (%d / %d pairs used)", topic.c_str(), lidar.num_used, lidar.num_pairs);
      if (lidar.num_used) {
        ImGui::Text("  t_xyz  : [%.4f, %.4f, %.4f] m   std [%.4f, %.4f, %.4f]", t.x(), t.y(), t.z(), t_std.x(), t_std.y(), t_std.z());
        ImGui::Text("  q_xyzw : [%.6f, %.6f, %.6f, %.6f]", q.x(), q.y(), q.z(), q.w());
        ImGui::Text("  rpy    : [%.4f, %.4f, %.4f] deg std [%.4f, %.4f, %.4f]", rpy.x(), rpy.y(), rpy.z(), r_std.x(), r_std.y(), r_std.z());
      }
    }
  }

  if (is_busy) {
    ImGui::EndDisabled();
  }
  ImGui::End();
}

/*** Background tasks ***/

void CalibrationViewer::dispatch_tasks() {
  auto loaded_odometry = progress_modal->run<OdometryTrajectory::Ptr>("odometry");
  if (loaded_odometry) {
    active_task.clear();
    odometry = *loaded_odometry;
    speed_t.clear();
    speed_v.clear();
    if (odometry) {
      for (const auto& s : odometry->samples()) {
        speed_t.push_back(s.stamp - bag->start_time());
        speed_v.push_back(s.speed);
      }
      logger->info("loaded odometry ({} samples, {})", speed_t.size(), odometry->has_pose() ? "with pose" : "no pose");
    } else {
      pfd::message("Error", "Failed to read the odometry").result();
    }
  }

  auto loaded_map = progress_modal->run<LoadedMapPtr>("map");
  if (loaded_map) {
    active_task.clear();
    if (*loaded_map) {
      map = *loaded_map;
      map->cloud->buffer = std::make_shared<glk::PointCloudBuffer>(map->cloud->raw->points, map->cloud->raw->size());
      if (map->cloud->raw->has_intensities()) {
        map->cloud->buffer->add_intensity(
          glk::COLORMAP::TURBO,
          map->cloud->raw->intensities,
          map->cloud->raw->size(),
          1.0 / *std::max_element(map->cloud->raw->intensities, map->cloud->raw->intensities + map->cloud->raw->size()));
      }
      logger->info("loaded map {} ({} pts, {} after downsampling)", map->path, map->cloud->raw->size(), map->cloud->preprocessed->size());

      // The map changed: previous alignments refer to the old map
      map_aligned = false;
      anchor_frame = -1;
      map_in_ref = nullptr;
      clear_results();
      update_drawables();
    } else {
      pfd::message("Error", "Failed to load the map").result();
    }
  }

  auto indexed = progress_modal->run<std::shared_ptr<std::unordered_map<std::string, std::vector<BagStamp>>>>("index");
  if (indexed) {
    active_task.clear();
    if (*indexed) {
      index = std::move(**indexed);
      rebuild_frames();
    } else {
      pfd::message("Error", "Failed to index the bag").result();
    }
  }

  auto frame = progress_modal->run<LoadedFramePtr>("frame");
  if (frame) {
    active_task.clear();
    if (*frame) {
      loaded_frame = *frame;
      std::vector<CloudPtr> clouds = {loaded_frame->ref};
      for (const auto& p : loaded_frame->partners) {
        clouds.push_back(p.second);
      }
      for (auto& cloud : clouds) {
        cloud->buffer = std::make_shared<glk::PointCloudBuffer>(cloud->raw->points, cloud->raw->size());
      }
      update_drawables();
    } else {
      pfd::message("Error", "Failed to read the frame from the bag").result();
    }
  }

  auto batch = progress_modal->run<BatchResultPtr>("batch");
  if (batch) {
    active_task.clear();
    if (*batch) {
      frames = std::move((*batch)->frames);
      logger->info("batch calibration done ({} frames processed)", (*batch)->num_processed);
      recompute_statistics();
      for (const auto& topic : calib_topics()) {
        if (lidars[topic].num_used) {
          logger->info("{} / {} pairs used for {}", lidars[topic].num_used, lidars[topic].num_pairs, topic);
          log_extrinsics(topic, lidars[topic].stats.mean);
        }
      }
      update_drawables();
    } else {
      pfd::message("Error", "Batch calibration failed").result();
    }
  }
}

void CalibrationViewer::open_bag(const std::string& path) {
  try {
    bag = std::make_shared<CalibrationBag>(path);
  } catch (const std::exception& e) {
    logger->error("failed to open bag {}: {}", path, e.what());
    pfd::message("Error", "Failed to open bag " + path + "\n" + e.what()).result();
    return;
  }

  logger->info("opened bag {} ({:.1f} s)", path, bag->duration());

  cloud_topics = bag->topics_of_type("sensor_msgs/msg/PointCloud2");
  odom_topics = bag->topics_of_type("nav_msgs/msg/Odometry");

  topic_roles.clear();
  for (const auto& topic : cloud_topics) {
    logger->info("  PointCloud2: {} ({} msgs)", topic.name, topic.message_count);
    int role = ROLE_NONE;
    if (topic.name == options.ref_topic) {
      role = ROLE_REF;
    } else if (std::find(options.calib_topics.begin(), options.calib_topics.end(), topic.name) != options.calib_topics.end()) {
      role = ROLE_CALIB;
    }
    topic_roles[topic.name] = role;
  }

  odom_topic_index = -1;
  for (int i = 0; i < static_cast<int>(odom_topics.size()); i++) {
    if (odom_topics[i].name == options.odom_topic) {
      odom_topic_index = i;
    }
  }
  if (odom_topic_index < 0) {
    logger->warn("odometry topic {} not found in the bag ({} other nav_msgs/msg/Odometry topics)", options.odom_topic, odom_topics.size());
    if (!odom_topics.empty()) {
      odom_topic_index = 0;
    }
  }

  odometry = nullptr;
  speed_t.clear();
  speed_v.clear();
  window_begin = 0.0;
  window_end = bag->duration();
  plot_range_begin = window_begin;
  plot_range_end = window_end;

  index.clear();
  frames.clear();
  lidars.clear();
  loaded_frame = nullptr;
  map_aligned = false;
  anchor_frame = -1;
  map_in_ref = nullptr;
  update_drawables();
}

void CalibrationViewer::start_load_odometry() {
  if (!bag || odom_topic_index < 0) {
    return;
  }

  const std::string topic = odom_topics[odom_topic_index].name;
  const bool header_stamp = use_header_stamp;
  auto bag = this->bag;
  auto logger = this->logger;

  active_task = "odometry";
  progress_modal->open<OdometryTrajectory::Ptr>("odometry", [=](guik::ProgressInterface& progress) -> OdometryTrajectory::Ptr {
    try {
      return bag->read_odometry(topic, header_stamp, progress);
    } catch (const std::exception& e) {
      logger->error("failed to read the odometry from {}: {}", topic, e.what());
      return nullptr;
    }
  });
}

void CalibrationViewer::start_load_map(const std::string& path_hint) {
  std::string path = path_hint;
  if (path.empty()) {
    guik::RecentFiles recent_files("lidar_calibration_pcd");
    const std::vector<std::string> results = pfd::open_file("Select the map PCD file", recent_files.most_recent(), {"PCD files", "*.pcd"}).result();
    if (results.empty()) {
      return;
    }
    path = results[0];
    recent_files.push(path);
  }

  const double resolution = map_downsample;
  const int num_threads = this->num_threads;

  active_task = "map";
  progress_modal->open<LoadedMapPtr>("map", [=](guik::ProgressInterface& progress) -> LoadedMapPtr {
    progress.set_title("Loading map");
    progress.set_maximum(3);
    progress.set_text("Reading " + path);

    auto raw = load_pcd(path);
    if (!raw) {
      return nullptr;
    }
    progress.increment();

    auto loaded = std::make_shared<LoadedMap>();
    loaded->path = path;
    loaded->cloud = std::make_shared<Cloud>();
    loaded->cloud->raw = raw;
    loaded->cloud->preprocessed = preprocess_cloud(raw, resolution, num_threads, progress);
    progress.increment();

    progress.set_text("Building KdTree");
    loaded->tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(loaded->cloud->preprocessed, num_threads);
    progress.increment();

    return loaded;
  });
}

void CalibrationViewer::start_index_window() {
  if (!bag || cloud_topics.empty()) {
    return;
  }

  std::vector<std::string> topics;
  for (const auto& topic : cloud_topics) {
    topics.push_back(topic.name);
  }

  indexed_begin = window_begin;
  indexed_end = window_end;
  indexed_header_stamp = use_header_stamp;

  const double t_begin = bag->start_time() + window_begin;
  const double t_end = bag->start_time() + window_end;
  const bool header_stamp = use_header_stamp;
  auto bag = this->bag;
  auto logger = this->logger;

  using Index = std::unordered_map<std::string, std::vector<BagStamp>>;
  active_task = "index";
  progress_modal->open<std::shared_ptr<Index>>("index", [=](guik::ProgressInterface& progress) -> std::shared_ptr<Index> {
    try {
      return std::make_shared<Index>(bag->index_messages(topics, t_begin, t_end, header_stamp, progress));
    } catch (const std::exception& e) {
      logger->error("failed to index the bag: {}", e.what());
      return nullptr;
    }
  });
}

ScanParams CalibrationViewer::scan_params() const {
  ScanParams params;
  params.min_range = min_range;
  params.max_range = max_range;
  params.downsample_resolution = scan_downsample;
  params.num_threads = num_threads;
  return params;
}

void CalibrationViewer::start_load_frame(int frame_index) {
  if (!bag || frame_index < 0 || frame_index >= static_cast<int>(frames.size())) {
    return;
  }

  const Frame frame = frames[frame_index];
  const std::string ref = frames_ref_topic;
  std::set<CalibrationBag::MessageKey> keys = {{ref, frame.ref.recv_ns}};
  for (const auto& [topic, partner] : frame.partners) {
    keys.emplace(topic, partner.recv_ns);
  }
  const ScanParams scan_params = this->scan_params();
  auto bag = this->bag;
  auto logger = this->logger;

  active_task = "frame";
  progress_modal->open<LoadedFramePtr>("frame", [=](guik::ProgressInterface& progress) -> LoadedFramePtr {
    progress.set_title("Reading frame " + std::to_string(frame_index));

    auto loaded = std::make_shared<LoadedFrame>();
    loaded->index = frame_index;
    try {
      bag->read_clouds(
        keys,
        [&](const CalibrationBag::MessageKey& key, const sensor_msgs::msg::PointCloud2& msg) {
          auto cloud = std::make_shared<Cloud>();
          cloud->raw = convert_cloud(msg, scan_params.min_range, scan_params.max_range);
          if (!cloud->raw || cloud->raw->size() < 10) {
            logger->warn("too few valid points in the scan of {}", key.first);
            return true;
          }
          guik::ProgressInterface no_progress;
          cloud->preprocessed = preprocess_cloud(cloud->raw, scan_params.downsample_resolution, scan_params.num_threads, no_progress);
          if (key.first == ref) {
            loaded->ref = cloud;
          } else {
            loaded->partners[key.first] = cloud;
          }
          return true;
        },
        &progress);
    } catch (const std::exception& e) {
      logger->error("failed to read frame {}: {}", frame_index, e.what());
      return nullptr;
    }

    return loaded->ref ? loaded : nullptr;
  });
}

void CalibrationViewer::start_batch() {
  std::map<std::string, LidarInitialGuess> initial_guesses;
  for (const auto& topic : calib_topics()) {
    const Lidar& lidar = lidars[topic];
    if (lidar.initialized) {
      initial_guesses[topic].T_ref_lidar = lidar.T_init;
      initial_guesses[topic].rotation_only = lidar.rotation_only;
    }
  }

  FineRegistrationParams registration_params;
  registration_params.max_correspondence_distance = batch_max_corr_dist;
  registration_params.max_iterations = 30;
  registration_params.num_threads = num_threads;

  auto result = std::make_shared<BatchResult>();
  result->frames = frames;

  const std::string ref = frames_ref_topic;
  const int first_frame = anchor_frame;
  const Eigen::Isometry3d T_ref_map_first = T_ref_map_anchor;
  const OdometryTrajectory::ConstPtr odometry = this->odometry;
  const ScanParams scan_params = this->scan_params();
  const double inlier_threshold = min_inlier_fraction;
  auto bag = this->bag;
  auto map = this->map;
  auto logger = this->logger;

  active_task = "batch";
  progress_modal->open<BatchResultPtr>("batch", [=](guik::ProgressInterface& progress) -> BatchResultPtr {
    try {
      result->num_processed = run_batch_calibration(
        *bag,
        ref,
        result->frames,
        first_frame,
        T_ref_map_first,
        odometry,
        map->cloud->preprocessed,
        map->tree,
        initial_guesses,
        scan_params,
        registration_params,
        inlier_threshold,
        progress);
    } catch (const std::exception& e) {
      logger->error("batch calibration failed: {}", e.what());
      return nullptr;
    }
    return result;
  });
}

/*** Pairing and statistics ***/

std::string CalibrationViewer::ref_topic() const {
  for (const auto& [topic, role] : topic_roles) {
    if (role == ROLE_REF) {
      return topic;
    }
  }
  return "";
}

std::vector<std::string> CalibrationViewer::calib_topics() const {
  std::vector<std::string> topics;
  for (const auto& [topic, role] : topic_roles) {
    if (role == ROLE_CALIB) {
      topics.push_back(topic);
    }
  }
  return topics;
}

void CalibrationViewer::rebuild_frames() {
  // Keep the anchor (and its map alignment) if its reference scan is still part of the frames
  const int64_t anchor_recv_ns = (anchor_frame >= 0 && anchor_frame < static_cast<int>(frames.size())) ? frames[anchor_frame].ref.recv_ns : -1;
  const int64_t display_recv_ns = (loaded_frame && loaded_frame->index >= 0 && loaded_frame->index < static_cast<int>(frames.size())) ? frames[loaded_frame->index].ref.recv_ns : -1;
  const std::string prev_ref_topic = frames_ref_topic;

  frames.clear();
  frames_ref_topic = ref_topic();
  const auto ref_found = index.find(frames_ref_topic);
  if (frames_ref_topic.empty() || ref_found == index.end()) {
    anchor_frame = -1;
    map_aligned = false;
    map_in_ref = nullptr;
    loaded_frame = nullptr;
    clear_results();
    update_drawables();
    return;
  }

  frames = pair_scans(index, frames_ref_topic, calib_topics(), max_time_offset_ms * 1e-3, frame_stride);

  const auto find_frame = [&](int64_t recv_ns) -> int {
    if (recv_ns < 0 || frames_ref_topic != prev_ref_topic) {
      return -1;
    }
    const auto found = std::find_if(frames.begin(), frames.end(), [=](const Frame& f) { return f.ref.recv_ns == recv_ns; });
    return found == frames.end() ? -1 : std::distance(frames.begin(), found);
  };

  anchor_frame = find_frame(anchor_recv_ns);
  if (anchor_frame < 0) {
    map_aligned = false;
    map_in_ref = nullptr;
  }

  const int display = find_frame(display_recv_ns);
  if (loaded_frame) {
    loaded_frame->index = display;
    if (display >= 0) {
      display_frame = display;
    } else {
      loaded_frame = nullptr;
    }
  }

  logger->info("{} frames with reference {} (max offset {:.1f} ms, stride {})", frames.size(), frames_ref_topic, max_time_offset_ms, frame_stride);
  clear_results();
  update_drawables();
}

void CalibrationViewer::clear_results() {
  for (auto& frame : frames) {
    frame.processed = false;
    frame.results.clear();
  }
  for (auto& [topic, lidar] : lidars) {
    lidar.num_pairs = 0;
    lidar.num_used = 0;
    lidar.stats = TransformStatistics();
  }
}

void CalibrationViewer::recompute_statistics() {
  for (const auto& topic : calib_topics()) {
    Lidar& lidar = lidars[topic];
    lidar.stats = compute_lidar_statistics(frames, topic, min_inlier_fraction, reject_outliers, lidar.num_pairs, lidar.num_used);
  }
}

double CalibrationViewer::speed_at(double t) const {
  if (speed_t.empty()) {
    return 0.0;
  }
  const auto it = std::lower_bound(speed_t.begin(), speed_t.end(), t);
  if (it == speed_t.begin()) {
    return speed_v.front();
  }
  if (it == speed_t.end()) {
    return speed_v.back();
  }
  const size_t i = std::distance(speed_t.begin(), it);
  const double dt = speed_t[i] - speed_t[i - 1];
  const double p = dt > 1e-9 ? (t - speed_t[i - 1]) / dt : 0.0;
  return (1.0 - p) * speed_v[i - 1] + p * speed_v[i];
}

/*** Interactive alignment ***/

void CalibrationViewer::open_align_map() {
  if (!map || !loaded_frame || loaded_frame->index < 0) {
    return;
  }

  const Frame& frame = frames[loaded_frame->index];
  // Initial guess: batch result > odometry pose (T_map_ref) > previous manual alignment
  Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
  const auto T_map_ref_odom = odometry ? odometry->pose_at(frame.ref.stamp) : std::nullopt;
  if (frame.processed) {
    init = frame.T_ref_map;
  } else if (T_map_ref_odom) {
    init = T_map_ref_odom->inverse();
  } else if (map_aligned) {
    init = T_ref_map_anchor;
  }

  align_modal->set_clouds(
    "Align the map (green, movable) to the reference scan '" + frames_ref_topic + "' (red, fixed)\nThis frame becomes the anchor of the batch calibration",
    loaded_frame->ref->preprocessed,
    map->cloud->preprocessed,
    init);
  pending_align_map = true;
}

void CalibrationViewer::open_align_lidar(const std::string& topic) {
  if (!map_aligned || !loaded_frame || !loaded_frame->partners.count(topic)) {
    return;
  }

  if (!map_in_ref) {
    map_in_ref = gtsam_points::transform(map->cloud->preprocessed, T_ref_map_anchor);
  }

  align_modal->set_clouds(
    "Align scan '" + topic + "' (green, movable) to the map (red, fixed, in reference LiDAR frame)\nThe resulting transformation is T_ref_lidar (initial guess of the batch)",
    map_in_ref,
    loaded_frame->partners[topic]->preprocessed,
    lidars[topic].T_init);
  pending_align = topic;
}

void CalibrationViewer::on_aligned(const AlignResult& result) {
  if (pending_align_map) {
    anchor_frame = loaded_frame->index;
    T_ref_map_anchor = result.T_target_source;
    map_aligned = true;
    map_in_ref = nullptr;

    log_extrinsics("map", T_ref_map_anchor);
  } else if (!pending_align.empty()) {
    Lidar& lidar = lidars[pending_align];
    lidar.T_init = result.T_target_source;
    lidar.rotation_only = result.rotation_only;
    lidar.init_error = result.error;
    lidar.initialized = true;

    log_extrinsics(pending_align + " (initial guess)", lidar.T_init);
  }

  pending_align_map = false;
  pending_align.clear();
  update_drawables();
}

/*** Drawing / output ***/

void CalibrationViewer::update_drawables() {
  auto viewer = guik::LightViewer::instance();
  viewer->clear_drawables();

  Eigen::Affine3f origin_scale = Eigen::Affine3f::Identity();
  origin_scale.scale(2.0f);
  viewer->update_drawable("origin", glk::Primitives::coordinate_system(), guik::VertexColor(origin_scale));

  if (!loaded_frame || loaded_frame->index < 0 || loaded_frame->index >= static_cast<int>(frames.size())) {
    return;
  }

  const Frame& frame = frames[loaded_frame->index];
  viewer->update_drawable("ref_scan", loaded_frame->ref->buffer, guik::FlatColor(1.0f, 0.35f, 0.35f, 1.0f));

  // Map pose in the reference frame of the displayed frame (if known)
  std::optional<Eigen::Isometry3d> T_ref_map;
  if (frame.processed) {
    T_ref_map = frame.T_ref_map;
  } else if (map_aligned && loaded_frame->index == anchor_frame) {
    T_ref_map = T_ref_map_anchor;
  }

  if (map && show_map && T_ref_map) {
    const Eigen::Affine3f T = T_ref_map->cast<float>();
    if (map->cloud->raw->has_intensities()) {
      viewer->update_drawable("map", map->cloud->buffer, guik::VertexColor(T));
    } else {
      viewer->update_drawable("map", map->cloud->buffer, guik::FlatColor(0.7f, 0.7f, 0.7f, 1.0f, T));
    }
  }

  int color_index = 1;
  for (const auto& [topic, cloud] : loaded_frame->partners) {
    const Lidar& lidar = lidars[topic];
    Eigen::Isometry3d T_ref_lidar = Eigen::Isometry3d::Identity();
    if (lidar.num_used) {
      T_ref_lidar = lidar.stats.mean;
    } else if (lidar.initialized) {
      T_ref_lidar = lidar.T_init;
    }

    const Eigen::Affine3f T = T_ref_lidar.cast<float>();
    const Eigen::Vector4f color = glk::colormap_categoricalf(glk::COLORMAP::TURBO, color_index++, 8);
    viewer->update_drawable("scan_" + topic, cloud->buffer, guik::FlatColor(color.x(), color.y(), color.z(), color.w(), T));
    viewer->update_drawable("coord_" + topic, glk::Primitives::coordinate_system(), guik::VertexColor(T));
  }
}

void CalibrationViewer::log_extrinsics(const std::string& child_name, const Eigen::Isometry3d& T) const {
  const Eigen::Vector3d t = T.translation();
  const Eigen::Quaterniond q = Eigen::Quaterniond(T.linear()).normalized();
  Eigen::Vector3d rpy;
  to_rpy(T, rpy);

  logger->info("T_ref_{} (ref: {})", child_name, frames_ref_topic);
  logger->info("  t_xyz   : [{:.6f}, {:.6f}, {:.6f}] m", t.x(), t.y(), t.z());
  logger->info("  q_xyzw  : [{:.8f}, {:.8f}, {:.8f}, {:.8f}]", q.x(), q.y(), q.z(), q.w());
  logger->info("  rpy_rad : [{:.8f}, {:.8f}, {:.8f}]", rpy.x(), rpy.y(), rpy.z());
  logger->info("  rpy_deg : [{:.6f}, {:.6f}, {:.6f}]", rpy.x() * 180.0 / M_PI, rpy.y() * 180.0 / M_PI, rpy.z() * 180.0 / M_PI);
}

void CalibrationViewer::save_calibration() {
  guik::RecentFiles recent_files("lidar_calibration_save");
  std::string path = pfd::save_file("Save calibration", recent_files.most_recent(), {"YAML", "*.yaml"}).result();
  if (path.empty()) {
    return;
  }
  if (path.size() < 5 || path.substr(path.size() - 5) != ".yaml") {
    path += ".yaml";
  }
  recent_files.push(path);

  std::ofstream ofs(path);
  if (!ofs) {
    pfd::message("Error", "Failed to open " + path + " for writing").result();
    return;
  }

  const auto write_transform = [&ofs](const std::string& indent, const Eigen::Isometry3d& T) {
    const Eigen::Vector3d t = T.translation();
    const Eigen::Quaterniond q = Eigen::Quaterniond(T.linear()).normalized();
    Eigen::Vector3d rpy;
    to_rpy(T, rpy);

    ofs << indent << "translation: [" << t.x() << ", " << t.y() << ", " << t.z() << "]\n";
    ofs << indent << "quaternion_xyzw: [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]\n";
    ofs << indent << "rpy_rad: [" << rpy.x() << ", " << rpy.y() << ", " << rpy.z() << "]\n";
    ofs << indent << "rpy_deg: [" << rpy.x() * 180.0 / M_PI << ", " << rpy.y() * 180.0 / M_PI << ", " << rpy.z() * 180.0 / M_PI << "]\n";
  };
  const auto write_vector = [&ofs](const Eigen::Vector3d& v) { ofs << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]\n"; };

  ofs << std::setprecision(12);
  ofs << "# Multi-LiDAR extrinsic calibration result\n";
  ofs << "# T_ref_lidar transforms points from the LiDAR frame into the reference LiDAR frame\n";
  ofs << "# Average over all used scan pairs; rpy: extrinsic XYZ (roll-pitch-yaw) Euler angles\n";
  ofs << "bag: " << bag->path() << "\n";
  ofs << "map: " << (map ? map->path : "") << "\n";
  ofs << "reference_topic: " << frames_ref_topic << "\n";
  ofs << "time_window: [" << indexed_begin << ", " << indexed_end << "]  # [s] since bag start\n";
  ofs << "header_stamps: " << (indexed_header_stamp ? "true" : "false") << "\n";
  ofs << "max_time_offset: " << max_time_offset_ms * 1e-3 << "  # [s]\n";
  ofs << "min_inlier_fraction: " << min_inlier_fraction << "\n";
  ofs << "reject_outliers: " << (reject_outliers ? "true" : "false") << "\n";

  ofs << "lidars:\n";
  for (const auto& topic : calib_topics()) {
    const Lidar& lidar = lidars[topic];
    if (!lidar.num_used) {
      continue;
    }
    ofs << "  - topic: " << topic << "\n";
    ofs << "    num_pairs: " << lidar.num_pairs << "\n";
    ofs << "    num_used: " << lidar.num_used << "\n";
    ofs << "    rotation_only: " << (lidar.rotation_only ? "true" : "false") << "\n";
    ofs << "    T_ref_lidar:\n";
    write_transform("      ", lidar.stats.mean);
    ofs << "    translation_std: ";
    write_vector(lidar.stats.translation_std);
    ofs << "    rotation_std_deg: ";
    write_vector(lidar.stats.rotation_std * 180.0 / M_PI);
    ofs << "    T_ref_lidar_initial_guess:\n";
    write_transform("      ", lidar.T_init);
  }

  logger->info("saved calibration to {}", path);
}

}  // namespace glim
