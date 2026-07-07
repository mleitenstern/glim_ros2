#include <glim_ros/lidar_calibration/calibration_viewer.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <portable-file-dialogs.h>

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

void to_rpy(const Eigen::Isometry3d& T, Eigen::Vector3d& rpy) {
  const Eigen::Matrix3d R = T.linear();
  rpy.x() = std::atan2(R(2, 1), R(2, 2));
  rpy.y() = std::asin(std::max(-1.0, std::min(1.0, -R(2, 0))));
  rpy.z() = std::atan2(R(1, 0), R(0, 0));
}

}  // namespace

CalibrationViewer::CalibrationViewer(int num_threads) : num_threads(num_threads), request_to_terminate(false) {
  map_downsample = 0.25f;
  scan_downsample = 0.1f;

  T_ref_map.setIdentity();

  loading_kind = -1;
  pending_align = PENDING_NONE;

  logger = spdlog::default_logger();
}

CalibrationViewer::~CalibrationViewer() {}

void CalibrationViewer::preload(const std::string& ref_scan_path, const std::string& map_path, const std::vector<std::string>& scan_paths) {
  if (!ref_scan_path.empty()) {
    pending_loads.emplace_back(KIND_REF, ref_scan_path);
  }
  if (!map_path.empty()) {
    pending_loads.emplace_back(KIND_MAP, map_path);
  }
  for (const auto& path : scan_paths) {
    pending_loads.emplace_back(KIND_SCAN, path);
  }
}

void CalibrationViewer::run() {
  auto viewer = guik::LightViewer::instance(Eigen::Vector2i(1920, 1080));

  progress_modal.reset(new guik::ProgressModal("calibration_progress"));
  align_modal.reset(new AlignModal(logger, num_threads));

  viewer->register_ui_callback("calibration_ui", [this] { ui_callback(); });

  while (!request_to_terminate && viewer->spin_once()) {
  }

  viewer->register_ui_callback("calibration_ui", nullptr);
  align_modal.reset();
  progress_modal.reset();
}

void CalibrationViewer::ui_callback() {
  main_menu();
  draw_panel();

  // Dispatch finished loading jobs
  auto loaded = progress_modal->run<LidarCloudPtr>("load");
  if (loaded) {
    on_loaded(*loaded);
  }

  // Alignment modal
  auto align_result = align_modal->run();
  if (align_result) {
    on_aligned(*align_result);
  } else if (!align_modal->is_active()) {
    pending_align = PENDING_NONE;
  }

  // Kick queued loads (from command line preload)
  if (!pending_loads.empty() && loading_kind < 0 && !align_modal->is_active()) {
    const auto [kind, path] = pending_loads.front();
    pending_loads.pop_front();
    start_load(kind, path);
  }
}

void CalibrationViewer::main_menu() {
  // Only set intent flags inside the menu scope. The actual work (file dialogs,
  // ProgressModal::open -> ImGui::OpenPopup) must run at the top level of the UI
  // callback so that ProgressModal::run's BeginPopupModal is in the same ImGui
  // scope; otherwise the worker thread is never joined and the next load calls
  // std::terminate by move-assigning onto a joinable std::thread.
  int start_load_kind = -1;
  bool start_save = false;
  bool start_quit = false;

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      const bool busy = loading_kind >= 0 || align_modal->is_active();

      if (ImGui::MenuItem("1. Load reference scan (PCD)", nullptr, false, !busy)) {
        start_load_kind = KIND_REF;
      }
      if (ImGui::MenuItem("2. Load map (PCD)", nullptr, false, !busy && ref_scan != nullptr)) {
        start_load_kind = KIND_MAP;
      }
      if (ImGui::MenuItem("3. Add LiDAR scan (PCD)", nullptr, false, !busy && map && map->aligned)) {
        start_load_kind = KIND_SCAN;
      }

      ImGui::Separator();
      const bool any_aligned = std::any_of(scans.begin(), scans.end(), [](const LidarCloudPtr& s) { return s->aligned; });
      if (ImGui::MenuItem("Save calibration (YAML)", nullptr, false, any_aligned)) {
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

  if (start_load_kind >= 0 && loading_kind < 0 && !align_modal->is_active()) {
    start_load(start_load_kind, "");
  }
  if (start_save) {
    save_calibration();
  }
  if (start_quit && pfd::message("Warning", "Quit?").result() == pfd::button::ok) {
    request_to_terminate = true;
  }
}

void CalibrationViewer::draw_panel() {
  ImGui::Begin("calibration", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::TextUnformatted("Downsampling resolution [m] (applied on load)");
  ImGui::DragFloat("map##downsample", &map_downsample, 0.01f, 0.05f, 5.0f);
  ImGui::DragFloat("scan##downsample", &scan_downsample, 0.01f, 0.05f, 5.0f);

  const bool busy = loading_kind >= 0 || align_modal->is_active();

  ImGui::Separator();
  if (ref_scan) {
    ImGui::Text("Reference: %s (%d pts)", ref_scan->name.c_str(), static_cast<int>(ref_scan->raw->size()));
  } else {
    ImGui::TextUnformatted("Reference: (not loaded)");
  }

  if (map) {
    ImGui::Text("Map: %s (%d pts) %s", map->name.c_str(), static_cast<int>(map->raw->size()), map->aligned ? "[aligned]" : "[NOT aligned]");
    ImGui::SameLine();
    if (!busy && ImGui::Button("Align##map")) {
      open_align_map();
    }
  } else {
    ImGui::TextUnformatted("Map: (not loaded)");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("LiDAR scans:");
  for (int i = 0; i < static_cast<int>(scans.size()); i++) {
    const auto& scan = scans[i];
    ImGui::Text("%s (%d pts) %s", scan->name.c_str(), static_cast<int>(scan->raw->size()), scan->aligned ? "[aligned]" : "[NOT aligned]");
    ImGui::SameLine();
    if (!busy && map && map->aligned && ImGui::Button(("Align##scan" + std::to_string(i)).c_str())) {
      open_align_scan(i);
    }

    if (scan->aligned) {
      const Eigen::Vector3d t = scan->T_ref.translation();
      const Eigen::Quaterniond q = Eigen::Quaterniond(scan->T_ref.linear()).normalized();
      Eigen::Vector3d rpy;
      to_rpy(scan->T_ref, rpy);
      rpy *= 180.0 / M_PI;

      ImGui::Text("  t_xyz  : [%.4f, %.4f, %.4f] m", t.x(), t.y(), t.z());
      ImGui::Text("  q_xyzw : [%.6f, %.6f, %.6f, %.6f]", q.x(), q.y(), q.z(), q.w());
      ImGui::Text("  rpy    : [%.4f, %.4f, %.4f] deg", rpy.x(), rpy.y(), rpy.z());
      if (scan->error >= 0.0) {
        ImGui::Text("  error  : %.3f", scan->error);
      }
    }
  }

  ImGui::End();
}

void CalibrationViewer::start_load(int kind, const std::string& path_hint) {
  std::string path = path_hint;
  if (path.empty()) {
    guik::RecentFiles recent_files("lidar_calibration_pcd");
    const std::vector<std::string> results = pfd::open_file("Select a PCD file", recent_files.most_recent(), {"PCD files", "*.pcd"}).result();
    if (results.empty()) {
      return;
    }
    path = results[0];
    recent_files.push(path);
  }

  const double resolution = (kind == KIND_MAP) ? map_downsample : scan_downsample;

  loading_kind = kind;
  progress_modal->open<LidarCloudPtr>("load", [this, path, resolution](guik::ProgressInterface& progress) { return load_cloud(progress, path, resolution); });
}

CalibrationViewer::LidarCloudPtr CalibrationViewer::load_cloud(guik::ProgressInterface& progress, const std::string& path, double resolution) const {
  progress.set_title("Loading point cloud");
  progress.set_maximum(4);
  progress.set_text("Reading " + path);
  progress.increment();

  auto raw = load_pcd(path);
  if (!raw) {
    return nullptr;
  }
  progress.increment();

  auto preprocessed = preprocess_cloud(raw, resolution, num_threads, progress);
  progress.increment();

  auto cloud = std::make_shared<LidarCloud>();
  cloud->path = path;
  cloud->name = std::filesystem::path(path).stem().string();
  cloud->raw = raw;
  cloud->preprocessed = preprocessed;

  progress.set_text("Done");
  return cloud;
}

void CalibrationViewer::on_loaded(const LidarCloudPtr& cloud) {
  const int kind = loading_kind;
  loading_kind = -1;

  if (!cloud) {
    pfd::message("Error", "Failed to load point cloud").result();
    return;
  }

  logger->info("loaded {} ({} pts, {} after downsampling)", cloud->path, cloud->raw->size(), cloud->preprocessed->size());
  cloud->buffer = std::make_shared<glk::PointCloudBuffer>(cloud->raw->points, cloud->raw->size());
  if (cloud->raw->has_intensities()) {
    cloud->buffer->add_intensity(
      glk::COLORMAP::TURBO,
      cloud->raw->intensities,
      cloud->raw->size(),
      1.0 / *std::max_element(cloud->raw->intensities, cloud->raw->intensities + cloud->raw->size()));
  }

  switch (kind) {
    case KIND_REF:
      reset_session();
      ref_scan = cloud;
      break;

    case KIND_MAP:
      map = cloud;
      open_align_map();
      break;

    case KIND_SCAN: {
      // Ensure unique names (used as drawable and output keys)
      int suffix = 2;
      const std::string base_name = cloud->name;
      while (std::any_of(scans.begin(), scans.end(), [&](const LidarCloudPtr& s) { return s->name == cloud->name; })) {
        cloud->name = base_name + "_" + std::to_string(suffix++);
      }

      scans.push_back(cloud);
      open_align_scan(scans.size() - 1);
      break;
    }
  }

  update_drawables();
}

void CalibrationViewer::open_align_map() {
  const Eigen::Isometry3d init = map->aligned ? T_ref_map : Eigen::Isometry3d::Identity();
  align_modal->set_clouds("Align the map (green, movable) to the reference scan '" + ref_scan->name + "' (red, fixed)", ref_scan->preprocessed, map->preprocessed, init);
  pending_align = PENDING_MAP;
}

void CalibrationViewer::open_align_scan(int index) {
  const auto& scan = scans[index];
  align_modal->set_clouds(
    "Align scan '" + scan->name + "' (green, movable) to the map (red, fixed, in reference LiDAR frame)\nThe resulting transformation is T_" + ref_scan->name + "_" + scan->name,
    map_in_ref,
    scan->preprocessed,
    scan->T_ref);
  pending_align = index;
}

void CalibrationViewer::on_aligned(const AlignResult& result) {
  if (pending_align == PENDING_MAP) {
    const Eigen::Isometry3d T_old = T_ref_map;
    T_ref_map = result.T_target_source;

    map->T_ref = T_ref_map;
    map->aligned = true;
    map->error = result.error;

    map_in_ref = gtsam_points::transform(map->preprocessed, T_ref_map);

    // Keep already aligned scans consistent with the updated map pose
    const Eigen::Isometry3d remap = T_ref_map * T_old.inverse();
    for (auto& scan : scans) {
      if (scan->aligned) {
        scan->T_ref = remap * scan->T_ref;
      }
    }

    log_extrinsics("map", T_ref_map, result.error);
  } else if (pending_align >= 0 && pending_align < static_cast<int>(scans.size())) {
    auto& scan = scans[pending_align];
    scan->T_ref = result.T_target_source;
    scan->aligned = true;
    scan->error = result.error;

    log_extrinsics(scan->name, scan->T_ref, scan->error);
  }

  pending_align = PENDING_NONE;
  update_drawables();
}

void CalibrationViewer::update_drawables() {
  auto viewer = guik::LightViewer::instance();

  Eigen::Affine3f origin_scale = Eigen::Affine3f::Identity();
  origin_scale.scale(2.0f);
  viewer->update_drawable("origin", glk::Primitives::coordinate_system(), guik::VertexColor(origin_scale));

  if (ref_scan) {
    viewer->update_drawable("ref_scan", ref_scan->buffer, guik::FlatColor(1.0f, 0.35f, 0.35f, 1.0f));
  }

  if (map) {
    const Eigen::Affine3f T = map->T_ref.cast<float>();
    if (map->raw->has_intensities()) {
      viewer->update_drawable("map", map->buffer, guik::VertexColor(T));
    } else {
      viewer->update_drawable("map", map->buffer, guik::FlatColor(0.7f, 0.7f, 0.7f, 1.0f, T));
    }
  }

  for (int i = 0; i < static_cast<int>(scans.size()); i++) {
    const auto& scan = scans[i];
    const Eigen::Affine3f T = scan->T_ref.cast<float>();
    const Eigen::Vector4f color = glk::colormap_categoricalf(glk::COLORMAP::TURBO, i + 1, 8);

    viewer->update_drawable("scan_" + scan->name, scan->buffer, guik::FlatColor(color.x(), color.y(), color.z(), color.w(), T));
    if (scan->aligned) {
      viewer->update_drawable("coord_" + scan->name, glk::Primitives::coordinate_system(), guik::VertexColor(T));
    }
  }
}

void CalibrationViewer::reset_session() {
  auto viewer = guik::LightViewer::instance();

  if (map) {
    viewer->remove_drawable("map");
  }
  for (const auto& scan : scans) {
    viewer->remove_drawable("scan_" + scan->name);
    viewer->remove_drawable("coord_" + scan->name);
  }

  ref_scan = nullptr;
  map = nullptr;
  scans.clear();
  map_in_ref = nullptr;
  T_ref_map.setIdentity();
  pending_align = PENDING_NONE;
}

void CalibrationViewer::log_extrinsics(const std::string& child_name, const Eigen::Isometry3d& T, double error) const {
  const std::string ref_name = ref_scan ? ref_scan->name : "ref";

  const Eigen::Vector3d t = T.translation();
  const Eigen::Quaterniond q = Eigen::Quaterniond(T.linear()).normalized();
  Eigen::Vector3d rpy;
  to_rpy(T, rpy);

  logger->info("T_{}_{}", ref_name, child_name);
  logger->info("  t_xyz   : [{:.6f}, {:.6f}, {:.6f}] m", t.x(), t.y(), t.z());
  logger->info("  q_xyzw  : [{:.8f}, {:.8f}, {:.8f}, {:.8f}]", q.x(), q.y(), q.z(), q.w());
  logger->info("  rpy_rad : [{:.8f}, {:.8f}, {:.8f}]", rpy.x(), rpy.y(), rpy.z());
  logger->info("  rpy_deg : [{:.6f}, {:.6f}, {:.6f}]", rpy.x() * 180.0 / M_PI, rpy.y() * 180.0 / M_PI, rpy.z() * 180.0 / M_PI);
  if (error >= 0.0) {
    logger->info("  fine registration error : {:.4f}", error);
  }
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

    ofs << std::setprecision(12);
    ofs << indent << "translation: [" << t.x() << ", " << t.y() << ", " << t.z() << "]\n";
    ofs << indent << "quaternion_xyzw: [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]\n";
    ofs << indent << "rpy_rad: [" << rpy.x() << ", " << rpy.y() << ", " << rpy.z() << "]\n";
    ofs << indent << "rpy_deg: [" << rpy.x() * 180.0 / M_PI << ", " << rpy.y() * 180.0 / M_PI << ", " << rpy.z() * 180.0 / M_PI << "]\n";
  };

  ofs << "# Multi-LiDAR extrinsic calibration result\n";
  ofs << "# T_ref_x transforms points from frame x into the reference LiDAR frame\n";
  ofs << "# rpy: extrinsic XYZ (roll-pitch-yaw) Euler angles\n";
  ofs << "reference: " << ref_scan->name << "\n";

  if (map && map->aligned) {
    ofs << "map:\n";
    ofs << "  path: " << map->path << "\n";
    ofs << "  T_ref_map:\n";
    write_transform("    ", T_ref_map);
  }

  ofs << "lidars:\n";
  for (const auto& scan : scans) {
    if (!scan->aligned) {
      continue;
    }
    ofs << "  - name: " << scan->name << "\n";
    ofs << "    path: " << scan->path << "\n";
    if (scan->error >= 0.0) {
      ofs << "    fine_registration_error: " << scan->error << "\n";
    }
    ofs << "    T_ref_lidar:\n";
    write_transform("      ", scan->T_ref);
  }

  logger->info("saved calibration to {}", path);
}

}  // namespace glim
