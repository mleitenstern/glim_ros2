#include <glim_ros/lidar_calibration/align_modal.hpp>

#include <cmath>
#include <algorithm>

#include <spdlog/spdlog.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/graduated_non_convexity.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glim/util/convert_to_string.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/gl_canvas.hpp>
#include <guik/model_control.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

AlignModal::AlignModal(const std::shared_ptr<spdlog::logger>& logger, int num_threads) : num_threads(num_threads), request_to_open(false), active(false), logger(logger) {
  seed = 53123;
  fpfh_radius = 5.0f;
  global_registration_type = 0;
  global_registration_4dof = false;

  ransac_max_iterations = 5000;
  ransac_early_stop_rate = 0.9;
  ransac_inlier_voxel_resolution = 1.0;

  gnc_max_samples = 10000;

  max_correspondence_distance = 2.0f;
  rotation_only = false;
  last_error = -1.0;

  init_T_target_source.setIdentity();

  canvas.reset(new guik::GLCanvas(Eigen::Vector2i(768, 576)));
  progress_modal.reset(new guik::ProgressModal("align_modal_progress"));
  model_control.reset(new guik::ModelControl("align_model_control"));

#ifdef GTSAM_USE_TBB
  tbb_task_arena = std::make_shared<tbb::task_arena>(1);
#endif
}

AlignModal::~AlignModal() {}

bool AlignModal::is_active() const {
  return active;
}

void AlignModal::set_clouds(
  const std::string& description,
  const gtsam_points::PointCloudCPU::Ptr& target,
  const gtsam_points::PointCloudCPU::Ptr& source,
  const Eigen::Isometry3d& init_T_target_source) {
  //
  this->description = description;
  this->target = target;
  this->source = source;
  this->init_T_target_source = init_T_target_source;

  this->target_drawable = std::make_shared<glk::PointCloudBuffer>(target->points, target->size());
  this->source_drawable = std::make_shared<glk::PointCloudBuffer>(source->points, source->size());

  this->target_fpfh_tree = nullptr;
  this->source_fpfh_tree = nullptr;

  this->last_error = -1.0;
  this->request_to_open = true;
  this->active = true;
}

void AlignModal::clear() {
  target = nullptr;
  source = nullptr;
  target_drawable = nullptr;
  source_drawable = nullptr;
  target_fpfh_tree = nullptr;
  source_fpfh_tree = nullptr;
  active = false;
}

std::optional<AlignResult> AlignModal::run() {
  std::optional<AlignResult> result;

  if (request_to_open && target && source) {
    model_control->set_model_matrix(init_T_target_source.cast<float>().matrix());
    ImGui::OpenPopup("align clouds");
  }
  request_to_open = false;

  if (ImGui::BeginPopupModal("align clouds", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted(description.c_str());
    ImGui::Text("target (red): %d pts   source (green): %d pts", static_cast<int>(target->size()), static_cast<int>(source->size()));

    // Draw canvas
    ImGui::BeginChild(
      "canvas",
      ImVec2(768, 576),
      false,
      ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoNavFocus);
    if (ImGui::IsWindowFocused() && !model_control->is_guizmo_using()) {
      canvas->mouse_control();
    }
    draw_canvas();
    ImGui::Image(reinterpret_cast<void*>(canvas->frame_buffer->color().id()), ImVec2(768, 576), ImVec2(0, 1), ImVec2(1, 0));

    ImVec2 canvas_rect_min = ImGui::GetItemRectMin();
    ImVec2 canvas_rect_max = ImGui::GetItemRectMax();

    model_control->set_gizmo_operation(rotation_only ? "ROTATE" : "UNIVERSAL");
    model_control->draw_gizmo(
      canvas_rect_min.x,
      canvas_rect_min.y,
      canvas_rect_max.x - canvas_rect_min.x,
      canvas_rect_max.y - canvas_rect_min.y,
      canvas->camera_control->view_matrix(),
      canvas->projection_control->projection_matrix(),
      true);

    ImGui::EndChild();

    /*** Initial guess ***/

    ImGui::Separator();
    draw_guess_ui();

    ImGui::Checkbox("Rotation only", &rotation_only) ||
      show_note("Constrain the fine registration to rotation angles only.\nThe translation is fixed to the values above (e.g., known from CAD data).");

    /*** Global registration ***/

    ImGui::Separator();
    ImGui::TextUnformatted("Global registration (automatic initial guess)");

    if (rotation_only) {
      ImGui::BeginDisabled();
    }

    ImGui::Combo("Global registration type", &global_registration_type, "RANSAC\0GNC\0");

    if (ImGui::DragFloat("fpfh_radius", &fpfh_radius, 0.01f, 0.01f, 100.0f) || show_note("Neighbor search radius for FPFH extraction.\n~2.5m for indoors, ~5.0m for outdoors.")) {
      target->aux_attributes.erase("fpfh");
      source->aux_attributes.erase("fpfh");
      target_fpfh_tree = nullptr;
      source_fpfh_tree = nullptr;
    }
    if (target->aux_attributes.count("fpfh")) {
      ImGui::SameLine();
      ImGui::Text("[Cached]");
    }

    switch (global_registration_type) {
      case 0:  // RANSAC
        ImGui::DragInt("max_iterations", &ransac_max_iterations, 100, 1, 100000) || show_note("Maximum number of RANSAC iterations.");
        ImGui::DragFloat("inlier_voxel_resolution", &ransac_inlier_voxel_resolution, 0.01f, 0.01f, 100.0f) || show_note("Resolution of voxelmap used for inlier check.");
        break;
      case 1:  // GNC
        ImGui::DragInt("max_samples", &gnc_max_samples, 100, 1, 100000) || show_note("Maximum number of feature samples for GNC.");
        break;
    }
    ImGui::Checkbox("4dof", &global_registration_4dof) ||
      show_note("Use 4DoF (XYZ + RZ) estimation instead of 6DoF (SE3).\nOnly valid if both clouds are roughly gravity aligned (Z up).");

    bool open_align_global_modal = false;
    if (ImGui::Button("Run global registration")) {
      open_align_global_modal = true;
    }

    if (rotation_only) {
      ImGui::EndDisabled();
    }

    /*** Fine registration ***/

    ImGui::Separator();
    ImGui::TextUnformatted("Fine registration (GICP)");
    ImGui::DragFloat("max_corr_dist", &max_correspondence_distance, 0.01f, 0.01f, 100.0f) || show_note("Maximum correspondence distance for scan matching.");

    bool open_align_modal = false;
    if (ImGui::Button("Run fine registration")) {
      open_align_modal = true;
    }

    if (last_error >= 0.0) {
      ImGui::SameLine();
      ImGui::Text("error: %.3f", last_error);
    }

    if (open_align_global_modal) {
      progress_modal->open<std::shared_ptr<Eigen::Isometry3d>>("align", [this](guik::ProgressInterface& progress) { return align_global(progress); });
    }
    if (open_align_modal) {
      progress_modal->open<std::shared_ptr<Eigen::Isometry3d>>("align", [this](guik::ProgressInterface& progress) { return align(progress); });
    }
    auto align_result = progress_modal->run<std::shared_ptr<Eigen::Isometry3d>>("align");
    if (align_result) {
      model_control->set_model_matrix((*align_result)->cast<float>().matrix());
    }

    /*** Accept / Cancel ***/

    ImGui::Separator();
    if (ImGui::Button("Accept")) {
      AlignResult res;
      res.T_target_source = Eigen::Isometry3d(model_control->model_matrix().cast<double>());
      res.error = last_error;
      result = res;

      ImGui::CloseCurrentPopup();
      clear();
    }
    show_note("Accept the current transformation as calibration result.");

    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
      clear();
    }

    ImGui::EndPopup();
  }

  return result;
}

void AlignModal::draw_guess_ui() {
  const Eigen::Isometry3f T(model_control->model_matrix());
  const Eigen::Matrix3f R = T.linear();

  Eigen::Vector3f xyz = T.translation();
  Eigen::Vector3f rpy;
  rpy.x() = std::atan2(R(2, 1), R(2, 2));
  rpy.y() = std::asin(std::max(-1.0f, std::min(1.0f, -R(2, 0))));
  rpy.z() = std::atan2(R(1, 0), R(0, 0));
  rpy *= 180.0f / M_PI;

  bool edited = false;
  edited |= ImGui::DragFloat3("translation xyz [m]", xyz.data(), 0.01f);
  show_note("Initial guess translation (e.g., from CAD data).");
  edited |= ImGui::DragFloat3("rotation rpy [deg]", rpy.data(), 0.5f);
  show_note("Initial guess rotation as extrinsic XYZ (roll-pitch-yaw) Euler angles.");

  if (edited) {
    rpy *= M_PI / 180.0f;
    Eigen::Isometry3f T_new = Eigen::Isometry3f::Identity();
    T_new.linear() =
      (Eigen::AngleAxisf(rpy.z(), Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(rpy.y(), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(rpy.x(), Eigen::Vector3f::UnitX()))
        .toRotationMatrix();
    T_new.translation() = xyz;
    model_control->set_model_matrix(T_new.matrix());
  }
}

std::shared_ptr<Eigen::Isometry3d> AlignModal::align_global(guik::ProgressInterface& progress) {
  logger->info("running global registration");
  progress.set_title("Global registration");
  progress.set_maximum(10);

  progress.increment();
  logger->info("Creating KdTree");
  progress.set_text("Creating KdTree");
  auto target_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(target);
  progress.increment();
  auto source_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(source);

  gtsam_points::FPFHEstimationParams fpfh_params;
  fpfh_params.num_threads = num_threads;
  fpfh_params.search_radius = fpfh_radius;

  progress.increment();
  if (!target->aux_attributes.count("fpfh")) {
    if (!target->has_normals()) {
      logger->info("Estimating target normals");
      progress.set_text("Estimating target normals");
      target->add_normals(gtsam_points::estimate_normals(target->points, target->covs, target->size(), num_threads));
    }

    logger->info("Estimating target FPFH features");
    progress.set_text("Estimating target FPFH features");
    const auto fpfh = gtsam_points::estimate_fpfh(target->points, target->normals, target->size(), *target_tree, fpfh_params);
    target->add_aux_attribute("fpfh", fpfh);
  }

  if (!target_fpfh_tree) {
    logger->info("Constructing target FPFH KdTree");
    progress.set_text("Constructing target FPFH KdTree");
    const auto target_fpfh = target->aux_attribute<gtsam_points::FPFHSignature>("fpfh");
    target_fpfh_tree = std::make_shared<gtsam_points::KdTreeX<gtsam_points::FPFH_DIM>>(target_fpfh, target->size());
  }

  progress.increment();
  if (!source->aux_attributes.count("fpfh")) {
    if (!source->has_normals()) {
      logger->info("Estimating source normals");
      progress.set_text("Estimating source normals");
      source->add_normals(gtsam_points::estimate_normals(source->points, source->covs, source->size(), num_threads));
    }

    logger->info("Estimating source FPFH features");
    progress.set_text("Estimating source FPFH features");
    const auto fpfh = gtsam_points::estimate_fpfh(source->points, source->normals, source->size(), *source_tree, fpfh_params);
    source->add_aux_attribute("fpfh", fpfh);
  }

  if (!source_fpfh_tree) {
    logger->info("Constructing source FPFH KdTree");
    progress.set_text("Constructing source FPFH KdTree");
    const auto source_fpfh = source->aux_attribute<gtsam_points::FPFHSignature>("fpfh");
    source_fpfh_tree = std::make_shared<gtsam_points::KdTreeX<gtsam_points::FPFH_DIM>>(source_fpfh, source->size());
  }

  const auto target_fpfh = target->aux_attribute<gtsam_points::FPFHSignature>("fpfh");
  const auto source_fpfh = source->aux_attribute<gtsam_points::FPFHSignature>("fpfh");

  progress.increment();

  gtsam_points::RegistrationResult result;

  if (global_registration_type == 0) {
    logger->info("Estimating transformation RANSAC (seed={})", seed);
    progress.set_text("Estimating transformation RANSAC");

    gtsam_points::RANSACParams ransac_params;
    ransac_params.max_iterations = ransac_max_iterations;
    ransac_params.early_stop_inlier_rate = ransac_early_stop_rate;
    ransac_params.inlier_voxel_resolution = ransac_inlier_voxel_resolution;
    ransac_params.dof = global_registration_4dof ? 4 : 6;
    ransac_params.seed = (seed += 4322);
    ransac_params.num_threads = num_threads;

    result = gtsam_points::estimate_pose_ransac(*target, *source, target_fpfh, source_fpfh, *target_tree, *target_fpfh_tree, ransac_params);
  } else {
    logger->info("Estimating transformation GNC (seed={})", seed);
    progress.set_text("Estimating transformation (GNC)");

    gtsam_points::GNCParams gnc_params;
    gnc_params.max_init_samples = gnc_max_samples;
    gnc_params.reciprocal_check = true;
    gnc_params.tuple_check = false;
    gnc_params.max_num_tuples = 5000;
    gnc_params.dof = global_registration_4dof ? 4 : 6;
    gnc_params.seed = (seed += 4322);
    gnc_params.num_threads = num_threads;

    result = gtsam_points::estimate_pose_gnc(*target, *source, target_fpfh, source_fpfh, *target_tree, *target_fpfh_tree, *source_fpfh_tree, gnc_params);
  }

  logger->info("Global registration result");
  logger->info("T_target_source={}", convert_to_string(result.T_target_source));
  logger->info("inlier_rate={}", result.inlier_rate);

  return std::make_shared<Eigen::Isometry3d>(result.T_target_source);
}

std::shared_ptr<Eigen::Isometry3d> AlignModal::align(guik::ProgressInterface& progress) {
  progress.set_title("Fine registration");
  const int num_iterations = 30;
  progress.set_maximum(num_iterations);

  progress.set_text("Creating graph");
  const Eigen::Isometry3d init_T(model_control->model_matrix().cast<double>());

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

  if (rotation_only) {
    // Anchor the translation of T_target_source to the initial guess (rotation stays free)
    gtsam::Vector6 sigmas;
    sigmas << 1e6, 1e6, 1e6, 1e-6, 1e-6, 1e-6;
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1, gtsam::Pose3(init_T.matrix()), gtsam::noiseModel::Diagonal::Sigmas(sigmas));
  }

  auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source);
  factor->set_num_threads(num_threads);
  factor->set_max_correspondence_distance(max_correspondence_distance);
  graph.add(factor);

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Identity());
  values.insert(1, gtsam::Pose3(init_T.matrix()));

  progress.set_text("Optimizing");
  double final_error = -1.0;

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(num_iterations);
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    progress.increment();
    progress.set_text(fmt::format("Optimizing iter:{} error:{:.3f}", status.iterations, status.error));
    final_error = status.error;
  };

  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

#ifdef GTSAM_USE_TBB
  auto arena = static_cast<tbb::task_arena*>(tbb_task_arena.get());
  arena->execute([&] {
#endif
    values = optimizer.optimize();
#ifdef GTSAM_USE_TBB
  });
#endif

  last_error = final_error;

  const gtsam::Pose3 estimated_pose = values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1);
  Eigen::Isometry3d estimated(estimated_pose.matrix());
  if (rotation_only) {
    estimated.translation() = init_T.translation();
  }

  return std::make_shared<Eigen::Isometry3d>(estimated);
}

void AlignModal::draw_canvas() {
  if (!target_drawable || !source_drawable) {
    return;
  }

  canvas->bind();
  canvas->shader->set_uniform("color_mode", guik::ColorMode::VERTEX_COLOR);
  canvas->shader->set_uniform("model_matrix", Eigen::Matrix4f::Identity().eval());

  glk::Primitives::coordinate_system()->draw(*canvas->shader);

  canvas->shader->set_uniform("color_mode", guik::ColorMode::FLAT_COLOR);
  canvas->shader->set_uniform("material_color", Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
  canvas->shader->set_uniform("model_matrix", Eigen::Matrix4f::Identity().eval());

  target_drawable->draw(*canvas->shader);

  canvas->shader->set_uniform("material_color", Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));
  canvas->shader->set_uniform("model_matrix", model_control->model_matrix());

  source_drawable->draw(*canvas->shader);

  canvas->unbind();
}

bool AlignModal::show_note(const std::string& note) {
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("%s", note.c_str());
    ImGui::EndTooltip();
  }
  return false;
}

}  // namespace glim
