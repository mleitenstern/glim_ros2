#include <iostream>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <boost/program_options.hpp>

#include <glim/util/logging.hpp>
#include <glim_ros/lidar_calibration/calibration_viewer.hpp>

int main(int argc, char** argv) {
  using namespace boost::program_options;
  options_description desc("Multi-LiDAR extrinsic calibration tool");
  desc.add_options()                                                                             //
    ("help", "produce help message")                                                             //
    ("ref_scan", value<std::string>(), "Scan of the reference LiDAR (.pcd)")                     //
    ("map", value<std::string>(), "Pointcloud map of the scene (.pcd)")                          //
    ("scan", value<std::vector<std::string>>()->multitoken(), "Scans of the other LiDARs (.pcd)")  //
    ("num_threads", value<int>()->default_value(0), "Number of threads (0: hardware concurrency)")  //
    ("debug", "Enable debug printing")                                                          //
    ;

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).run(), vm);
  notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // Setup logger
  auto logger = spdlog::stdout_color_mt("lidar_calibration");
  logger->sinks().push_back(glim::get_ringbuffer_sink());
  if (vm.count("debug")) {
    logger->sinks().push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/lidar_calibration_log.log", true));
    logger->set_level(spdlog::level::trace);
  }
  spdlog::set_default_logger(logger);

  int num_threads = vm["num_threads"].as<int>();
  if (num_threads <= 0) {
    num_threads = std::max(1u, std::thread::hardware_concurrency());
  }

  const std::string ref_scan_path = vm.count("ref_scan") ? vm["ref_scan"].as<std::string>() : "";
  const std::string map_path = vm.count("map") ? vm["map"].as<std::string>() : "";
  const std::vector<std::string> scan_paths = vm.count("scan") ? vm["scan"].as<std::vector<std::string>>() : std::vector<std::string>();

  glim::CalibrationViewer viewer(num_threads);
  viewer.preload(ref_scan_path, map_path, scan_paths);
  viewer.run();

  return 0;
}
