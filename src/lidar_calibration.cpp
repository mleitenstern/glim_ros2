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
  options_description desc("Multi-LiDAR extrinsic calibration tool\nUsage: lidar_calibration <rosbag> [options]");
  desc.add_options()                                                                                                    //
    ("help", "produce help message")                                                                                    //
    ("bag", value<std::string>(), "rosbag2 (directory or .mcap) containing the LiDAR scans")                             //
    ("map", value<std::string>(), "Pointcloud map of the scene (.pcd)")                                                 //
    ("odom_topic", value<std::string>()->default_value("/core/state/odometry"), "nav_msgs/Odometry topic (speed profile)")  //
    ("ref_topic", value<std::string>(), "PointCloud2 topic of the reference LiDAR (preselection)")                      //
    ("calib_topic", value<std::vector<std::string>>()->multitoken(), "PointCloud2 topics to calibrate (preselection)")  //
    ("max_time_offset", value<double>()->default_value(0.05, "0.05"), "Maximum stamp offset of a scan pair [s]")                //
    ("num_threads", value<int>()->default_value(0), "Number of threads (0: hardware concurrency)")                      //
    ("debug", "Enable debug printing")                                                                                 //
    ;

  positional_options_description positional;
  positional.add("bag", 1);

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
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

  glim::CalibrationViewer::Options options;
  options.bag_path = vm.count("bag") ? vm["bag"].as<std::string>() : "";
  options.map_path = vm.count("map") ? vm["map"].as<std::string>() : "";
  options.odom_topic = vm["odom_topic"].as<std::string>();
  options.ref_topic = vm.count("ref_topic") ? vm["ref_topic"].as<std::string>() : "";
  options.calib_topics = vm.count("calib_topic") ? vm["calib_topic"].as<std::vector<std::string>>() : std::vector<std::string>();
  options.max_time_offset = vm["max_time_offset"].as<double>();

  glim::CalibrationViewer viewer(num_threads, options);
  viewer.run();

  return 0;
}
