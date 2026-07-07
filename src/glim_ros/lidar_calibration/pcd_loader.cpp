#include <glim_ros/lidar_calibration/pcd_loader.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <spdlog/spdlog.h>

#include <gtsam_points/ann/kdtree2.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <guik/progress_modal.hpp>

namespace glim {

namespace {

struct PCDField {
  std::string name;
  int size = 4;    // bytes per element
  char type = 'F';  // F: float, I: signed int, U: unsigned int
  int count = 1;
};

struct PCDHeader {
  std::vector<PCDField> fields;
  size_t num_points = 0;
  std::string data;  // ascii, binary, binary_compressed
};

/// @brief LZF decompression as used by PCL for binary_compressed data
bool lzf_decompress(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
  size_t ip = 0;
  size_t op = 0;

  while (ip < in.size()) {
    const uint32_t ctrl = in[ip++];

    if (ctrl < 32) {
      // Literal run of ctrl + 1 bytes
      const uint32_t len = ctrl + 1;
      if (ip + len > in.size() || op + len > out.size()) {
        return false;
      }
      std::memcpy(out.data() + op, in.data() + ip, len);
      ip += len;
      op += len;
    } else {
      // Back reference of len + 2 bytes
      uint32_t len = ctrl >> 5;
      if (ip >= in.size()) {
        return false;
      }
      if (len == 7) {
        len += in[ip++];
        if (ip >= in.size()) {
          return false;
        }
      }
      len += 2;

      const int64_t ref = static_cast<int64_t>(op) - ((ctrl & 0x1f) << 8) - 1 - in[ip++];
      if (ref < 0 || op + len > out.size()) {
        return false;
      }
      // Byte-wise copy (regions may overlap)
      for (uint32_t i = 0; i < len; i++) {
        out[op + i] = out[ref + i];
      }
      op += len;
    }
  }

  return op == out.size();
}

bool parse_header(std::istream& ifs, PCDHeader& header) {
  size_t width = 0;
  size_t height = 1;
  std::vector<int> sizes;
  std::vector<char> types;
  std::vector<int> counts;

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::string tag;
    iss >> tag;

    if (tag == "FIELDS") {
      std::string name;
      while (iss >> name) {
        PCDField field;
        field.name = name;
        header.fields.push_back(field);
      }
    } else if (tag == "SIZE") {
      int size;
      while (iss >> size) {
        sizes.push_back(size);
      }
    } else if (tag == "TYPE") {
      char type;
      while (iss >> type) {
        types.push_back(type);
      }
    } else if (tag == "COUNT") {
      int count;
      while (iss >> count) {
        counts.push_back(count);
      }
    } else if (tag == "WIDTH") {
      iss >> width;
    } else if (tag == "HEIGHT") {
      iss >> height;
    } else if (tag == "POINTS") {
      iss >> header.num_points;
    } else if (tag == "DATA") {
      iss >> header.data;
      break;
    }
    // VERSION and VIEWPOINT are ignored
  }

  if (header.data.empty() || header.fields.empty()) {
    return false;
  }
  if (sizes.size() != header.fields.size() || types.size() != header.fields.size()) {
    return false;
  }
  for (size_t i = 0; i < header.fields.size(); i++) {
    header.fields[i].size = sizes[i];
    header.fields[i].type = types[i];
    header.fields[i].count = i < counts.size() ? counts[i] : 1;
  }
  if (header.num_points == 0) {
    header.num_points = width * height;
  }

  return header.num_points > 0;
}

double read_scalar(const uint8_t* data, char type, int size) {
  switch (type) {
    case 'F':
      if (size == 4) {
        float v;
        std::memcpy(&v, data, 4);
        return v;
      } else if (size == 8) {
        double v;
        std::memcpy(&v, data, 8);
        return v;
      }
      break;
    case 'I':
      if (size == 1) {
        int8_t v;
        std::memcpy(&v, data, 1);
        return v;
      } else if (size == 2) {
        int16_t v;
        std::memcpy(&v, data, 2);
        return v;
      } else if (size == 4) {
        int32_t v;
        std::memcpy(&v, data, 4);
        return v;
      }
      break;
    case 'U':
      if (size == 1) {
        return data[0];
      } else if (size == 2) {
        uint16_t v;
        std::memcpy(&v, data, 2);
        return v;
      } else if (size == 4) {
        uint32_t v;
        std::memcpy(&v, data, 4);
        return v;
      }
      break;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

gtsam_points::PointCloudCPU::Ptr load_pcd(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    spdlog::error("failed to open {}", path);
    return nullptr;
  }

  PCDHeader header;
  if (!parse_header(ifs, header)) {
    spdlog::error("failed to parse PCD header of {}", path);
    return nullptr;
  }

  int x_field = -1;
  int y_field = -1;
  int z_field = -1;
  int intensity_field = -1;
  for (size_t i = 0; i < header.fields.size(); i++) {
    if (header.fields[i].name == "x") {
      x_field = i;
    } else if (header.fields[i].name == "y") {
      y_field = i;
    } else if (header.fields[i].name == "z") {
      z_field = i;
    } else if (header.fields[i].name == "intensity") {
      intensity_field = i;
    }
  }

  if (x_field < 0 || y_field < 0 || z_field < 0) {
    spdlog::error("missing x/y/z fields in {}", path);
    return nullptr;
  }

  const size_t num_points = header.num_points;
  std::vector<Eigen::Vector4d> raw_points(num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
  std::vector<double> raw_intensities(num_points, 0.0);

  if (header.data == "ascii") {
    // Token index of each field (fields with count > 1 span multiple tokens)
    std::vector<int> token_indices(header.fields.size());
    int num_tokens = 0;
    for (size_t i = 0; i < header.fields.size(); i++) {
      token_indices[i] = num_tokens;
      num_tokens += header.fields[i].count;
    }

    std::string line;
    std::vector<double> tokens(num_tokens);
    for (size_t i = 0; i < num_points; i++) {
      if (!std::getline(ifs, line)) {
        spdlog::error("unexpected end of file in {}", path);
        return nullptr;
      }

      std::istringstream iss(line);
      for (int j = 0; j < num_tokens; j++) {
        if (!(iss >> tokens[j])) {
          tokens[j] = std::numeric_limits<double>::quiet_NaN();
        }
      }

      raw_points[i] << tokens[token_indices[x_field]], tokens[token_indices[y_field]], tokens[token_indices[z_field]], 1.0;
      if (intensity_field >= 0) {
        raw_intensities[i] = tokens[token_indices[intensity_field]];
      }
    }
  } else if (header.data == "binary" || header.data == "binary_compressed") {
    // Byte offsets
    std::vector<size_t> aos_offsets(header.fields.size());  // offset within a point record (array of structures)
    std::vector<size_t> soa_offsets(header.fields.size());  // offset of the field block (structure of arrays)
    size_t point_step = 0;
    for (size_t i = 0; i < header.fields.size(); i++) {
      aos_offsets[i] = point_step;
      soa_offsets[i] = point_step * num_points;
      point_step += header.fields[i].size * header.fields[i].count;
    }

    std::vector<uint8_t> data(point_step * num_points);

    if (header.data == "binary") {
      if (!ifs.read(reinterpret_cast<char*>(data.data()), data.size())) {
        spdlog::error("failed to read binary data from {}", path);
        return nullptr;
      }
    } else {
      uint32_t compressed_size = 0;
      uint32_t uncompressed_size = 0;
      if (!ifs.read(reinterpret_cast<char*>(&compressed_size), 4) || !ifs.read(reinterpret_cast<char*>(&uncompressed_size), 4)) {
        spdlog::error("failed to read compressed data header from {}", path);
        return nullptr;
      }

      std::vector<uint8_t> compressed(compressed_size);
      if (!ifs.read(reinterpret_cast<char*>(compressed.data()), compressed.size())) {
        spdlog::error("failed to read compressed data from {}", path);
        return nullptr;
      }

      data.resize(uncompressed_size);
      if (!lzf_decompress(compressed, data)) {
        spdlog::error("failed to decompress data of {}", path);
        return nullptr;
      }
    }

    // binary: array of structures / binary_compressed: structure of arrays
    const bool compressed = header.data == "binary_compressed";
    const auto field_ptr = [&](int field, size_t point_index) -> const uint8_t* {
      if (compressed) {
        return data.data() + soa_offsets[field] + point_index * header.fields[field].size * header.fields[field].count;
      }
      return data.data() + point_index * point_step + aos_offsets[field];
    };

    for (size_t i = 0; i < num_points; i++) {
      raw_points[i] << read_scalar(field_ptr(x_field, i), header.fields[x_field].type, header.fields[x_field].size),
        read_scalar(field_ptr(y_field, i), header.fields[y_field].type, header.fields[y_field].size),
        read_scalar(field_ptr(z_field, i), header.fields[z_field].type, header.fields[z_field].size), 1.0;
      if (intensity_field >= 0) {
        raw_intensities[i] = read_scalar(field_ptr(intensity_field, i), header.fields[intensity_field].type, header.fields[intensity_field].size);
      }
    }
  } else {
    spdlog::error("unsupported PCD data type {} in {}", header.data, path);
    return nullptr;
  }

  // Drop non-finite points
  std::vector<Eigen::Vector4d> points;
  std::vector<double> intensities;
  points.reserve(num_points);
  intensities.reserve(num_points);

  bool has_intensity = false;
  for (size_t i = 0; i < num_points; i++) {
    if (!raw_points[i].head<3>().allFinite()) {
      continue;
    }
    points.push_back(raw_points[i]);

    const double intensity = std::isfinite(raw_intensities[i]) ? raw_intensities[i] : 0.0;
    intensities.push_back(intensity);
    has_intensity = has_intensity || intensity > 0.0;
  }

  if (points.empty()) {
    spdlog::error("no valid points in {}", path);
    return nullptr;
  }

  auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
  if (has_intensity && intensity_field >= 0) {
    frame->add_intensities(intensities);
  }
  return frame;
}

gtsam_points::PointCloudCPU::Ptr preprocess_cloud(
  const gtsam_points::PointCloud::ConstPtr& points,
  double downsample_resolution,
  int num_threads,
  guik::ProgressInterface& progress) {
  //
  progress.set_text("Downsampling");
  auto downsampled = gtsam_points::voxelgrid_sampling(points, downsample_resolution, num_threads);

  progress.set_text("Finding neighbors");
  const int k = 10;
  gtsam_points::KdTree2<gtsam_points::PointCloud> tree(downsampled);
  std::vector<int> neighbors(downsampled->size() * k);

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < downsampled->size(); i++) {
    std::vector<size_t> k_indices(k);
    std::vector<double> k_sq_dists(k);
    tree.knn_search(downsampled->points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
  }

  progress.set_text("Estimating normals and covariances");
  glim::CloudCovarianceEstimation covest(num_threads);
  covest.estimate(downsampled->points_storage, neighbors, downsampled->normals_storage, downsampled->covs_storage);

  downsampled->normals = downsampled->normals_storage.data();
  downsampled->covs = downsampled->covs_storage.data();

  return downsampled;
}

}  // namespace glim
