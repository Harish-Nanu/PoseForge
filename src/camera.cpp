#include "poseforge/camera.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace poseforge {

CameraIntrinsics CameraIntrinsics::from_fov(cv::Size size, double horizontal_fov_deg) {
    if (size.width <= 0 || size.height <= 0) {
        throw std::invalid_argument("image size must be positive");
    }
    if (horizontal_fov_deg <= 0.0 || horizontal_fov_deg >= 180.0) {
        throw std::invalid_argument("field of view must be in (0, 180) degrees");
    }
    const double half_fov = horizontal_fov_deg * std::numbers::pi / 360.0;
    const double f = (size.width / 2.0) / std::tan(half_fov);

    CameraIntrinsics cam;
    cam.K = cv::Matx33d(f, 0.0, size.width / 2.0,
                        0.0, f, size.height / 2.0,
                        0.0, 0.0, 1.0);
    cam.image_size = size;
    return cam;
}

CameraIntrinsics CameraIntrinsics::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("cannot open calibration file: " + path);
    }
    cv::Mat K;
    cv::Mat dist;
    int width = 0;
    int height = 0;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    if (K.rows != 3 || K.cols != 3) {
        throw std::runtime_error("calibration file has no 3x3 camera_matrix: " + path);
    }

    CameraIntrinsics cam;
    K.convertTo(K, CV_64F);
    cam.K = cv::Matx33d(K);
    if (!dist.empty()) {
        dist.convertTo(cam.distortion, CV_64F);
        cam.distortion = cam.distortion.reshape(1, 1);
    }
    cam.image_size = {width, height};
    return cam;
}

void CameraIntrinsics::save(const std::string& path) const {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("cannot write calibration file: " + path);
    }
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "camera_matrix" << cv::Mat(K);
    fs << "distortion_coefficients" << distortion;
}

}  // namespace poseforge
