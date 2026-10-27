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

}  // namespace poseforge
