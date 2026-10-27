#pragma once

#include <opencv2/core.hpp>

namespace poseforge {

// Pinhole camera model: intrinsic matrix K and lens distortion coefficients.
struct CameraIntrinsics {
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);  // k1, k2, p1, p2, k3
    cv::Size image_size;

    double fx() const { return K(0, 0); }
    double fy() const { return K(1, 1); }
    double cx() const { return K(0, 2); }
    double cy() const { return K(1, 2); }

    // Ideal camera (no distortion) with the given horizontal field of view in degrees.
    static CameraIntrinsics from_fov(cv::Size size, double horizontal_fov_deg);
};

}  // namespace poseforge
