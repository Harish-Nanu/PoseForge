#pragma once

#include <array>

#include <opencv2/core.hpp>

namespace poseforge {

// Rigid transform from the marker frame to the camera frame (OpenCV convention:
// camera x right, y down, z forward). rvec is a Rodrigues rotation vector.
struct Pose {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
};

// A marker found in an image. Corners are in ArUco order:
// top-left, top-right, bottom-right, bottom-left (as printed on the marker).
struct MarkerDetection {
    int id = -1;
    std::array<cv::Point2f, 4> corners;
};

// A detected marker with its estimated 6-DoF pose.
struct MarkerPose {
    int id = -1;
    Pose pose;
    double reprojection_error_px = 0.0;
};

}  // namespace poseforge
