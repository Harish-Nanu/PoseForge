#pragma once

#include <optional>
#include <vector>

#include "poseforge/camera.hpp"
#include "poseforge/types.hpp"

namespace poseforge {

// 3D corners of a square marker of side `length` in its own frame (z = 0 plane),
// in the same order as MarkerDetection::corners.
std::vector<cv::Point3f> marker_object_points(double length);

// Solves the Perspective-n-Point problem for one square marker using OpenCV's IPPE-square
// solver. Returns nullopt if the solver fails.
std::optional<MarkerPose> estimate_pose(const MarkerDetection& detection, double marker_length,
                                        const CameraIntrinsics& camera);

std::vector<MarkerPose> estimate_poses(const std::vector<MarkerDetection>& detections,
                                       double marker_length, const CameraIntrinsics& camera);

// Angle in degrees of the rotation that takes `a` to `b`.
double rotation_error_deg(const Pose& a, const Pose& b);

// Euclidean distance between the two translations, in the units of the poses.
double translation_error(const Pose& a, const Pose& b);

}  // namespace poseforge
