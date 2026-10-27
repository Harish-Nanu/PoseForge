#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "poseforge/camera.hpp"
#include "poseforge/types.hpp"

namespace poseforge {

// Draws a wireframe cube of side `length` standing on the marker (the AR overlay), plus the
// marker's coordinate axes and id. Markers behind the camera are skipped.
void draw_cube(cv::Mat& frame, const MarkerPose& marker, double length,
               const CameraIntrinsics& camera);

void draw_overlay(cv::Mat& frame, const std::vector<MarkerPose>& markers, double marker_length,
                  const CameraIntrinsics& camera);

}  // namespace poseforge
