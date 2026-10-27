#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include "poseforge/camera.hpp"
#include "poseforge/types.hpp"

namespace poseforge {

// One marker placed in a synthetic scene at a known pose.
struct SceneMarker {
    int id = 0;
    Pose pose;
};

// Rotation that makes a marker face the camera (marker z axis pointing at the viewer).
cv::Vec3d facing_camera_rotation();

// Pose for a marker `distance` metres in front of the camera, centred at (x, y), tilted by
// `tilt_x_deg` / `tilt_y_deg` around the marker's own x / y axes.
Pose make_marker_pose(double distance, double x = 0.0, double y = 0.0,
                      double tilt_x_deg = 0.0, double tilt_y_deg = 0.0);

// Renders printed ArUco markers into images with a perspective warp. Because the pose of each
// marker is known exactly, the frames double as ground truth for tests and benchmarks.
class SyntheticScene {
public:
    SyntheticScene(CameraIntrinsics camera, double marker_length_m,
                   cv::aruco::PredefinedDictionaryType dictionary);

    // noise_stddev adds Gaussian pixel noise (0 = clean image).
    cv::Mat render(const std::vector<SceneMarker>& markers, double noise_stddev = 0.0,
                   std::uint64_t seed = 0) const;

    const CameraIntrinsics& camera() const { return camera_; }
    double marker_length() const { return marker_length_; }

private:
    CameraIntrinsics camera_;
    double marker_length_;
    cv::aruco::Dictionary dictionary_;
    cv::Mat background_;
};

}  // namespace poseforge
