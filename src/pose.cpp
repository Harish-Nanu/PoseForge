#include "poseforge/pose.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/calib3d.hpp>

namespace poseforge {

std::vector<cv::Point3f> marker_object_points(double length) {
    const auto h = static_cast<float>(length / 2.0);
    return {{-h, h, 0.f}, {h, h, 0.f}, {h, -h, 0.f}, {-h, -h, 0.f}};
}

std::optional<MarkerPose> estimate_pose(const MarkerDetection& detection, double marker_length,
                                        const CameraIntrinsics& camera) {
    const auto object_points = marker_object_points(marker_length);
    const std::vector<cv::Point2f> image_points(detection.corners.begin(),
                                                detection.corners.end());

    // IPPE_SQUARE is the solver designed for exactly this case: four coplanar corners of a
    // square, in a fixed order. It is fast and avoids the planar ambiguity issues of iterative PnP.
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    if (!cv::solvePnP(object_points, image_points, camera.K, camera.distortion, rvec, tvec, false,
                      cv::SOLVEPNP_IPPE_SQUARE)) {
        return std::nullopt;
    }

    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(object_points, rvec, tvec, camera.K, camera.distortion, reprojected);
    double squared = 0.0;
    for (std::size_t i = 0; i < reprojected.size(); ++i) {
        const cv::Point2f diff = reprojected[i] - image_points[i];
        squared += static_cast<double>(diff.dot(diff));
    }

    MarkerPose result;
    result.id = detection.id;
    result.pose = {rvec, tvec};
    result.reprojection_error_px = std::sqrt(squared / static_cast<double>(reprojected.size()));
    return result;
}

std::vector<MarkerPose> estimate_poses(const std::vector<MarkerDetection>& detections,
                                       double marker_length, const CameraIntrinsics& camera) {
    std::vector<MarkerPose> poses;
    poses.reserve(detections.size());
    for (const auto& d : detections) {
        if (auto p = estimate_pose(d, marker_length, camera)) {
            poses.push_back(*p);
        }
    }
    return poses;
}

double rotation_error_deg(const Pose& a, const Pose& b) {
    cv::Matx33d Ra;
    cv::Matx33d Rb;
    cv::Rodrigues(a.rvec, Ra);
    cv::Rodrigues(b.rvec, Rb);
    const cv::Matx33d delta = Ra.t() * Rb;
    // trace(R) = 1 + 2 cos(theta)
    const double cos_theta = std::clamp((cv::trace(delta) - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(cos_theta) * 180.0 / std::numbers::pi;
}

double translation_error(const Pose& a, const Pose& b) { return cv::norm(a.tvec - b.tvec); }

}  // namespace poseforge
