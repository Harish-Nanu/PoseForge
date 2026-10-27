#include "poseforge/synthetic.hpp"

#include <numbers>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

namespace poseforge {

namespace {

constexpr int kMarkerPixels = 240;   // resolution of the printed marker texture
constexpr int kBorderPixels = 40;    // white quiet zone around the black square

double deg2rad(double deg) { return deg * std::numbers::pi / 180.0; }

cv::Matx33d rodrigues(const cv::Vec3d& rvec) {
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    return R;
}

cv::Vec3d to_rvec(const cv::Matx33d& R) {
    cv::Vec3d rvec;
    cv::Rodrigues(R, rvec);
    return rvec;
}

// Soft gradient + checker pattern so the detector has some clutter to reject.
cv::Mat make_background(cv::Size size) {
    cv::Mat bg(size, CV_8UC3);
    for (int y = 0; y < size.height; ++y) {
        auto* row = bg.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size.width; ++x) {
            const int checker = ((x / 64) + (y / 64)) % 2 == 0 ? 20 : 0;
            const int base = 90 + (60 * x) / size.width + checker;
            row[x] = cv::Vec3b(static_cast<uchar>(base), static_cast<uchar>(base + 10),
                               static_cast<uchar>(base + 20));
        }
    }
    return bg;
}

}  // namespace

cv::Vec3d facing_camera_rotation() {
    // 180 degrees about x: marker y (up) maps to image up, marker z points back at the camera.
    return {std::numbers::pi, 0.0, 0.0};
}

Pose make_marker_pose(double distance, double x, double y, double tilt_x_deg, double tilt_y_deg) {
    const cv::Matx33d facing = rodrigues(facing_camera_rotation());
    const cv::Matx33d tilt_x = rodrigues({deg2rad(tilt_x_deg), 0.0, 0.0});
    const cv::Matx33d tilt_y = rodrigues({0.0, deg2rad(tilt_y_deg), 0.0});
    return {to_rvec(facing * tilt_x * tilt_y), {x, y, distance}};
}

SyntheticScene::SyntheticScene(CameraIntrinsics camera, double marker_length_m,
                               cv::aruco::PredefinedDictionaryType dictionary)
    : camera_(std::move(camera)),
      marker_length_(marker_length_m),
      dictionary_(cv::aruco::getPredefinedDictionary(dictionary)),
      background_(make_background(camera_.image_size)) {}

cv::Mat SyntheticScene::render(const std::vector<SceneMarker>& markers, double noise_stddev,
                               std::uint64_t seed) const {
    cv::Mat frame = background_.clone();

    const int texture_size = kMarkerPixels + 2 * kBorderPixels;
    const auto half = static_cast<float>(marker_length_ / 2.0);
    const float half_outer =
        half * static_cast<float>(texture_size) / static_cast<float>(kMarkerPixels);

    // Outer corners of the printed sheet (marker + white border) in the marker frame.
    const std::vector<cv::Point3f> outer = {
        {-half_outer, half_outer, 0.f}, {half_outer, half_outer, 0.f},
        {half_outer, -half_outer, 0.f}, {-half_outer, -half_outer, 0.f}};
    const auto t = static_cast<float>(texture_size);
    const std::vector<cv::Point2f> texture_corners = {{0.f, 0.f}, {t, 0.f}, {t, t}, {0.f, t}};

    for (const auto& marker : markers) {
        cv::Mat bits;
        cv::aruco::generateImageMarker(dictionary_, marker.id, kMarkerPixels, bits, 1);
        cv::Mat sheet;
        cv::copyMakeBorder(bits, sheet, kBorderPixels, kBorderPixels, kBorderPixels,
                           kBorderPixels, cv::BORDER_CONSTANT, cv::Scalar(255));
        cv::cvtColor(sheet, sheet, cv::COLOR_GRAY2BGR);

        std::vector<cv::Point2f> projected;
        cv::projectPoints(outer, marker.pose.rvec, marker.pose.tvec, camera_.K,
                          camera_.distortion, projected);

        const cv::Mat H = cv::getPerspectiveTransform(texture_corners, projected);
        cv::Mat warped;
        cv::Mat mask;
        cv::warpPerspective(sheet, warped, H, frame.size(), cv::INTER_LINEAR);
        cv::warpPerspective(cv::Mat(sheet.size(), CV_8U, cv::Scalar(255)), mask, H, frame.size(),
                            cv::INTER_NEAREST);
        warped.copyTo(frame, mask);
    }

    if (noise_stddev > 0.0) {
        cv::RNG rng(seed);
        cv::Mat noise(frame.size(), CV_16SC3);
        rng.fill(noise, cv::RNG::NORMAL, 0.0, noise_stddev);
        cv::Mat noisy;
        frame.convertTo(noisy, CV_16SC3);
        noisy += noise;
        noisy.convertTo(frame, CV_8UC3);
    }
    return frame;
}

}  // namespace poseforge
