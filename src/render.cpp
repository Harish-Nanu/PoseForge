#include "poseforge/render.hpp"

#include <array>
#include <string>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace poseforge {

namespace {

const cv::Scalar kBaseColor(0, 200, 255);   // BGR orange
const cv::Scalar kPillarColor(255, 160, 0);  // BGR blue
const cv::Scalar kTopColor(80, 255, 80);    // BGR green

// Edges of a cube whose vertices are 0-3 (base) and 4-7 (top).
constexpr std::array<std::pair<int, int>, 4> kBaseEdges{{{0, 1}, {1, 2}, {2, 3}, {3, 0}}};
constexpr std::array<std::pair<int, int>, 4> kPillarEdges{{{0, 4}, {1, 5}, {2, 6}, {3, 7}}};
constexpr std::array<std::pair<int, int>, 4> kTopEdges{{{4, 5}, {5, 6}, {6, 7}, {7, 4}}};

template <std::size_t N>
void draw_edges(cv::Mat& frame, const std::vector<cv::Point2f>& pts,
                const std::array<std::pair<int, int>, N>& edges, const cv::Scalar& color) {
    for (const auto& [a, b] : edges) {
        cv::line(frame, pts[static_cast<std::size_t>(a)], pts[static_cast<std::size_t>(b)], color,
                 2, cv::LINE_AA);
    }
}

}  // namespace

void draw_cube(cv::Mat& frame, const MarkerPose& marker, double length,
               const CameraIntrinsics& camera) {
    if (marker.pose.tvec[2] <= 0.0) {
        return;  // behind the camera; projection would be meaningless
    }
    const auto h = static_cast<float>(length / 2.0);
    const auto l = static_cast<float>(length);
    // The marker's z axis points out of the marker towards the viewer, so the cube rises along +z.
    const std::vector<cv::Point3f> vertices = {
        {-h, h, 0.f}, {h, h, 0.f}, {h, -h, 0.f}, {-h, -h, 0.f},
        {-h, h, l},   {h, h, l},   {h, -h, l},   {-h, -h, l}};

    std::vector<cv::Point2f> projected;
    cv::projectPoints(vertices, marker.pose.rvec, marker.pose.tvec, camera.K, camera.distortion,
                      projected);

    draw_edges(frame, projected, kBaseEdges, kBaseColor);
    draw_edges(frame, projected, kPillarEdges, kPillarColor);
    draw_edges(frame, projected, kTopEdges, kTopColor);
}

void draw_overlay(cv::Mat& frame, const std::vector<MarkerPose>& markers, double marker_length,
                  const CameraIntrinsics& camera) {
    for (const auto& m : markers) {
        if (m.pose.tvec[2] <= 0.0) continue;
        draw_cube(frame, m, marker_length, camera);
        cv::drawFrameAxes(frame, camera.K, camera.distortion, m.pose.rvec, m.pose.tvec,
                          static_cast<float>(marker_length * 0.5), 2);

        std::vector<cv::Point2f> origin;
        cv::projectPoints(std::vector<cv::Point3f>{{0.f, 0.f, 0.f}}, m.pose.rvec, m.pose.tvec,
                          camera.K, camera.distortion, origin);
        cv::putText(frame, "id " + std::to_string(m.id), origin[0] + cv::Point2f(8.f, -8.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}

}  // namespace poseforge
