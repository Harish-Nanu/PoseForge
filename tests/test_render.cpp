#include "poseforge/render.hpp"
#include "poseforge/synthetic.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

using namespace poseforge;

namespace {

const auto kCamera = CameraIntrinsics::from_fov({640, 480}, 70.0);

int changed_pixels(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
    return cv::countNonZero(diff);
}

}  // namespace

TEST(Render, CubeTopIsDrawnWhereItProjects) {
    const MarkerPose marker{1, make_marker_pose(0.5), 0.0};
    cv::Mat frame(kCamera.image_size, CV_8UC3, cv::Scalar::all(0));
    draw_cube(frame, marker, 0.1, kCamera);

    // A top corner of the cube (z = length) should be painted with the top-edge colour.
    std::vector<cv::Point2f> pts;
    cv::projectPoints(std::vector<cv::Point3f>{{0.05f, 0.05f, 0.1f}}, marker.pose.rvec,
                      marker.pose.tvec, kCamera.K, kCamera.distortion, pts);
    const auto px = frame.at<cv::Vec3b>(cv::Point(cvRound(pts[0].x), cvRound(pts[0].y)));
    EXPECT_GT(px[1], 100);  // green channel of the top edges
}

TEST(Render, SkipsMarkersBehindTheCamera) {
    MarkerPose marker{1, make_marker_pose(0.5), 0.0};
    marker.pose.tvec[2] = -0.5;
    cv::Mat frame(kCamera.image_size, CV_8UC3, cv::Scalar::all(0));
    const cv::Mat before = frame.clone();
    draw_overlay(frame, {marker}, 0.1, kCamera);
    EXPECT_EQ(changed_pixels(before, frame), 0);
}

TEST(Render, OverlayDrawsEveryMarker) {
    cv::Mat frame(kCamera.image_size, CV_8UC3, cv::Scalar::all(0));
    const cv::Mat before = frame.clone();
    draw_overlay(frame,
                 {{1, make_marker_pose(0.8, -0.2, 0.0), 0.0}, {2, make_marker_pose(0.8, 0.2, 0.0), 0.0}},
                 0.1, kCamera);
    cv::Mat left = frame(cv::Rect(0, 0, 320, 480));
    cv::Mat right = frame(cv::Rect(320, 0, 320, 480));
    EXPECT_GT(changed_pixels(before(cv::Rect(0, 0, 320, 480)), left), 500);
    EXPECT_GT(changed_pixels(before(cv::Rect(320, 0, 320, 480)), right), 500);
}
