#include "poseforge/detector.hpp"
#include "poseforge/synthetic.hpp"

#include <algorithm>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

using namespace poseforge;

namespace {

SyntheticScene make_scene() {
    return SyntheticScene(CameraIntrinsics::from_fov({640, 480}, 70.0), 0.10, kDefaultDictionary);
}

// Centre of a detection in pixels.
cv::Point2f centre(const MarkerDetection& d) {
    cv::Point2f c{0.f, 0.f};
    for (const auto& p : d.corners) c += p;
    return c * 0.25f;
}

}  // namespace

TEST(MarkerDetector, FindsNothingInEmptyScene) {
    const auto scene = make_scene();
    const MarkerDetector detector;
    EXPECT_TRUE(detector.detect(scene.render({})).empty());
}

TEST(MarkerDetector, FindsMarkerFacingCamera) {
    const auto scene = make_scene();
    const MarkerDetector detector;
    const auto detections = detector.detect(scene.render({{7, make_marker_pose(0.5)}}));

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].id, 7);
    // Marker centred on the optical axis lands on the principal point.
    EXPECT_NEAR(centre(detections[0]).x, 320.f, 1.5f);
    EXPECT_NEAR(centre(detections[0]).y, 240.f, 1.5f);
    // ArUco order: first corner is top-left in the image for an upright marker.
    EXPECT_LT(detections[0].corners[0].x, detections[0].corners[1].x);
    EXPECT_LT(detections[0].corners[0].y, detections[0].corners[3].y);
}

TEST(MarkerDetector, FindsSeveralTiltedMarkersWithNoise) {
    const auto scene = make_scene();
    const MarkerDetector detector;
    const std::vector<SceneMarker> markers = {
        {3, make_marker_pose(0.7, -0.15, -0.05, 25.0, 0.0)},
        {11, make_marker_pose(0.8, 0.18, 0.08, 0.0, -30.0)},
    };
    auto detections = detector.detect(scene.render(markers, 4.0, 42));

    ASSERT_EQ(detections.size(), 2u);
    std::ranges::sort(detections, {}, &MarkerDetection::id);
    EXPECT_EQ(detections[0].id, 3);
    EXPECT_EQ(detections[1].id, 11);
}

TEST(MarkerDetector, AcceptsGrayscaleFrames) {
    const auto scene = make_scene();
    const MarkerDetector detector;
    cv::Mat gray;
    cv::cvtColor(scene.render({{1, make_marker_pose(0.5)}}), gray, cv::COLOR_BGR2GRAY);
    EXPECT_EQ(detector.detect(gray).size(), 1u);
}
