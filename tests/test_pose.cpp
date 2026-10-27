#include "poseforge/detector.hpp"
#include "poseforge/pose.hpp"
#include "poseforge/synthetic.hpp"

#include <gtest/gtest.h>

using namespace poseforge;

namespace {

constexpr double kMarkerLength = 0.10;  // 10 cm

struct PoseCase {
    const char* name;
    Pose pose;
    double noise;
};

class PoseAccuracy : public ::testing::TestWithParam<PoseCase> {};

}  // namespace

TEST(Pose, ObjectPointsAreSquareInZPlane) {
    const auto pts = marker_object_points(0.2);
    ASSERT_EQ(pts.size(), 4u);
    EXPECT_FLOAT_EQ(pts[0].x, -0.1f);
    EXPECT_FLOAT_EQ(pts[0].y, 0.1f);
    EXPECT_FLOAT_EQ(pts[2].x, 0.1f);
    EXPECT_FLOAT_EQ(pts[2].y, -0.1f);
    for (const auto& p : pts) EXPECT_FLOAT_EQ(p.z, 0.f);
}

TEST(Pose, ErrorMetricsAreZeroForSamePose) {
    const Pose p = make_marker_pose(0.5, 0.1, 0.0, 20.0, 10.0);
    EXPECT_NEAR(rotation_error_deg(p, p), 0.0, 1e-6);
    EXPECT_DOUBLE_EQ(translation_error(p, p), 0.0);
}

TEST(Pose, RotationErrorMeasuresTilt) {
    EXPECT_NEAR(rotation_error_deg(make_marker_pose(0.5), make_marker_pose(0.5, 0, 0, 30.0)),
                30.0, 1e-6);
}

TEST_P(PoseAccuracy, RecoversGroundTruthPose) {
    const auto& param = GetParam();
    const SyntheticScene scene(CameraIntrinsics::from_fov({1280, 720}, 70.0), kMarkerLength,
                               kDefaultDictionary);
    const MarkerDetector detector;

    const auto frame = scene.render({{5, param.pose}}, param.noise, 7);
    const auto detections = detector.detect(frame);
    ASSERT_EQ(detections.size(), 1u);

    const auto estimate = estimate_pose(detections[0], kMarkerLength, scene.camera());
    ASSERT_TRUE(estimate.has_value());
    EXPECT_EQ(estimate->id, 5);

    const double distance = cv::norm(param.pose.tvec);
    EXPECT_LT(translation_error(estimate->pose, param.pose), 0.01 * distance);  // < 1% of range
    EXPECT_LT(rotation_error_deg(estimate->pose, param.pose), 2.0);
    EXPECT_LT(estimate->reprojection_error_px, 1.0);
}

INSTANTIATE_TEST_SUITE_P(
    SyntheticPoses, PoseAccuracy,
    ::testing::Values(PoseCase{"frontal", make_marker_pose(0.5), 0.0},
                      PoseCase{"far_off_axis", make_marker_pose(1.2, 0.3, -0.15), 0.0},
                      PoseCase{"tilted_x", make_marker_pose(0.6, 0.0, 0.0, 35.0, 0.0), 0.0},
                      PoseCase{"tilted_xy_noisy", make_marker_pose(0.7, -0.1, 0.05, 25.0, -25.0),
                               3.0}),
    [](const ::testing::TestParamInfo<PoseCase>& case_info) { return std::string(case_info.param.name); });
