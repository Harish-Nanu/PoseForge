#include "poseforge/camera.hpp"

#include <gtest/gtest.h>

using poseforge::CameraIntrinsics;

TEST(CameraIntrinsics, FromFovPutsPrincipalPointAtImageCentre) {
    const auto cam = CameraIntrinsics::from_fov({640, 480}, 90.0);
    EXPECT_DOUBLE_EQ(cam.cx(), 320.0);
    EXPECT_DOUBLE_EQ(cam.cy(), 240.0);
    // tan(45 deg) == 1, so f == half the image width.
    EXPECT_NEAR(cam.fx(), 320.0, 1e-9);
    EXPECT_DOUBLE_EQ(cam.fx(), cam.fy());
}

TEST(CameraIntrinsics, FromFovRejectsBadInput) {
    EXPECT_THROW(CameraIntrinsics::from_fov({0, 480}, 60.0), std::invalid_argument);
    EXPECT_THROW(CameraIntrinsics::from_fov({640, 480}, 180.0), std::invalid_argument);
}
