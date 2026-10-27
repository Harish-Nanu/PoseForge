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

TEST(CameraIntrinsics, SaveAndLoadRoundTrip) {
    auto cam = CameraIntrinsics::from_fov({1280, 720}, 65.0);
    cam.distortion = (cv::Mat_<double>(1, 5) << 0.1, -0.05, 0.001, 0.002, 0.0);
    const std::string path = ::testing::TempDir() + "poseforge_calib.yml";
    cam.save(path);

    const auto loaded = CameraIntrinsics::load(path);
    EXPECT_EQ(loaded.image_size, cam.image_size);
    EXPECT_NEAR(loaded.fx(), cam.fx(), 1e-9);
    EXPECT_NEAR(loaded.cy(), cam.cy(), 1e-9);
    EXPECT_EQ(cv::norm(loaded.distortion, cam.distortion), 0.0);
}

TEST(CameraIntrinsics, LoadMissingFileThrows) {
    EXPECT_THROW(CameraIntrinsics::load("/nonexistent/calib.yml"), std::runtime_error);
}
