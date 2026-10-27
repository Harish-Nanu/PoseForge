#pragma once

#include <vector>

#include <opencv2/objdetect/aruco_detector.hpp>

#include "poseforge/types.hpp"

namespace poseforge {

inline constexpr auto kDefaultDictionary = cv::aruco::DICT_4X4_50;

// Wraps cv::aruco::ArucoDetector. Each worker thread owns its own instance.
class MarkerDetector {
public:
    explicit MarkerDetector(cv::aruco::PredefinedDictionaryType dictionary = kDefaultDictionary);

    // Accepts BGR or grayscale frames.
    std::vector<MarkerDetection> detect(const cv::Mat& frame) const;

    const cv::aruco::Dictionary& dictionary() const { return dictionary_; }

private:
    cv::aruco::Dictionary dictionary_;
    cv::aruco::ArucoDetector detector_;
};

}  // namespace poseforge
