#include "poseforge/detector.hpp"

#include <opencv2/imgproc.hpp>

namespace poseforge {

namespace {

cv::aruco::DetectorParameters make_parameters() {
    cv::aruco::DetectorParameters params;
    // Sub-pixel corner refinement noticeably improves pose accuracy for a small cost.
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    return params;
}

}  // namespace

MarkerDetector::MarkerDetector(cv::aruco::PredefinedDictionaryType dictionary)
    : dictionary_(cv::aruco::getPredefinedDictionary(dictionary)),
      detector_(dictionary_, make_parameters()) {}

std::vector<MarkerDetection> MarkerDetector::detect(const cv::Mat& frame) const {
    cv::Mat gray;
    if (frame.channels() == 1) {
        gray = frame;
    } else {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    }

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector_.detectMarkers(gray, corners, ids);

    std::vector<MarkerDetection> detections;
    detections.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        MarkerDetection d;
        d.id = ids[i];
        for (std::size_t c = 0; c < 4; ++c) {
            d.corners[c] = corners[i][c];
        }
        detections.push_back(d);
    }
    return detections;
}

}  // namespace poseforge
