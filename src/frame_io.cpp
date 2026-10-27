#include "poseforge/frame_io.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <thread>

#include <opencv2/highgui.hpp>

#include "poseforge/pipeline.hpp"

namespace poseforge {

VideoSource::VideoSource(const std::string& path) : capture_(path) {
    if (!capture_.isOpened()) {
        throw std::runtime_error("cannot open video: " + path);
    }
}

VideoSource::VideoSource(int camera_index) : capture_(camera_index) {
    if (!capture_.isOpened()) {
        throw std::runtime_error("cannot open camera " + std::to_string(camera_index));
    }
}

std::optional<cv::Mat> VideoSource::next() {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        return std::nullopt;
    }
    return frame;
}

cv::Size VideoSource::frame_size() const {
    return {static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT))};
}

double VideoSource::fps() const { return capture_.get(cv::CAP_PROP_FPS); }

MemorySource::MemorySource(std::vector<cv::Mat> frames, std::size_t total)
    : frames_(std::move(frames)), total_(total) {
    if (frames_.empty()) {
        throw std::invalid_argument("MemorySource needs at least one frame");
    }
}

std::optional<cv::Mat> MemorySource::next() {
    if (emitted_ >= total_) {
        return std::nullopt;
    }
    return frames_[emitted_++ % frames_.size()].clone();
}

SyntheticSource::SyntheticSource(SyntheticScene scene, std::size_t frames,
                                 std::vector<int> marker_ids, double noise_stddev)
    : scene_(std::move(scene)), frames_(frames), ids_(std::move(marker_ids)), noise_(noise_stddev) {}

std::vector<SceneMarker> SyntheticSource::markers_at(std::size_t frame_index) const {
    constexpr double kTwoPi = 2.0 * std::numbers::pi;
    constexpr double kFramesPerOrbit = 120.0;
    std::vector<SceneMarker> markers;
    const auto n = static_cast<double>(ids_.size());
    for (std::size_t k = 0; k < ids_.size(); ++k) {
        const double phase = kTwoPi * static_cast<double>(k) / n;
        const double a = kTwoPi * static_cast<double>(frame_index) / kFramesPerOrbit + phase;
        const double distance = 0.7 + 0.1 * std::sin(a);
        markers.push_back({ids_[k], make_marker_pose(distance, 0.18 * std::cos(a),
                                                     0.10 * std::sin(a), 20.0 * std::sin(a),
                                                     20.0 * std::cos(a))});
    }
    return markers;
}

std::optional<cv::Mat> SyntheticSource::next() {
    if (emitted_ >= frames_) {
        return std::nullopt;
    }
    const std::size_t i = emitted_++;
    return scene_.render(markers_at(i), noise_, i);
}

PacedSource::PacedSource(FrameSource& inner, double fps)
    : inner_(inner),
      period_(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / fps))) {
    if (fps <= 0.0) {
        throw std::invalid_argument("fps must be positive");
    }
}

std::optional<cv::Mat> PacedSource::next() {
    const auto now = std::chrono::steady_clock::now();
    if (!next_due_) {
        next_due_ = now;
    }
    std::this_thread::sleep_until(*next_due_);
    *next_due_ += period_;
    return inner_.next();
}

VideoWriterSink::VideoWriterSink(std::string path, double fps) : path_(std::move(path)), fps_(fps) {}

bool VideoWriterSink::consume(const FrameResult& result) {
    if (!writer_.isOpened()) {
        writer_.open(path_, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_, result.image.size());
        if (!writer_.isOpened()) {
            throw std::runtime_error("cannot write video: " + path_);
        }
    }
    writer_.write(result.image);
    return true;
}

DisplaySink::DisplaySink(std::string window_name) : window_(std::move(window_name)) {
    cv::namedWindow(window_, cv::WINDOW_AUTOSIZE);
}

bool DisplaySink::consume(const FrameResult& result) {
    cv::imshow(window_, result.image);
    const int key = cv::waitKey(1);
    return key != 27 && key != 'q';
}

}  // namespace poseforge
