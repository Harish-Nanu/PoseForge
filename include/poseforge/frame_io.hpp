#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "poseforge/synthetic.hpp"

namespace poseforge {

struct FrameResult;

// Produces frames for the pipeline. next() returns nullopt at end of stream.
class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual std::optional<cv::Mat> next() = 0;
};

// Video file or camera through cv::VideoCapture.
class VideoSource : public FrameSource {
public:
    explicit VideoSource(const std::string& path);
    explicit VideoSource(int camera_index);
    std::optional<cv::Mat> next() override;
    cv::Size frame_size() const;
    double fps() const;

private:
    cv::VideoCapture capture_;
};

// Replays pre-loaded frames `total` times in a loop. Each frame is a fresh copy, like a camera
// buffer, so downstream stages can draw on it. Used for benchmarks: no decode or render cost.
class MemorySource : public FrameSource {
public:
    MemorySource(std::vector<cv::Mat> frames, std::size_t total);
    std::optional<cv::Mat> next() override;

private:
    std::vector<cv::Mat> frames_;
    std::size_t total_;
    std::size_t emitted_ = 0;
};

// Markers orbiting and tilting in front of the camera; markers_at(i) is the ground truth.
class SyntheticSource : public FrameSource {
public:
    SyntheticSource(SyntheticScene scene, std::size_t frames, std::vector<int> marker_ids,
                    double noise_stddev = 2.0);
    std::optional<cv::Mat> next() override;
    std::vector<SceneMarker> markers_at(std::size_t frame_index) const;

private:
    SyntheticScene scene_;
    std::size_t frames_;
    std::vector<int> ids_;
    double noise_;
    std::size_t emitted_ = 0;
};

// Wraps a source and releases frames at a fixed rate, like a real camera.
class PacedSource : public FrameSource {
public:
    PacedSource(FrameSource& inner, double fps);
    std::optional<cv::Mat> next() override;

private:
    FrameSource& inner_;
    std::chrono::steady_clock::duration period_;
    std::optional<std::chrono::steady_clock::time_point> next_due_;
};

// Receives finished frames in order. Returning false asks the pipeline to stop.
class FrameSink {
public:
    virtual ~FrameSink() = default;
    virtual bool consume(const FrameResult& result) = 0;
};

class NullSink : public FrameSink {
public:
    bool consume(const FrameResult&) override { return true; }
};

// Writes frames to a video file; opened on the first frame so the size is known.
class VideoWriterSink : public FrameSink {
public:
    VideoWriterSink(std::string path, double fps);
    bool consume(const FrameResult& result) override;

private:
    std::string path_;
    double fps_;
    cv::VideoWriter writer_;
};

// Shows frames in a window. Esc or q stops the pipeline.
class DisplaySink : public FrameSink {
public:
    explicit DisplaySink(std::string window_name);
    bool consume(const FrameResult& result) override;

private:
    std::string window_;
};

}  // namespace poseforge
