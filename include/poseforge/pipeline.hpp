#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "poseforge/camera.hpp"
#include "poseforge/detector.hpp"
#include "poseforge/frame_io.hpp"
#include "poseforge/stats.hpp"
#include "poseforge/types.hpp"

namespace poseforge {

using Clock = std::chrono::steady_clock;

struct FrameTiming {
    Clock::time_point captured{};      // frame handed over by the source
    Clock::time_point detect_start{};  // a detection worker picked it up
    Clock::time_point detect_end{};    // markers detected and poses estimated
    Clock::time_point render_start{};  // render stage picked it up (in frame order)
    Clock::time_point render_end{};    // overlay drawn
};

struct FrameResult {
    std::uint64_t id = 0;
    cv::Mat image;
    std::vector<MarkerPose> poses;
    FrameTiming timing;
};

struct PipelineConfig {
    CameraIntrinsics camera;
    double marker_length = 0.05;  // metres, side of the black square
    cv::aruco::PredefinedDictionaryType dictionary = kDefaultDictionary;
    std::size_t detect_workers = 1;  // parallel detection threads (pipelined mode)
    std::size_t queue_capacity = 4;  // frames buffered between stages
    bool drop_frames = false;        // DropOldest at the capture queue (live sources)
    bool draw = true;
};

struct PipelineReport {
    std::size_t frames = 0;   // frames that reached the sink
    std::size_t dropped = 0;  // frames evicted at the capture queue
    std::size_t markers = 0;  // total marker poses estimated
    double wall_seconds = 0.0;
    LatencySummary capture;     // time inside source.next()
    LatencySummary queue_wait;  // captured -> detection start
    LatencySummary detect;      // detection + pose estimation
    LatencySummary render;      // overlay drawing
    LatencySummary end_to_end;  // captured -> overlay drawn

    double fps() const { return wall_seconds > 0.0 ? static_cast<double>(frames) / wall_seconds : 0.0; }
};

// Detects markers and estimates their poses for one frame (the work of one detection worker).
std::vector<MarkerPose> process_frame(const cv::Mat& frame, const MarkerDetector& detector,
                                      const PipelineConfig& config);

// Everything on the calling thread, one frame at a time. Baseline for the benchmark.
PipelineReport run_sequential(FrameSource& source, FrameSink& sink, const PipelineConfig& config);

// Capture thread -> bounded queue -> N detection workers -> bounded queue -> render thread.
// The render thread restores frame order, so the sink sees frames exactly as captured.
// Exceptions from any stage stop the pipeline and are rethrown here.
PipelineReport run_pipelined(FrameSource& source, FrameSink& sink, const PipelineConfig& config);

}  // namespace poseforge
