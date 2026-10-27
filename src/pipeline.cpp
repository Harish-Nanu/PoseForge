#include "poseforge/pipeline.hpp"

#include <atomic>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "poseforge/bounded_queue.hpp"
#include "poseforge/pose.hpp"
#include "poseforge/render.hpp"

namespace poseforge {

namespace {

// Records per-frame timings into the report's recorders. Only ever used from one thread.
class TimingCollector {
public:
    void add_capture(Clock::duration d) { capture_.add(d); }

    void add_frame(const FrameResult& r) {
        const auto& t = r.timing;
        queue_wait_.add(t.detect_start - t.captured);
        detect_.add(t.detect_end - t.detect_start);
        render_.add(t.render_end - t.render_start);
        end_to_end_.add(t.render_end - t.captured);
    }

    void fill(PipelineReport& report) const {
        report.capture = capture_.summary();
        report.queue_wait = queue_wait_.summary();
        report.detect = detect_.summary();
        report.render = render_.summary();
        report.end_to_end = end_to_end_.summary();
    }

private:
    LatencyRecorder capture_, queue_wait_, detect_, render_, end_to_end_;
};

void render_frame(FrameResult& result, const PipelineConfig& config) {
    result.timing.render_start = Clock::now();
    if (config.draw) {
        draw_overlay(result.image, result.poses, config.marker_length, config.camera);
    }
    result.timing.render_end = Clock::now();
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct CapturedFrame {
    std::uint64_t id = 0;
    cv::Mat image;
    Clock::time_point captured;
};

// Output of a detection worker. A dropped frame travels as a tombstone so the render stage
// knows not to wait for it when restoring order.
struct DetectedFrame {
    FrameResult result;
    bool dropped = false;
};

// Remembers the first exception thrown by any stage.
class ErrorSlot {
public:
    void capture(std::exception_ptr e) {
        std::lock_guard lock(mutex_);
        if (!error_) error_ = std::move(e);
    }
    void rethrow_if_set() {
        if (error_) std::rethrow_exception(error_);
    }

private:
    std::mutex mutex_;
    std::exception_ptr error_;
};

}  // namespace

std::vector<MarkerPose> process_frame(const cv::Mat& frame, const MarkerDetector& detector,
                                      const PipelineConfig& config) {
    return estimate_poses(detector.detect(frame), config.marker_length, config.camera);
}

PipelineReport run_sequential(FrameSource& source, FrameSink& sink, const PipelineConfig& config) {
    const MarkerDetector detector(config.dictionary);
    TimingCollector timings;
    PipelineReport report;
    const auto start = Clock::now();

    for (std::uint64_t id = 0;; ++id) {
        const auto before = Clock::now();
        auto image = source.next();
        if (!image) break;

        FrameResult result;
        result.id = id;
        result.image = std::move(*image);
        result.timing.captured = Clock::now();
        timings.add_capture(result.timing.captured - before);

        result.timing.detect_start = result.timing.captured;
        result.poses = process_frame(result.image, detector, config);
        result.timing.detect_end = Clock::now();
        render_frame(result, config);

        timings.add_frame(result);
        ++report.frames;
        report.markers += result.poses.size();
        if (!sink.consume(result)) break;
    }

    report.wall_seconds = seconds_since(start);
    timings.fill(report);
    return report;
}

PipelineReport run_pipelined(FrameSource& source, FrameSink& sink, const PipelineConfig& config) {
    if (config.detect_workers == 0) {
        throw std::invalid_argument("detect_workers must be at least 1");
    }

    BoundedQueue<CapturedFrame> captured(config.queue_capacity, config.drop_frames
                                                                    ? OverflowPolicy::DropOldest
                                                                    : OverflowPolicy::Block);
    // Room for every in-flight frame so workers rarely wait on the render stage.
    BoundedQueue<DetectedFrame> detected(config.queue_capacity + config.detect_workers);

    std::atomic<bool> stop{false};
    ErrorSlot error;
    TimingCollector timings;  // written by the render thread only
    LatencyRecorder capture;  // written by the capture thread only
    PipelineReport report;

    auto fail = [&](std::exception_ptr e) {
        error.capture(std::move(e));
        stop = true;
        captured.close();
        detected.close();
    };

    const auto start = Clock::now();

    std::thread render_thread([&] {
        try {
            std::map<std::uint64_t, DetectedFrame> pending;  // reorder buffer
            std::uint64_t next_id = 0;
            bool sink_open = true;
            while (auto item = detected.pop()) {
                const auto id = item->result.id;
                pending.emplace(id, std::move(*item));
                for (auto it = pending.find(next_id); it != pending.end();
                     it = pending.find(next_id)) {
                    DetectedFrame frame = std::move(it->second);
                    pending.erase(it);
                    ++next_id;
                    if (frame.dropped) {
                        ++report.dropped;
                        continue;
                    }
                    if (!sink_open) continue;  // sink asked to stop; drain the rest
                    render_frame(frame.result, config);
                    timings.add_frame(frame.result);
                    ++report.frames;
                    report.markers += frame.result.poses.size();
                    if (!sink.consume(frame.result)) {
                        sink_open = false;
                        stop = true;
                    }
                }
            }
        } catch (...) {
            fail(std::current_exception());
        }
    });

    std::vector<std::thread> workers;
    workers.reserve(config.detect_workers);
    for (std::size_t w = 0; w < config.detect_workers; ++w) {
        workers.emplace_back([&] {
            try {
                const MarkerDetector detector(config.dictionary);  // one per thread
                while (auto frame = captured.pop()) {
                    DetectedFrame out;
                    out.result.id = frame->id;
                    out.result.image = std::move(frame->image);
                    out.result.timing.captured = frame->captured;
                    out.result.timing.detect_start = Clock::now();
                    out.result.poses = process_frame(out.result.image, detector, config);
                    out.result.timing.detect_end = Clock::now();
                    if (!detected.push(std::move(out)).accepted) break;
                }
            } catch (...) {
                fail(std::current_exception());
            }
        });
    }

    std::thread capture_thread([&] {
        try {
            for (std::uint64_t id = 0; !stop; ++id) {
                const auto before = Clock::now();
                auto image = source.next();
                if (!image) break;
                const auto now = Clock::now();
                capture.add(now - before);

                auto pushed = captured.push({id, std::move(*image), now});
                if (!pushed.accepted) break;
                if (pushed.evicted) {
                    DetectedFrame tombstone;
                    tombstone.result.id = pushed.evicted->id;
                    tombstone.dropped = true;
                    detected.push(std::move(tombstone));
                }
            }
        } catch (...) {
            fail(std::current_exception());
        }
        captured.close();
    });

    capture_thread.join();
    for (auto& w : workers) w.join();
    detected.close();  // all producers are done
    render_thread.join();

    report.wall_seconds = seconds_since(start);
    error.rethrow_if_set();
    timings.fill(report);
    report.capture = capture.summary();
    return report;
}

}  // namespace poseforge
