#include "poseforge/pipeline.hpp"
#include "poseforge/pose.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using namespace poseforge;
using namespace std::chrono_literals;

namespace {

constexpr double kMarkerLength = 0.08;
constexpr std::size_t kFrames = 40;

PipelineConfig make_config(std::size_t workers) {
    PipelineConfig c;
    c.camera = CameraIntrinsics::from_fov({640, 480}, 70.0);
    c.marker_length = kMarkerLength;
    c.detect_workers = workers;
    return c;
}

SyntheticSource make_source(const PipelineConfig& c, std::size_t frames = kFrames) {
    return SyntheticSource(SyntheticScene(c.camera, kMarkerLength, kDefaultDictionary), frames,
                           {2, 9}, 1.0);
}

struct Recorded {
    std::uint64_t id;
    std::vector<MarkerPose> poses;
};

class RecordingSink : public FrameSink {
public:
    explicit RecordingSink(std::chrono::milliseconds delay = 0ms, std::size_t stop_after = 0)
        : delay_(delay), stop_after_(stop_after) {}

    bool consume(const FrameResult& r) override {
        if (delay_ > 0ms) std::this_thread::sleep_for(delay_);
        frames.push_back({r.id, r.poses});
        return stop_after_ == 0 || frames.size() < stop_after_;
    }

    std::vector<Recorded> frames;

private:
    std::chrono::milliseconds delay_;
    std::size_t stop_after_;
};

class ThrowingSource : public FrameSource {
public:
    std::optional<cv::Mat> next() override {
        if (++calls_ > 3) throw std::runtime_error("camera unplugged");
        return cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(128));
    }

private:
    int calls_ = 0;
};

void expect_in_order(const std::vector<Recorded>& frames) {
    for (std::size_t i = 1; i < frames.size(); ++i) {
        ASSERT_LT(frames[i - 1].id, frames[i].id) << "frame order broken at " << i;
    }
}

}  // namespace

TEST(Pipeline, SequentialTracksMarkersAgainstGroundTruth) {
    const auto config = make_config(1);
    auto source = make_source(config);
    const auto truth = make_source(config);
    RecordingSink sink;
    const auto report = run_sequential(source, sink, config);

    ASSERT_EQ(report.frames, kFrames);
    EXPECT_EQ(report.markers, 2 * kFrames);
    for (const auto& f : sink.frames) {
        for (const auto& gt : truth.markers_at(f.id)) {
            bool found = false;
            for (const auto& p : f.poses) {
                if (p.id != gt.id) continue;
                found = true;
                EXPECT_LT(translation_error(p.pose, gt.pose), 0.02 * cv::norm(gt.pose.tvec));
                EXPECT_LT(rotation_error_deg(p.pose, gt.pose), 3.0);
            }
            EXPECT_TRUE(found) << "marker " << gt.id << " missing in frame " << f.id;
        }
    }
}

TEST(Pipeline, PipelinedMatchesSequentialAndKeepsFrameOrder) {
    const auto config = make_config(4);
    auto seq_source = make_source(config);
    auto par_source = make_source(config);
    RecordingSink seq_sink;
    RecordingSink par_sink;
    run_sequential(seq_source, seq_sink, config);
    const auto report = run_pipelined(par_source, par_sink, config);

    EXPECT_EQ(report.frames, kFrames);
    EXPECT_EQ(report.dropped, 0u);
    ASSERT_EQ(par_sink.frames.size(), seq_sink.frames.size());
    expect_in_order(par_sink.frames);
    for (std::size_t i = 0; i < kFrames; ++i) {
        EXPECT_EQ(par_sink.frames[i].id, i);
        ASSERT_EQ(par_sink.frames[i].poses.size(), seq_sink.frames[i].poses.size());
        for (std::size_t m = 0; m < seq_sink.frames[i].poses.size(); ++m) {
            // Same input, same algorithm: results must be identical, not just close.
            EXPECT_EQ(par_sink.frames[i].poses[m].id, seq_sink.frames[i].poses[m].id);
            EXPECT_EQ(par_sink.frames[i].poses[m].pose.tvec, seq_sink.frames[i].poses[m].pose.tvec);
        }
    }
}

TEST(Pipeline, ReportHasOneLatencySamplePerFrame) {
    const auto config = make_config(2);
    auto source = make_source(config, 10);
    RecordingSink sink;
    const auto r = run_pipelined(source, sink, config);
    EXPECT_EQ(r.end_to_end.count, 10u);
    EXPECT_EQ(r.detect.count, 10u);
    EXPECT_EQ(r.capture.count, 10u);
    EXPECT_GT(r.fps(), 0.0);
    EXPECT_GE(r.end_to_end.p50_ms, r.detect.p50_ms);
}

TEST(Pipeline, DropOldestShedsLoadWhenSinkIsSlow) {
    auto config = make_config(1);
    config.drop_frames = true;
    config.queue_capacity = 1;
    std::vector<cv::Mat> frames = {cv::Mat(240, 320, CV_8UC3, cv::Scalar::all(90))};
    MemorySource source(std::move(frames), 200);
    RecordingSink sink(5ms);
    const auto report = run_pipelined(source, sink, config);

    EXPECT_GT(report.dropped, 0u);
    EXPECT_EQ(report.frames + report.dropped, 200u);
    EXPECT_EQ(sink.frames.size(), report.frames);
    expect_in_order(sink.frames);
}

TEST(Pipeline, SinkCanStopThePipeline) {
    const auto config = make_config(3);
    auto source = make_source(config, 200);
    RecordingSink sink(0ms, 5);
    const auto report = run_pipelined(source, sink, config);
    EXPECT_EQ(sink.frames.size(), 5u);
    EXPECT_EQ(report.frames, 5u);
}

TEST(Pipeline, SourceExceptionPropagatesToCaller) {
    const auto config = make_config(2);
    ThrowingSource source;
    RecordingSink sink;
    EXPECT_THROW(run_pipelined(source, sink, config), std::runtime_error);
    ThrowingSource source2;
    EXPECT_THROW(run_sequential(source2, sink, config), std::runtime_error);
}

TEST(Pipeline, RejectsZeroWorkers) {
    const auto config = make_config(0);
    auto source = make_source(config, 1);
    NullSink sink;
    EXPECT_THROW(run_pipelined(source, sink, config), std::invalid_argument);
}

TEST(FrameIo, MemorySourceReplaysCopies) {
    MemorySource source({cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(1))}, 3);
    auto a = source.next();
    auto b = source.next();
    ASSERT_TRUE(a && b);
    EXPECT_NE(a->data, b->data);  // independent buffers
    EXPECT_TRUE(source.next().has_value());
    EXPECT_FALSE(source.next().has_value());
}

TEST(FrameIo, PacedSourceHoldsTheFrameRate) {
    MemorySource inner({cv::Mat(4, 4, CV_8UC3)}, 6);
    PacedSource paced(inner, 100.0);  // 10 ms per frame
    const auto start = Clock::now();
    while (paced.next()) {
    }
    EXPECT_GE(Clock::now() - start, 50ms);  // 6 frames -> at least 5 periods
}
