// poseforge: command-line front end for the AR marker tracking pipeline.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "args.hpp"
#include "poseforge/pipeline.hpp"
#include "poseforge/synthetic.hpp"

using namespace poseforge;

namespace {

constexpr const char* kUsage = R"(poseforge - real-time AR marker tracking

Usage:
  poseforge run [--synthetic N | --video FILE | --camera INDEX] [options]
      --calib FILE          camera calibration (OpenCV YAML); default: ideal 60 deg camera
      --marker-length M     side of the black marker square in metres (default 0.05)
      --mode MODE           pipelined (default) or sequential
      --workers N           detection threads in pipelined mode (default 2)
      --queue N             frames buffered between stages (default 4)
      --drop-frames         drop the oldest frame when behind (default for --camera)
      --out FILE            write the annotated video (mp4)
      --show                show the annotated video in a window (Esc/q quits)

  poseforge bench [options]
      --frames N            frames per run (default 300)
      --width W --height H  frame size (default 1280x720)
      --markers N           markers in view (default 3)
      --workers LIST        detection worker counts to try (default 1,2,4)
      --fps F               pace the source like a camera (default 0 = as fast as possible)
      --drop-frames         drop the oldest frame when behind instead of blocking
      --queue N             frames buffered between stages (default 4)
      --opencv-threads N    OpenCV internal threads (default 1; -1 = OpenCV default)
      --json FILE           also write results as JSON

  poseforge generate-marker --id ID [--size PX] [--out FILE]
)";

void print_summary(const std::string& label, const LatencySummary& s) {
    std::printf("  %-11s p50 %7.2f ms   p95 %7.2f ms   p99 %7.2f ms   max %7.2f ms\n",
                label.c_str(), s.p50_ms, s.p95_ms, s.p99_ms, s.max_ms);
}

void print_report(const PipelineReport& r) {
    std::printf("frames %zu   dropped %zu   markers %zu   %.1f FPS (%.2f s)\n", r.frames,
                r.dropped, r.markers, r.fps(), r.wall_seconds);
    print_summary("capture", r.capture);
    print_summary("queue wait", r.queue_wait);
    print_summary("detect+pose", r.detect);
    print_summary("render", r.render);
    print_summary("end-to-end", r.end_to_end);
}

std::vector<int> marker_ids(int count) {
    std::vector<int> ids;
    for (int i = 0; i < count; ++i) ids.push_back(i * 7 % 50);
    return ids;
}

int cmd_generate_marker(const Args& args) {
    args.allow_only({"id", "size", "out"});
    const int id = args.integer("id", 0);
    const int size = args.integer("size", 600);
    const std::string out = args.str("out", "marker_" + std::to_string(id) + ".png");

    const auto dict = cv::aruco::getPredefinedDictionary(kDefaultDictionary);
    cv::Mat marker;
    cv::aruco::generateImageMarker(dict, id, size, marker, 1);
    cv::copyMakeBorder(marker, marker, size / 6, size / 6, size / 6, size / 6,
                       cv::BORDER_CONSTANT, cv::Scalar(255));
    if (!cv::imwrite(out, marker)) {
        throw std::runtime_error("cannot write " + out);
    }
    std::printf("wrote %s (DICT_4X4_50 id %d). Measure the black square after printing and pass it "
                "as --marker-length.\n",
                out.c_str(), id);
    return 0;
}

int cmd_run(const Args& args) {
    args.allow_only({"synthetic", "video", "camera", "calib", "marker-length", "mode", "workers",
                     "queue", "out"});

    PipelineConfig config;
    config.marker_length = args.number("marker-length", 0.05);
    config.detect_workers = static_cast<std::size_t>(args.integer("workers", 2));
    config.queue_capacity = static_cast<std::size_t>(args.integer("queue", 4));
    config.drop_frames = args.flag("drop-frames") || args.has("camera");

    std::unique_ptr<FrameSource> source;
    cv::Size size{1280, 720};
    double fps = 30.0;
    if (args.has("video") || args.has("camera")) {
        auto video = args.has("video") ? std::make_unique<VideoSource>(args.str("video", ""))
                                       : std::make_unique<VideoSource>(args.integer("camera", 0));
        size = video->frame_size();
        if (video->fps() > 0.0) fps = video->fps();
        source = std::move(video);
    }

    config.camera = args.has("calib") ? CameraIntrinsics::load(args.str("calib", ""))
                                      : CameraIntrinsics::from_fov(size, 60.0);
    if (!args.has("calib")) {
        std::fprintf(stderr, "note: no --calib given, assuming an ideal 60 deg camera; poses are "
                             "approximate\n");
    }
    if (!source) {
        const auto frames = static_cast<std::size_t>(args.integer("synthetic", 300));
        source = std::make_unique<SyntheticSource>(
            SyntheticScene(config.camera, config.marker_length, config.dictionary), frames,
            marker_ids(3));
    }

    std::vector<std::unique_ptr<FrameSink>> sinks;
    if (args.has("out")) sinks.push_back(std::make_unique<VideoWriterSink>(args.str("out", ""), fps));
    if (args.flag("show")) sinks.push_back(std::make_unique<DisplaySink>("PoseForge"));

    struct FanOut : FrameSink {
        std::vector<std::unique_ptr<FrameSink>>& sinks;
        explicit FanOut(std::vector<std::unique_ptr<FrameSink>>& s) : sinks(s) {}
        bool consume(const FrameResult& r) override {
            bool keep_going = true;
            for (auto& s : sinks) keep_going = s->consume(r) && keep_going;
            return keep_going;
        }
    } sink(sinks);

    const auto mode = args.str("mode", "pipelined");
    PipelineReport report;
    if (mode == "sequential") {
        report = run_sequential(*source, sink, config);
    } else if (mode == "pipelined") {
        report = run_pipelined(*source, sink, config);
    } else {
        throw std::invalid_argument("unknown --mode " + mode);
    }
    print_report(report);
    return 0;
}

struct BenchRow {
    std::string mode;
    std::size_t workers;
    PipelineReport report;
};

void write_json(const std::string& path, const Args& args, const std::vector<BenchRow>& rows) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    auto summary = [&](const LatencySummary& s) {
        char buf[160];
        std::snprintf(buf, sizeof buf, R"({"p50_ms": %.3f, "p95_ms": %.3f, "p99_ms": %.3f, "max_ms": %.3f})",
                      s.p50_ms, s.p95_ms, s.p99_ms, s.max_ms);
        return std::string(buf);
    };
    out << "{\n  \"frames\": " << args.integer("frames", 300) << ",\n  \"width\": "
        << args.integer("width", 1280) << ",\n  \"height\": " << args.integer("height", 720)
        << ",\n  \"hardware_threads\": " << std::thread::hardware_concurrency()
        << ",\n  \"runs\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i].report;
        out << "    {\"mode\": \"" << rows[i].mode << "\", \"workers\": " << rows[i].workers
            << ", \"fps\": " << r.fps() << ", \"dropped\": " << r.dropped
            << ", \"detect\": " << summary(r.detect) << ", \"end_to_end\": " << summary(r.end_to_end)
            << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
}

int cmd_bench(const Args& args) {
    args.allow_only({"frames", "width", "height", "markers", "workers", "fps", "opencv-threads",
                     "queue", "json"});
    const auto frames = static_cast<std::size_t>(args.integer("frames", 300));
    const cv::Size size{args.integer("width", 1280), args.integer("height", 720)};
    const double fps = args.number("fps", 0.0);
    // One OpenCV thread by default: PoseForge does its own frame-level parallelism, and letting
    // OpenCV also fan out inside each call oversubscribes the cores and muddies the comparison.
    cv::setNumThreads(args.integer("opencv-threads", 1));

    PipelineConfig config;
    config.camera = CameraIntrinsics::from_fov(size, 60.0);
    config.marker_length = 0.05;
    config.draw = true;
    config.drop_frames = args.flag("drop-frames");
    config.queue_capacity = static_cast<std::size_t>(args.integer("queue", 4));

    // Pre-render distinct frames once so the benchmark measures tracking, not scene synthesis.
    SyntheticSource synth(SyntheticScene(config.camera, config.marker_length, config.dictionary),
                          60, marker_ids(args.integer("markers", 3)));
    std::vector<cv::Mat> clip;
    while (auto f = synth.next()) clip.push_back(std::move(*f));

    auto run_once = [&](bool pipelined, std::size_t workers) {
        config.detect_workers = workers;
        MemorySource memory(clip, frames);
        std::unique_ptr<PacedSource> paced;
        FrameSource* source = &memory;
        if (fps > 0.0) {
            paced = std::make_unique<PacedSource>(memory, fps);
            source = paced.get();
        }
        NullSink sink;
        return pipelined ? run_pipelined(*source, sink, config) : run_sequential(*source, sink, config);
    };

    std::vector<BenchRow> rows;
    rows.push_back({"sequential", 1, run_once(false, 1)});
    for (const auto w : args.sizes("workers", {1, 2, 4})) {
        rows.push_back({"pipelined", w, run_once(true, w)});
    }

    const std::string pacing = fps > 0.0 ? std::to_string(static_cast<int>(fps)) + " FPS" : "unpaced";
    std::printf("%zu frames %dx%d, %d markers, %u hardware threads, source %s, %s\n\n", frames,
                size.width, size.height, args.integer("markers", 3),
                std::thread::hardware_concurrency(), pacing.c_str(),
                config.drop_frames ? "drop oldest" : "blocking queues");
    std::printf("%-11s %7s %8s %8s %10s %10s %10s %10s %8s\n", "mode", "workers", "FPS", "speedup",
                "detect p50", "e2e p50", "e2e p95", "e2e p99", "dropped");
    const double base_fps = rows.front().report.fps();
    for (const auto& row : rows) {
        const auto& r = row.report;
        std::printf("%-11s %7zu %8.1f %7.2fx %8.2fms %8.2fms %8.2fms %8.2fms %8zu\n",
                    row.mode.c_str(), row.workers, r.fps(), r.fps() / base_fps, r.detect.p50_ms,
                    r.end_to_end.p50_ms, r.end_to_end.p95_ms, r.end_to_end.p99_ms, r.dropped);
    }
    if (args.has("json")) write_json(args.str("json", ""), args, rows);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        std::fputs(kUsage, argc < 2 ? stderr : stdout);
        return argc < 2 ? 2 : 0;
    }
    const std::string command = argv[1];
    try {
        if (command == "run") return cmd_run(Args(argc, argv, 2, {"drop-frames", "show"}));
        if (command == "bench") return cmd_bench(Args(argc, argv, 2, {"drop-frames"}));
        if (command == "generate-marker") return cmd_generate_marker(Args(argc, argv, 2, {}));
        std::fprintf(stderr, "unknown command '%s'\n\n%s", command.c_str(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
