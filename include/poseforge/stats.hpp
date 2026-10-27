#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

namespace poseforge {

struct LatencySummary {
    std::size_t count = 0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

// Collects durations and reports percentiles (nearest-rank). Not thread-safe: the pipeline records
// every sample from a single thread.
class LatencyRecorder {
public:
    void reserve(std::size_t n) { samples_ms_.reserve(n); }
    void add(std::chrono::nanoseconds duration);
    std::size_t count() const { return samples_ms_.size(); }
    LatencySummary summary() const;

private:
    std::vector<double> samples_ms_;
};

// q in [0, 1]. `sorted` must be sorted ascending and non-empty.
double percentile(const std::vector<double>& sorted, double q);

}  // namespace poseforge
