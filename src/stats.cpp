#include "poseforge/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace poseforge {

void LatencyRecorder::add(std::chrono::nanoseconds duration) {
    samples_ms_.push_back(std::chrono::duration<double, std::milli>(duration).count());
}

double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        throw std::invalid_argument("percentile of an empty sample");
    }
    q = std::clamp(q, 0.0, 1.0);
    // Nearest-rank: the smallest value with at least q of the samples at or below it.
    const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    return sorted[rank == 0 ? 0 : rank - 1];
}

LatencySummary LatencyRecorder::summary() const {
    LatencySummary s;
    s.count = samples_ms_.size();
    if (s.count == 0) {
        return s;
    }
    std::vector<double> sorted = samples_ms_;
    std::ranges::sort(sorted);
    s.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(s.count);
    s.p50_ms = percentile(sorted, 0.50);
    s.p95_ms = percentile(sorted, 0.95);
    s.p99_ms = percentile(sorted, 0.99);
    s.max_ms = sorted.back();
    return s;
}

}  // namespace poseforge
