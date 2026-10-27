#include "poseforge/stats.hpp"

#include <gtest/gtest.h>

using namespace poseforge;
using namespace std::chrono_literals;

TEST(Stats, EmptyRecorderGivesZeroSummary) {
    const LatencyRecorder r;
    const auto s = r.summary();
    EXPECT_EQ(s.count, 0u);
    EXPECT_EQ(s.p99_ms, 0.0);
}

TEST(Stats, NearestRankPercentiles) {
    LatencyRecorder r;
    for (int i = 100; i >= 1; --i) r.add(std::chrono::milliseconds(i));  // unsorted input
    const auto s = r.summary();
    EXPECT_EQ(s.count, 100u);
    EXPECT_DOUBLE_EQ(s.p50_ms, 50.0);
    EXPECT_DOUBLE_EQ(s.p95_ms, 95.0);
    EXPECT_DOUBLE_EQ(s.p99_ms, 99.0);
    EXPECT_DOUBLE_EQ(s.max_ms, 100.0);
    EXPECT_DOUBLE_EQ(s.mean_ms, 50.5);
}

TEST(Stats, SubMillisecondResolution) {
    LatencyRecorder r;
    r.add(250us);
    EXPECT_DOUBLE_EQ(r.summary().p50_ms, 0.25);
}

TEST(Stats, PercentileOfSingleSample) {
    EXPECT_DOUBLE_EQ(percentile({7.0}, 0.0), 7.0);
    EXPECT_DOUBLE_EQ(percentile({7.0}, 1.0), 7.0);
    EXPECT_THROW(percentile({}, 0.5), std::invalid_argument);
}
