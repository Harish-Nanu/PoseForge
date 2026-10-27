#include "poseforge/bounded_queue.hpp"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using namespace poseforge;
using namespace std::chrono_literals;

TEST(BoundedQueue, FifoOrder) {
    BoundedQueue<int> q(3);
    q.push(1);
    q.push(2);
    q.push(3);
    EXPECT_EQ(q.pop(), 1);
    EXPECT_EQ(q.pop(), 2);
    EXPECT_EQ(q.pop(), 3);
}

TEST(BoundedQueue, RejectsZeroCapacity) {
    EXPECT_THROW(BoundedQueue<int>(0), std::invalid_argument);
}

TEST(BoundedQueue, DropOldestEvictsAndReturnsTheOldItem) {
    BoundedQueue<int> q(2, OverflowPolicy::DropOldest);
    EXPECT_FALSE(q.push(1).evicted.has_value());
    EXPECT_FALSE(q.push(2).evicted.has_value());
    const auto r = q.push(3);
    EXPECT_TRUE(r.accepted);
    ASSERT_TRUE(r.evicted.has_value());
    EXPECT_EQ(*r.evicted, 1);
    EXPECT_EQ(q.size(), 2u);
    EXPECT_EQ(q.pop(), 2);
    EXPECT_EQ(q.pop(), 3);
}

TEST(BoundedQueue, BlockPolicyWaitsForSpace) {
    BoundedQueue<int> q(1);
    q.push(1);
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        q.push(2);
        pushed = true;
    });
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(pushed.load());  // still blocked: queue is full
    EXPECT_EQ(q.pop(), 1);
    producer.join();
    EXPECT_TRUE(pushed.load());
    EXPECT_EQ(q.pop(), 2);
}

TEST(BoundedQueue, CloseDrainsThenEnds) {
    BoundedQueue<int> q(4);
    q.push(1);
    q.close();
    EXPECT_FALSE(q.push(2).accepted);
    EXPECT_EQ(q.pop(), 1);
    EXPECT_EQ(q.pop(), std::nullopt);
}

TEST(BoundedQueue, CloseWakesBlockedProducerAndConsumer) {
    BoundedQueue<int> full(1);
    full.push(1);
    BoundedQueue<int> empty(1);
    std::thread producer([&] { EXPECT_FALSE(full.push(2).accepted); });
    std::thread consumer([&] { EXPECT_EQ(empty.pop(), std::nullopt); });
    std::this_thread::sleep_for(20ms);
    full.close();
    empty.close();
    producer.join();
    consumer.join();
}

TEST(BoundedQueue, ManyProducersManyConsumersLoseNothing) {
    constexpr int kProducers = 4;
    constexpr int kConsumers = 4;
    constexpr int kPerProducer = 5000;
    BoundedQueue<int> q(8);

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kPerProducer; ++i) q.push(p * kPerProducer + i);
        });
    }
    std::atomic<long long> sum{0};
    std::atomic<int> count{0};
    std::vector<std::thread> consumers;
    for (int c = 0; c < kConsumers; ++c) {
        consumers.emplace_back([&] {
            while (auto v = q.pop()) {
                sum += *v;
                ++count;
            }
        });
    }
    for (auto& t : producers) t.join();
    q.close();
    for (auto& t : consumers) t.join();

    constexpr long long n = kProducers * kPerProducer;
    EXPECT_EQ(count.load(), n);
    EXPECT_EQ(sum.load(), n * (n - 1) / 2);
}
