#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace poseforge {

enum class OverflowPolicy {
    Block,       // producer waits for space: no frame is lost (offline video, benchmarks)
    DropOldest,  // producer evicts the oldest item: lowest latency (live camera)
};

// Multi-producer multi-consumer bounded queue between pipeline stages. The capacity bounds both
// memory use and how stale a frame can get while waiting.
template <typename T>
class BoundedQueue {
public:
    struct PushResult {
        bool accepted = false;     // false only if the queue was closed
        std::optional<T> evicted;  // set when DropOldest made room
    };

    explicit BoundedQueue(std::size_t capacity, OverflowPolicy policy = OverflowPolicy::Block)
        : capacity_(capacity), policy_(policy) {
        if (capacity == 0) {
            throw std::invalid_argument("queue capacity must be at least 1");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    PushResult push(T item) {
        PushResult result;
        {
            std::unique_lock lock(mutex_);
            if (policy_ == OverflowPolicy::Block) {
                not_full_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
            }
            if (closed_) {
                return result;
            }
            if (items_.size() >= capacity_) {
                result.evicted = std::move(items_.front());
                items_.pop_front();
            }
            items_.push_back(std::move(item));
            result.accepted = true;
        }
        not_empty_.notify_one();
        return result;
    }

    // Blocks until an item is available. Returns nullopt once the queue is closed and drained.
    std::optional<T> pop() {
        std::optional<T> item;
        {
            std::unique_lock lock(mutex_);
            not_empty_.wait(lock, [&] { return closed_ || !items_.empty(); });
            if (items_.empty()) {
                return std::nullopt;
            }
            item = std::move(items_.front());
            items_.pop_front();
        }
        not_full_.notify_one();
        return item;
    }

    // Wakes every waiter. Later pushes are rejected; consumers still drain what is queued.
    void close() {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return items_.size();
    }

    bool closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    std::size_t capacity() const { return capacity_; }

private:
    const std::size_t capacity_;
    const OverflowPolicy policy_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> items_;
    bool closed_ = false;
};

}  // namespace poseforge
