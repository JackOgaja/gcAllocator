/*
 * MIT License
 * 
 * Copyright (c) 2025 Jack Ogaja
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "retry_strategy.h"
#include <c10/cuda/CUDACachingAllocator.h>

namespace gc_allocator {

RetryStrategy::RetryStrategy(const RetryConfig& config) : config_(config) {
    // Constructor implementation
}

void RetryStrategy::updateConfig(const RetryConfig& config) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    config_ = config;
}

void RetryStrategy::registerCheckpointCallback(std::function<bool()> callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    checkpoint_callback_ = std::move(callback);
}

void RetryStrategy::clearCheckpointCallback() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    checkpoint_callback_ = nullptr;
}

void RetryStrategy::resetStats() {
    stats_.reset();
}

bool RetryStrategy::performCacheFlush(int device) {
    try {
        // Flush PyTorch's CUDA cache
        c10::cuda::CUDACachingAllocator::emptyCache();
        stats_.cache_flushes.fetch_add(1);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[RetryStrategy] Cache flush failed: " << e.what() << std::endl;
        return false;
    }
}

bool RetryStrategy::performCheckpointing() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (checkpoint_callback_) {
        try {
            bool success = checkpoint_callback_();
            if (success) {
                stats_.checkpoint_activations.fetch_add(1);
            }
            return success;
        } catch (const std::exception& e) {
            std::cerr << "[RetryStrategy] Checkpoint callback failed: " << e.what() << std::endl;
            return false;
        }
    }
    return false;
}

void RetryStrategy::performBackoff(int attempt) {
    auto delay = config_.initial_delay;
    
    // Apply exponential backoff
    for (int i = 1; i < attempt; ++i) {
        delay = std::chrono::milliseconds(
            static_cast<long long>(delay.count() * config_.backoff_multiplier)
        );
        if (delay > config_.max_delay) {
            delay = config_.max_delay;
            break;
        }
    }
    
    std::this_thread::sleep_for(delay);
}

} // namespace gc_allocator
