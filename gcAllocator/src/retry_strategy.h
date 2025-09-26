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

#ifndef RETRY_STRATEGY_H
#define RETRY_STRATEGY_H

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>

namespace gc_allocator {

// Configuration for retry behavior
struct RetryConfig {
    int max_retries = 3;
    std::chrono::milliseconds initial_delay{100};
    double backoff_multiplier = 2.0;
    std::chrono::milliseconds max_delay{5000};
    bool enable_cache_flush = true;
    bool enable_gradient_checkpointing = false;
};

//
struct RetryStats {
    std::atomic<size_t> total_retry_attempts{0};
    std::atomic<size_t> cache_flushes{0};
    std::atomic<size_t> checkpoint_activations{0};
    std::atomic<size_t> successful_recoveries{0};
    std::atomic<size_t> exhausted_retries{0};
    std::atomic<size_t> terminal_failures{0}; // final failures that propagate OOM
    std::atomic<size_t> failed_intermediate_attempts{0};


    // Default constructor
    RetryStats() = default;

    // Copy constructor - load atomic values
    RetryStats(const RetryStats& other)
        : total_retry_attempts(other.total_retry_attempts.load()),
          cache_flushes(other.cache_flushes.load()),
          checkpoint_activations(other.checkpoint_activations.load()),
          successful_recoveries(other.successful_recoveries.load()),
	  exhausted_retries(other.exhausted_retries.load()),
	  terminal_failures(other.terminal_failures.load()),
	  failed_intermediate_attempts(other.failed_intermediate_attempts.load()){
    }

    // Copy assignment operator - load atomic values
    RetryStats& operator=(const RetryStats& other) {
        if (this != &other) {
            total_retry_attempts.store(other.total_retry_attempts.load());
            cache_flushes.store(other.cache_flushes.load());
            checkpoint_activations.store(other.checkpoint_activations.load());
            successful_recoveries.store(other.successful_recoveries.load());
	    exhausted_retries.store(other.exhausted_retries.load());
	    terminal_failures.store(other.terminal_failures.load());
	    failed_intermediate_attempts.store(other.failed_intermediate_attempts.load());
        }
        return *this;
    }

    // Move constructor
    RetryStats(RetryStats&& other) noexcept
        : total_retry_attempts(other.total_retry_attempts.load()),
          cache_flushes(other.cache_flushes.load()),
          checkpoint_activations(other.checkpoint_activations.load()),
          successful_recoveries(other.successful_recoveries.load()),
	  exhausted_retries(other.exhausted_retries.load()),
	  terminal_failures(other.terminal_failures.load()),
	  failed_intermediate_attempts(other.failed_intermediate_attempts.load()){
    }

    // Move assignment operator
    RetryStats& operator=(RetryStats&& other) noexcept {
        if (this != &other) {
            total_retry_attempts.store(other.total_retry_attempts.load());
            cache_flushes.store(other.cache_flushes.load());
            checkpoint_activations.store(other.checkpoint_activations.load());
            successful_recoveries.store(other.successful_recoveries.load());
	    exhausted_retries.store(other.exhausted_retries.load());
	    terminal_failures.store(other.terminal_failures.load());
	    failed_intermediate_attempts.store(other.failed_intermediate_attempts.load());
        }
        return *this;
    }

    // Existing getter methods remain...
    size_t getTotalRetryAttempts() const { return total_retry_attempts.load(); }
    size_t getCacheFlushes() const { return cache_flushes.load(); }
    size_t getCheckpointActivations() const { return checkpoint_activations.load(); }
    size_t getSuccessfulRecoveries() const { return successful_recoveries.load(); }
    size_t getExhaustedRetries() const { return exhausted_retries.load(); }
    size_t getTerminalFailures() const { return terminal_failures.load(); }
    size_t getFailedIntermediateAttempts() const { return failed_intermediate_attempts.load(); }

    void reset() {
        total_retry_attempts.store(0);
        cache_flushes.store(0);
        checkpoint_activations.store(0);
        successful_recoveries.store(0);
	exhausted_retries.store(0);
        terminal_failures.store(0);
	failed_intermediate_attempts.store(0);
    }
};
// JO-

class RetryStrategy {
public:
    explicit RetryStrategy(const RetryConfig& config = RetryConfig{});
    
    // Execute allocation with retry logic
    template<typename AllocatorFunc>
    c10::DataPtr executeWithRetry(AllocatorFunc allocator_func, size_t size, int device);
    
    // Configuration management
    void updateConfig(const RetryConfig& config);
    const RetryConfig& getConfig() const { return config_; }
    
    // Callback management
    void registerCheckpointCallback(std::function<bool()> callback);
    void clearCheckpointCallback();
    
    // Statistics
    const RetryStats& getStats() const { return stats_; }
    void resetStats();

private:
    RetryConfig config_;
    RetryStats stats_;
    std::function<bool()> checkpoint_callback_;
    mutable std::mutex callback_mutex_;
    
    bool performCacheFlush(int device);
    bool performCheckpointing();
    void performBackoff(int attempt);
};

// JO+
// total_retry_attempts counts only attempts after the initial (attempt 0).
// This keeps the metric focused on recovery behavior rather than baseline allocation.
// JO -

// Template implementation
template<typename AllocatorFunc>
c10::DataPtr RetryStrategy::executeWithRetry(AllocatorFunc allocator_func, size_t size, int device) {
    c10::DataPtr result;

    std::cout << "[RetryStrategy] Starting allocation attempt for " << size << " bytes on device " << device << std::endl;
    
    for (int attempt = 0; attempt <= config_.max_retries; ++attempt) {
        try {
            if (attempt > 0) {
                //if (isLoggingEnabled()) {
                    std::cout << "[RetryStrategy] Retry attempt " << attempt 
                              << "/" << config_.max_retries << std::endl;
                //}
                stats_.total_retry_attempts.fetch_add(1);
                
                // Perform cache flush if enabled
                if (config_.enable_cache_flush) {
                    //if (isLoggingEnabled()) {
                        std::cout << "[RetryStrategy] Performing cache flush..." << std::endl;
                    //}
                    performCacheFlush(device);
                }
                
                // Perform checkpointing if enabled and we've had multiple failures
                if (config_.enable_gradient_checkpointing && attempt >= 2) {
                    //if (isLoggingEnabled()) {
                        std::cout << "[RetryStrategy] Performing checkpointing..." << std::endl;
                    //}
                    performCheckpointing();
                }
                
                // Wait with exponential backoff
                performBackoff(attempt);
            }
            
            // Attempt allocation
            result = allocator_func();
            
            if (result.get()) {
                if (attempt > 0) {
                    //if (isLoggingEnabled()) {
                        std::cout << "[RetryStrategy] Allocation succeeded after " 
                                  << attempt << " retries" << std::endl;
                    //}
                    stats_.successful_recoveries.fetch_add(1);
                }
                return result;
            }
            
        } catch (const c10::OutOfMemoryError& e) {
            //if (isLoggingEnabled()) {
                std::cout << "[RetryStrategy] OOM on attempt " << (attempt + 1) 
                          << ": " << e.what() << std::endl;
            //}
            if (attempt == config_.max_retries) {
                //if (isLoggingEnabled()) {
                    std::cout << "[RetryStrategy] All retry attempts exhausted" << std::endl;
                //}
		// JO+
	        stats_.exhausted_retries.fetch_add(1);
                stats_.terminal_failures.fetch_add(1);

                throw;
            //}
	    // JO+
	    }else {
               stats_.failed_intermediate_attempts.fetch_add(1);
               continue;
            }  
            //continue;
        }
    }
    
    throw c10::OutOfMemoryError(
        c10::SourceLocation{__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
        "All retry attempts exhausted"
    );
}

// Explicit template instantiation
template c10::DataPtr RetryStrategy::executeWithRetry<std::function<c10::DataPtr()>>(
    std::function<c10::DataPtr()> allocator_func, size_t size, int device);

} // namespace gc_allocator

#endif // RETRY_STRATEGY_H
