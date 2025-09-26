#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include "gc_allocator_core.h"
#include "retry_strategy.h"
#include "allocator_stats.h"

namespace py = pybind11;

using namespace gc_allocator;

PYBIND11_MODULE(gc_allocator_core, m) {
    m.doc() = "GC Allocator for PyTorch CUDA memory management";
    
    // RetryConfig structure
    // JO:
    py::class_<RetryConfig>(m, "RetryConfig")
        .def(py::init<>())
        .def_readwrite("max_retries", &RetryConfig::max_retries)
        .def_readwrite("initial_delay", &RetryConfig::initial_delay)
        .def_readwrite("backoff_multiplier", &RetryConfig::backoff_multiplier)
        .def_readwrite("max_delay", &RetryConfig::max_delay)
        .def_readwrite("enable_cache_flush", &RetryConfig::enable_cache_flush)
        // JO+ .def_readwrite("enable_gradient_checkpointing", &RetryConfig::enable_gradient_checkpointing);
        .def_readwrite("enable_gradient_checkpointing", &RetryConfig::enable_gradient_checkpointing)
        // Enhanced RetryConfig with proper milliseconds handling
        .def_property("initial_delay_ms",
            [](const RetryConfig& self) {
                return self.initial_delay.count();
            },
            [](RetryConfig& self, int ms) {
                self.initial_delay = std::chrono::milliseconds(ms);
            })
        .def_property("max_delay_ms",
            [](const RetryConfig& self) {
                return self.max_delay.count();
            },
            [](RetryConfig& self, int ms) {
                self.max_delay = std::chrono::milliseconds(ms);
            });

    // JO+
    // Add CombinedStats binding
    py::class_<GCAllocator::CombinedStats>(m, "CombinedStats")
    	.def_readonly("allocation_stats", &GCAllocator::CombinedStats::allocation_stats)
    	.def_readonly("retry_stats", &GCAllocator::CombinedStats::retry_stats)
    	.def("__str__", &GCAllocator::CombinedStats::toString)
    	.def("__repr__", [](const GCAllocator::CombinedStats& self) {
            return "<CombinedStats allocations=" + 
                   std::to_string(self.allocation_stats.getTotalAllocations()) + 
                   " retries=" + 
                   std::to_string(self.retry_stats.getTotalRetryAttempts()) + ">";
    	});

    // Enhanced manager methods
    m.def("get_combined_stats", []() {
    	auto& manager = GCAllocatorManager::getInstance();
    	if (manager.isInstalled() && manager.getAllocator()) {
            return manager.getAllocator()->getCombinedStats();
    	}
    	throw std::runtime_error("GCAllocator not installed");
    }, "Get combined allocation and retry statistics");

    // Direct retry stats access
    m.def("get_retry_stats", []() {
    	auto& manager = GCAllocatorManager::getInstance();
    	if (manager.isInstalled()) {
           return manager.getRetryStats();
    	}
    	throw std::runtime_error("GCAllocator not installed");
    }, "Get retry statistics directly");
    // JO-
    
    // RetryStats structure - use getter methods instead of direct atomic access
    py::class_<RetryStats>(m, "RetryStats")
        .def("get_total_retry_attempts", &RetryStats::getTotalRetryAttempts)
        .def("get_cache_flushes", &RetryStats::getCacheFlushes)
        .def("get_checkpoint_activations", &RetryStats::getCheckpointActivations)
        .def("get_successful_recoveries", &RetryStats::getSuccessfulRecoveries)
	// JO+
        .def("get_exhausted_retries", &RetryStats::getExhaustedRetries)
        .def("get_terminal_failures", &RetryStats::getTerminalFailures)
        .def("reset", &RetryStats::reset);
    
    // AllocationStats structure
    py::class_<AllocationStats>(m, "AllocationStats")
        .def("get_total_allocations", &AllocationStats::getTotalAllocations)
        .def("get_total_deallocations", &AllocationStats::getTotalDeallocations)
        .def("get_total_bytes_allocated", &AllocationStats::getTotalBytesAllocated)
        .def("get_current_bytes_allocated", &AllocationStats::getCurrentBytesAllocated)
        .def("get_peak_bytes_allocated", &AllocationStats::getPeakBytesAllocated)
        .def("get_oom_count", &AllocationStats::getOOMCount)
        .def("get_active_devices", &AllocationStats::getActiveDevices)
        .def("reset", &AllocationStats::reset)
        .def("__str__", &AllocationStats::toString);
 
    // GCAllocatorManager - properly expose the singleton
    py::class_<GCAllocatorManager>(m, "GCAllocatorManager")
        .def("install_allocator", &GCAllocatorManager::installAllocator)
        .def("uninstall_allocator", &GCAllocatorManager::uninstallAllocator)
        .def("is_installed", &GCAllocatorManager::isInstalled)
        .def("enable_logging", &GCAllocatorManager::enableLogging)
        .def("disable_logging", &GCAllocatorManager::disableLogging)
        .def("configure_retry_strategy", &GCAllocatorManager::configureRetryStrategy)
        .def("register_checkpoint_callback", 
             [](GCAllocatorManager& self, py::function callback) {
                 // Convert Python function to C++ std::function
                 std::function<bool()> cpp_callback = [callback]() -> bool {
                     try {
                         return callback().cast<bool>();
                     } catch (const std::exception& e) {
                         py::print("Checkpoint callback error:", e.what());
                         return false;
                     }
                 };
                 self.registerCheckpointCallback(cpp_callback);
             })
        .def("get_retry_stats", &GCAllocatorManager::getRetryStats, 
             py::return_value_policy::reference_internal)
        .def("reset_retry_stats", &GCAllocatorManager::resetRetryStats);
  
    // Helper function to create milliseconds duration
    m.def("milliseconds", [](int64_t ms) {
        return std::chrono::milliseconds(ms);
    });
    // Global function to get manager instance
    m.def("get_manager", &gc_allocator::GCAllocatorManager::getInstance,
          py::return_value_policy::reference);
    // Main module functions
    m.def("install_allocator", []() {
        GCAllocatorManager::getInstance().installAllocator();
    }, "Install the GCAllocator as the default CUDA allocator");

    m.def("uninstall_allocator", []() {
        GCAllocatorManager::getInstance().uninstallAllocator();
    }, "Uninstall the GCAllocator and restore the original allocator");

    m.def("is_installed", []() {
        return GCAllocatorManager::getInstance().isInstalled();
    }, "Check if GCAllocator is currently installed");

    m.def("get_stats", []() {
        auto* allocator = GCAllocatorManager::getInstance().getAllocator();
        if (allocator) {
	    // JO + 20250925
            //return allocator->getStats();
	    const AllocationStats& stats = allocator->getStats();
	    return stats;
        }
        throw std::runtime_error("GCAllocator is not installed");
    }, "Get current allocation statistics");

    m.def("reset_stats", []() {
        auto* allocator = GCAllocatorManager::getInstance().getAllocator();
        if (allocator) {
            allocator->resetStats();
        } else {
            throw std::runtime_error("GCAllocator is not installed");
        }
    }, "Reset allocation statistics");

    m.def("set_logging_enabled", [](bool enabled) {
        auto* allocator = GCAllocatorManager::getInstance().getAllocator();
        if (allocator) {
            allocator->setLoggingEnabled(enabled);
        } else {
            throw std::runtime_error("GCAllocator is not installed");
        }
    }, py::arg("enabled"), "Enable or disable logging");

    m.def("is_logging_enabled", []() {
        auto* allocator = GCAllocatorManager::getInstance().getAllocator();
        if (allocator) {
            return allocator->isLoggingEnabled();
        }
        return false;
    }, "Check if logging is enabled");
    // Manual allocation tracking functions for testing
    m.def("track_allocation", [](size_t ptr_int, size_t size, int device) {
        auto* allocator = GCAllocatorManager::getInstance().getAllocator();
        if (allocator) {
            void* ptr = reinterpret_cast<void*>(ptr_int);
            allocator->recordAllocation(ptr, size, device);
        }
    }, "Manually track an allocation",
       py::arg("ptr"), py::arg("size"), py::arg("device") = 0);

    m.def("track_deallocation", [](size_t ptr_int) {
        auto* allocator = GCAllocatorManager::getInstance().getAllocator();
        if (allocator) {
            void* ptr = reinterpret_cast<void*>(ptr_int);
            allocator->recordDeallocation(ptr);
        }
    }, "Manually track a deallocation", py::arg("ptr"));
    
    // Helper function to create RetryConfig with dictionary
    m.def("create_retry_config", [](py::dict config) {
        RetryConfig retry_config;
        
        if (config.contains("max_retries")) {
            retry_config.max_retries = config["max_retries"].cast<int>();
        }
        if (config.contains("initial_delay_ms")) {
            retry_config.initial_delay = std::chrono::milliseconds(
                config["initial_delay_ms"].cast<int>());
        }
        if (config.contains("backoff_multiplier")) {
            retry_config.backoff_multiplier = config["backoff_multiplier"].cast<double>();
        }
        if (config.contains("max_delay_ms")) {
            retry_config.max_delay = std::chrono::milliseconds(
                config["max_delay_ms"].cast<int>());
        }
        if (config.contains("enable_cache_flush")) {
            retry_config.enable_cache_flush = config["enable_cache_flush"].cast<bool>();
        }
        if (config.contains("enable_gradient_checkpointing")) {
            retry_config.enable_gradient_checkpointing = 
                config["enable_gradient_checkpointing"].cast<bool>();
        }
        
        return retry_config;
    });
}
