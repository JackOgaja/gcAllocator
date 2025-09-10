#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "gc_allocator_core.h"
#include "allocator_stats.h"

namespace py = pybind11;
using namespace gc_allocator;

PYBIND11_MODULE(gc_allocator_core, m) {
    m.doc() = "GCAllocator - Graceful CUDA Allocator for PyTorch";
    
    // AllocationStats class
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
            return allocator->getStats();
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
}
