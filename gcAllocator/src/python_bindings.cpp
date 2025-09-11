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
    
    m.def("enable_logging", []() {
        GCAllocatorManager::getInstance().enableLogging();
    }, "Enable detailed allocation logging");
    
    m.def("disable_logging", []() {
        GCAllocatorManager::getInstance().disableLogging();
    }, "Disable detailed allocation logging");
}
