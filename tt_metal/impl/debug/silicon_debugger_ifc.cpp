// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/kernel.hpp>
#include "llrt/llrt.hpp"
#include "dev_msgs.h"

namespace tt::tt_metal {

// This mutex is used to synchronize access to the silicon debugger interface file, as multiple threads may
// be writing to the file concurrently.
std::mutex silicon_debugger_ifc_mutex;

// The file containing the map from kernel ID to source file
const std::string SILICON_DEBUGER_RUNTIME_DATA_FILE("generated/silicon_debugger/runtime_data.yaml");

FILE *OpenSiliconDebuggerInterfaceFile() {
    FILE *f;
    if ((f = fopen((MetalContext::instance().rtoptions().get_root_dir() + SILICON_DEBUGER_RUNTIME_DATA_FILE).c_str(), "a")) == nullptr) {
        TT_THROW("Failed to open silicon debugger data file: {}", SILICON_DEBUGER_RUNTIME_DATA_FILE);
    }
    return f;
}

// Initilize file used by the silicon debugger (gdb server)
void InitSiliconDebuggerInterfaceFile() {
    static bool created = false;
    if (created) {
        return;
    }
    created = true;

    FILE *f;
    string fname = MetalContext::instance().rtoptions().get_root_dir() + SILICON_DEBUGER_RUNTIME_DATA_FILE;
    string parent_dir = std::filesystem::path(fname).parent_path().string();
    std::filesystem::create_directories(parent_dir);

    if ((f = fopen(fname.c_str(), "w")) == nullptr) {
        TT_THROW("Failed to create silicon debugger data file: {}", SILICON_DEBUGER_RUNTIME_DATA_FILE);
    }

    fprintf(f, "kernels:\n");
    fclose(f);
}

// We log the kernel ID and the paths to the source file and the build output directory
void SiliconDebuggerInterfaceLogKernel(std::shared_ptr<Kernel> kernel, const tt::tt_metal::JitBuildOptions &build_options) {
    std::lock_guard<std::mutex> lock(silicon_debugger_ifc_mutex); // Allow for concurrent threads to log kernel information
    FILE *gdb_ifc_file = tt::tt_metal::OpenSiliconDebuggerInterfaceFile();
    fprintf(gdb_ifc_file, "  %d:\n    src: %s\n    out: %s\n", kernel->get_watcher_kernel_id(), kernel->kernel_source().name().c_str(), build_options.path.c_str());
    fflush(gdb_ifc_file);
    fclose(gdb_ifc_file);
}

}  // namespace tt::tt_metal
