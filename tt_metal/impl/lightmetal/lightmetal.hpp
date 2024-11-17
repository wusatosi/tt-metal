// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include "binary_generated.h"

namespace tt::tt_metal {
inline namespace v0 {

// struct LightMetalTraceConfig {
//     bool capture_enabled = false;
//     bool auto_serialize_metal_trace = true;
//     std::string filename = "/tmp/light_metal_trace.bin";
// };

// struct LightMetalTrace {
//     std::vector<std::pair<uint32_t, detail::TraceDescriptor>> traces;
//     LightMetalTraceConfig config;
// };

using TraceDescriptorByTraceIdOffset = flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>;

//////////////////////////////
// Serialization            //
//////////////////////////////

// Convert TraceDescriptor + trace_id to flatbuffer
TraceDescriptorByTraceIdOffset toFlatBuffer(flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& traceDesc, const uint32_t trace_id) {

    auto trace_data_offset = builder.CreateVector(traceDesc.data);

    // Create TraceDescriptor wrapped in TraceDescriptorByTraceId
    return tt::target::lightmetal::CreateTraceDescriptorByTraceId(
        builder,
        trace_id,
        tt::target::lightmetal::CreateTraceDescriptor(
            builder,
            trace_data_offset,
            traceDesc.num_completion_worker_cores,
            traceDesc.num_traced_programs_needing_go_signal_multicast,
            traceDesc.num_traced_programs_needing_go_signal_unicast
        )
    );
}

// Take built up light metal trace data and serialize to flatbuffer binary.
std::vector<std::uint8_t> createLightMetalBinary(LightMetalTrace& light_metal_trace) {

    flatbuffers::FlatBufferBuilder builder;

    // Serialize each trace
    std::vector<flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>> trace_desc_vec;
    for (const auto& trace_pair : light_metal_trace.traces) {
        // Assuming toFlatBuffer returns flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>
        trace_desc_vec.emplace_back(toFlatBuffer(builder, trace_pair.second, trace_pair.first));
        log_info(tt::LogMetal, "KCM {} for trace_id: {}", __FUNCTION__, trace_pair.first);
    }

    // Build the binary - sorting tracedesc required since trace_id is sorted key.
    auto sorted_trace_descriptors = builder.CreateVectorOfSortedTables(&trace_desc_vec);
    auto light_metal_binary = CreateLightMetalBinary(builder, sorted_trace_descriptors);
    builder.Finish(light_metal_binary);

    // Copy binary data to vector and return.
    const uint8_t* buffer_ptr = builder.GetBufferPointer();
    size_t buffer_size = builder.GetSize();
    return std::vector<std::uint8_t>(buffer_ptr, buffer_ptr + buffer_size);

}

// Write an arbitrary binary blob to file.
bool writeBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob) {
    // Open the file in binary mode
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return false;
    }

    // Write the binary blob to the file
    outFile.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    if (!outFile.good()) {
        std::cerr << "Error: Failed to write binary data to file " << filename << "." << std::endl;
        return false;
    }

    outFile.close();
    return true;
}

//////////////////////////////
// Deserialization          //
//////////////////////////////

// Function to convert from Flatbuffer TraceDescriptor to C++ struct
detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc) {
    detail::TraceDescriptor traceDesc;

    if (fb_desc) {
        // Get trace_data
        auto trace_data_fb = fb_desc->trace_data();
        if (trace_data_fb) {
            traceDesc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
        }
        traceDesc.num_completion_worker_cores = fb_desc->num_completion_worker_cores();
        traceDesc.num_traced_programs_needing_go_signal_multicast = fb_desc->num_traced_programs_needing_go_signal_multicast();
        traceDesc.num_traced_programs_needing_go_signal_unicast = fb_desc->num_traced_programs_needing_go_signal_unicast();
    } else {
        // Handle the case where 'fb_desc' is null
        // You can throw an exception or handle it as per your error handling strategy
        std::cerr << "TraceDescriptor is null." << std::endl;
    }

    return traceDesc;
}


// Read an arbitrary binary blob from file.
std::vector<uint8_t> readBinaryBlobFromFile(const std::string& filename) {
    // Open the file in binary mode and position at the end to determine file size
    std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Error: Unable to open file " + filename + " for reading.");
    }

    // Determine the size of the file
    std::streamsize size = inFile.tellg();
    if (size < 0) {
        throw std::runtime_error("Error: Failed to determine the size of file " + filename + ".");
    }

    // Rewind to the beginning of the file
    inFile.seekg(0, std::ios::beg);

    // Read the binary data into the vector
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (!inFile.read(reinterpret_cast<char*>(blob.data()), size)) {
        throw std::runtime_error("Error: Failed to read binary data from file " + filename + ".");
    }

    return blob;
}

// FIXME - For simplicity right now this is re-reading from disk every time, but that's not efficient.
// Function to deserialize a specific trace by trace_id
std::optional<detail::TraceDescriptor> getTraceByTraceId(const std::string& filename, uint32_t target_trace_id) {
    try {
        // Read the binary blob from file and get pointer to buffer
        std::vector<uint8_t> blob = readBinaryBlobFromFile(filename);
        const uint8_t* data = blob.data();
        size_t size = blob.size();

        // Verify the buffer
        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify Flatbuffer data." << std::endl;
            return std::nullopt;
        }

        // Get the root of the Flatbuffer
        const auto* lm_binary = target::lightmetal::GetLightMetalBinary(data);

        // Access the trace_descriptors vector
        auto trace_descriptors = lm_binary->trace_descriptors();
        if (trace_descriptors) {
            // Use LookupByKey to find the TraceDescriptorByTraceId with the specific trace_id
            const auto* fb_trace_desc_by_id = trace_descriptors->LookupByKey(target_trace_id);
            if (fb_trace_desc_by_id) {
                const target::lightmetal::TraceDescriptor* fb_desc = fb_trace_desc_by_id->desc();
                if (fb_desc) {
                    // Convert to C++ struct
                    detail::TraceDescriptor traceDesc = fromFlatBuffer(fb_desc);
                    return traceDesc;
                } else {
                    std::cerr << "Descriptor is null for trace_id: " << target_trace_id << std::endl;
                    return std::nullopt;
                }
            } else {
                std::cout << "Trace ID " << target_trace_id << " not found." << std::endl;
                return std::nullopt;
            }
        } else {
            std::cerr << "No trace_descriptors found in the Flatbuffer file." << std::endl;
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return std::nullopt;
    }
}


}  // namespace v0
}  // namespace tt::tt_metal
