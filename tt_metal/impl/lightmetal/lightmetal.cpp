// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal.hpp"
#include "binary_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt::tt_metal {
inline namespace v0 {

TraceDescriptorByTraceIdOffset toFlatBuffer(flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& traceDesc, const uint32_t trace_id) {
    auto trace_data_offset = builder.CreateVector(traceDesc.data);

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

std::vector<std::uint8_t> createLightMetalBinary(LightMetalTrace& light_metal_trace) {
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>> trace_desc_vec;
    for (const auto& trace_pair : light_metal_trace.traces) {
        trace_desc_vec.emplace_back(toFlatBuffer(builder, trace_pair.second, trace_pair.first));
        log_info(tt::LogMetal, "KCM {} for trace_id: {}", __FUNCTION__, trace_pair.first);
    }

    auto sorted_trace_descriptors = builder.CreateVectorOfSortedTables(&trace_desc_vec);
    auto light_metal_binary = CreateLightMetalBinary(builder, sorted_trace_descriptors);
    builder.Finish(light_metal_binary);

    const uint8_t* buffer_ptr = builder.GetBufferPointer();
    size_t buffer_size = builder.GetSize();
    return std::vector<std::uint8_t>(buffer_ptr, buffer_ptr + buffer_size);
}

bool writeBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return false;
    }

    outFile.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    if (!outFile.good()) {
        std::cerr << "Error: Failed to write binary data to file " << filename << "." << std::endl;
        return false;
    }

    outFile.close();
    return true;
}

detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc) {
    detail::TraceDescriptor traceDesc;

    if (fb_desc) {
        auto trace_data_fb = fb_desc->trace_data();
        if (trace_data_fb) {
            traceDesc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
        }
        traceDesc.num_completion_worker_cores = fb_desc->num_completion_worker_cores();
        traceDesc.num_traced_programs_needing_go_signal_multicast = fb_desc->num_traced_programs_needing_go_signal_multicast();
        traceDesc.num_traced_programs_needing_go_signal_unicast = fb_desc->num_traced_programs_needing_go_signal_unicast();
    } else {
        std::cerr << "TraceDescriptor is null." << std::endl;
    }

    return traceDesc;
}

std::vector<uint8_t> readBinaryBlobFromFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Error: Unable to open file " + filename + " for reading.");
    }

    std::streamsize size = inFile.tellg();
    if (size < 0) {
        throw std::runtime_error("Error: Failed to determine the size of file " + filename + ".");
    }

    inFile.seekg(0, std::ios::beg);

    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (!inFile.read(reinterpret_cast<char*>(blob.data()), size)) {
        throw std::runtime_error("Error: Failed to read binary data from file " + filename + ".");
    }

    return blob;
}

std::optional<detail::TraceDescriptor> getTraceByTraceId(const std::string& filename, uint32_t target_trace_id) {
    try {
        std::vector<uint8_t> blob = readBinaryBlobFromFile(filename);
        const uint8_t* data = blob.data();
        size_t size = blob.size();

        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify Flatbuffer data." << std::endl;
            return std::nullopt;
        }

        const auto* lm_binary = target::lightmetal::GetLightMetalBinary(data);
        auto trace_descriptors = lm_binary->trace_descriptors();
        if (trace_descriptors) {
            const auto* fb_trace_desc_by_id = trace_descriptors->LookupByKey(target_trace_id);
            if (fb_trace_desc_by_id) {
                const target::lightmetal::TraceDescriptor* fb_desc = fb_trace_desc_by_id->desc();
                if (fb_desc) {
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
