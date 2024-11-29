// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal.hpp"
#include "binary_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Serialization Functions          //
//////////////////////////////////////

TraceDescriptorByTraceIdOffset toFlatBuffer(
    flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& traceDesc, const uint32_t trace_id) {
    // Serialize the trace_data vector
    auto trace_data_offset = builder.CreateVector(traceDesc.data);

    // Serialize the sub_device_descriptors (map)
    std::vector<flatbuffers::Offset<tt::target::lightmetal::SubDeviceDescriptorMapping>> sub_device_descriptor_offsets;
    for (const auto& [sub_device_id, descriptor] : traceDesc.descriptors) {
        auto descriptor_offset = tt::target::lightmetal::CreateDescriptor(
            builder,
            descriptor.num_completion_worker_cores,
            descriptor.num_traced_programs_needing_go_signal_multicast,
            descriptor.num_traced_programs_needing_go_signal_unicast);
        auto mapping_offset = tt::target::lightmetal::CreateSubDeviceDescriptorMapping(
            builder,
            sub_device_id.to_index(),  // No need for static_cast; directly use uint8_t
            descriptor_offset);
        sub_device_descriptor_offsets.push_back(mapping_offset);
    }
    auto sub_device_descriptors_offset = builder.CreateVector(sub_device_descriptor_offsets);

    // Serialize the sub_device_ids vector
    std::vector<uint8_t> sub_device_ids_converted;
    sub_device_ids_converted.reserve(traceDesc.sub_device_ids.size());
    for (const auto& sub_device_id : traceDesc.sub_device_ids) {
        sub_device_ids_converted.push_back(sub_device_id.to_index());
    }
    auto sub_device_ids_offset = builder.CreateVector(sub_device_ids_converted);

    // Create the TraceDescriptor
    auto trace_descriptor_offset = tt::target::lightmetal::CreateTraceDescriptor(
        builder, trace_data_offset, sub_device_descriptors_offset, sub_device_ids_offset);

    // Create the TraceDescriptorByTraceId
    return tt::target::lightmetal::CreateTraceDescriptorByTraceId(builder, trace_id, trace_descriptor_offset);
}

}  // namespace v0
}  // namespace tt::tt_metal
