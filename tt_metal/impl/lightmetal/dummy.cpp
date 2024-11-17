#include <iostream>
#include <flatbuffers/flatbuffers.h>
#include "binary_generated.h" // Generated header


void createAndReadFlatBuffer() {
    using namespace tt::target::lightmetal; // Use the namespace to access generated types

    // Step 1: Create a FlatBufferBuilder
    flatbuffers::FlatBufferBuilder builder;

    // Step 2: Create TraceDescriptor objects
    std::vector<uint32_t> traceData = {0x10, 0x20, 0x30}; // Example data
    auto trace_data_offset = builder.CreateVector(traceData);

    auto desc0 = CreateTraceDescriptorByTraceId(
        builder,
        1, // trace_id
        CreateTraceDescriptor(
            builder,
            trace_data_offset,  // trace_data
            4,                  // num_completion_worker_cores
            2,                  // num_traced_programs_needing_go_signal_multicast
            1                   // num_traced_programs_needing_go_signal_unicast
        ) // desc
    );

    auto desc1 = CreateTraceDescriptorByTraceId(
        builder,
        2, // trace_id
        CreateTraceDescriptor(
            builder,
            trace_data_offset,  // trace_data
            8,                  // num_completion_worker_cores
            5,                  // num_traced_programs_needing_go_signal_multicast
            3                   // num_traced_programs_needing_go_signal_unicast
        ) // desc
    );

    // Step 3: Create a vector of TraceDescriptorByTraceId
    std::vector<flatbuffers::Offset<TraceDescriptorByTraceId>> trace_descriptors_vector = {
        desc0,
        desc1
    };

    auto trace_descriptors_offset = builder.CreateVector(trace_descriptors_vector);

    // Step 4: Create the LightMetalBinary
    auto light_metal_binary = CreateLightMetalBinary(builder, trace_descriptors_offset);

    // Step 5: Finalize the buffer
    builder.Finish(light_metal_binary);

    // Step 6: Get a pointer to the serialized buffer
    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    // Save to file or send over the network as needed.
    // For this example, print the buffer size.
    std::cout << "Serialized FlatBuffer size: " << size << " bytes" << std::endl;

    // Deserialize and read back
    auto light_metal_binary_read = GetLightMetalBinary(buf);
    auto trace_descriptors = light_metal_binary_read->trace_descriptors();

    for (const auto* trace_descriptor_by_trace_id : *trace_descriptors) {
        // Access trace_id from TraceDescriptorByTraceId
        uint32_t trace_id = trace_descriptor_by_trace_id->trace_id();
        const auto* trace_descriptor = trace_descriptor_by_trace_id->desc();

        std::cout << "Trace ID: " << trace_id << std::endl;
        std::cout << "Completion Worker Cores: " << trace_descriptor->num_completion_worker_cores() << std::endl;
        std::cout << "Programs Needing Go Signal (Multicast): "
                  << trace_descriptor->num_traced_programs_needing_go_signal_multicast() << std::endl;
        std::cout << "Programs Needing Go Signal (Unicast): "
                  << trace_descriptor->num_traced_programs_needing_go_signal_unicast() << std::endl;

        std::cout << "Trace Data: ";
        for (auto data : *trace_descriptor->trace_data()) {
            std::cout << static_cast<int>(data) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    createAndReadFlatBuffer();
    return 0;
}
