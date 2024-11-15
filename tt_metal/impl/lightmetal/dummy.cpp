#include <iostream>
#include <flatbuffers/flatbuffers.h>
#include "binary_generated.h" // Generated header

// Use the namespace from the schema
using namespace tt::target::lightmetal;

flatbuffers::DetachedBuffer constructLightMetalBinary(int id, const std::string& text) {
    // Step 1: Create a FlatBufferBuilder
    flatbuffers::FlatBufferBuilder builder;

    // Step 2: Create the string offset for the `text` field
    auto text_offset = builder.CreateString(text);

    // Step 3: Create the LightMetalBinary object
    auto light_metal_binary = CreateLightMetalBinary(builder, id, text_offset);

    // Step 4: Finalize the buffer
    builder.Finish(light_metal_binary);

    // Return the serialized buffer
    return builder.Release();
}
