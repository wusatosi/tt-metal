import json
import os


def main():
    # Get the absolute path of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "transpose_forge.json")

    # Read the original JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    processed = []

    for entry in data:
        # Parse the shape from input_shapes
        input_shape_str = entry["input_shapes"][0]
        # Remove the prefix 'tensor<[' and suffix ']>'
        shape_content = input_shape_str.replace("tensor<[", "").replace("]>", "")
        shape_parts = shape_content.split(",")

        # The last part is the data type, remove it
        shape_dims = shape_parts[:-1]
        shape = list(map(int, shape_dims))

        # Parse the attributes for dim0, dim1
        dim0_str = entry["attributes"]["dim0"]
        dim1_str = entry["attributes"]["dim1"]

        # Extract the integer values
        dim0 = int(dim0_str.split(":")[0].strip())
        dim1 = int(dim1_str.split(":")[0].strip())

        processed.append({"shape": shape, "dim0": dim0, "dim1": dim1})

    # Write the processed data to a new JSON file with minimal whitespace
    output_path = os.path.join(base_dir, "transpose_forge_processed.json")
    with open(output_path, "w") as f:
        json.dump(processed, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
