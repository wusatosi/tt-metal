import json
import os
import re


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "slice_forge.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    processed = []

    for entry in data:
        # Take the first input shape
        # example: "tensor<[196,196,2,i32]>"
        input_shape_str = entry["input_shapes"][0]
        # remove "tensor<[" and "]>"
        shape_content = input_shape_str.replace("tensor<[", "").replace("]>", "")
        # split by comma
        shape_parts = shape_content.split(",")
        # remove the last element (the datatype)
        shape_dims = shape_parts[:-1]
        # convert each to int
        dims = list(map(int, shape_dims))

        # For begins, ends, step:
        # Example: "[0 : i32, 0 : i32, 0 : i32]"
        # We want to extract just the integers
        def parse_attr_list(attr_str):
            # Remove brackets
            attr_str = attr_str.strip().strip("[]")
            # split by comma
            parts = [p.strip() for p in attr_str.split(",")]
            # each part looks like "0 : i32"
            # split by ':'
            nums = [int(p.split(":")[0].strip()) for p in parts]
            return nums

        begins = parse_attr_list(entry["attributes"]["begins"])
        ends = parse_attr_list(entry["attributes"]["ends"])
        step = parse_attr_list(entry["attributes"]["step"])

        processed.append({"dims": dims, "begins": begins, "ends": ends, "step": step})

    # Write the processed data to a new JSON file with minimal whitespace
    output_path = os.path.join(base_dir, "slice_forge_processed.json")
    with open(output_path, "w") as f:
        json.dump(processed, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
