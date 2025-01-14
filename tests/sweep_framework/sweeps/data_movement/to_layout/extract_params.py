import json
import re


def replace_nan_in_dict(d):
    if isinstance(d, dict):
        return {k: replace_nan_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nan_in_dict(item) for item in d]
    elif d == "NaN":
        return float("nan")  # Replace the string "NaN" with a float NaN
    else:
        return d


def parse_input(data):
    result = []
    count = 0
    # Check if 'data' is a list
    if isinstance(data, list):
        for item in data:
            # Ensure each item is a dictionary and has the key 'input_shapes'
            if isinstance(item, dict) and "input_shapes" in item:
                for shape_str in item["input_shapes"]:
                    # Regular expression to capture the 4D shape and the dtype
                    match = re.match(r"tensor<\[(\d+),(\d+),(\d+),(\d+),(\w+)\]>", shape_str)
                    if match:
                        # Extract shape dimensions and dtype
                        shape = tuple(map(int, match.groups()[:4]))  # Get the first four dimensions as a tuple
                        dtype = match.group(5)  # The dtype is the 5th captured group (f32, f64, etc.)

                        # Extract layout type from the 'attributes' dictionary
                        layout = item["attributes"].get("layout", "")
                        layout_match = re.search(r"<(.*?)>", layout)
                        if layout_match:
                            layout_type = layout_match.group(1)  # Extract layout type (e.g., 'tile')
                        else:
                            layout_type = None  # If no layout type is found

                        # Append the tuple (input_shape, input_dtype, input_layout)
                        result.append((shape, dtype, layout_type))
                    else:
                        match = re.match(r"tensor<\[(\d+),(\d+),(\d+),(\w+)\]>", shape_str)
                        if match:
                            # Extract shape dimensions and dtype
                            shape = tuple(map(int, match.groups()[:3]))  # Get the first four dimensions as a tuple
                            dtype = match.group(4)  # The dtype is the 5th captured group (f32, f64, etc.)

                            # Extract layout type from the 'attributes' dictionary
                            layout = item["attributes"].get("layout", "")
                            layout_match = re.search(r"<(.*?)>", layout)
                            if layout_match:
                                layout_type = layout_match.group(1)  # Extract layout type (e.g., 'tile')
                            else:
                                layout_type = None  # If no layout type is found

                            # Append the tuple (input_shape, input_dtype, input_layout)
                            result.append((shape, dtype, layout_type))

                        else:
                            match = re.match(r"tensor<\[(\d+),(\d+),(\w+)\]>", shape_str)
                            if match:
                                # Extract shape dimensions and dtype
                                shape = tuple(map(int, match.groups()[:2]))  # Get the first four dimensions as a tuple
                                dtype = match.group(3)  # The dtype is the 5th captured group (f32, f64, etc.)

                                # Extract layout type from the 'attributes' dictionary
                                layout = item["attributes"].get("layout", "")
                                layout_match = re.search(r"<(.*?)>", layout)
                                if layout_match:
                                    layout_type = layout_match.group(1)  # Extract layout type (e.g., 'tile')
                                else:
                                    layout_type = None  # If no layout type is found

                                # Append the tuple (input_shape, input_dtype, input_layout)
                                result.append((shape, dtype, layout_type))

                            else:
                                match = re.match(r"tensor<\[(\d+),(\w+)\]>", shape_str)
                                if match:
                                    # Extract shape dimensions and dtype
                                    shape = tuple(
                                        map(int, match.groups()[:1])
                                    )  # Get the first four dimensions as a tuple
                                    dtype = match.group(2)  # The dtype is the 5th captured group (f32, f64, etc.)

                                    # Extract layout type from the 'attributes' dictionary
                                    layout = item["attributes"].get("layout", "")
                                    layout_match = re.search(r"<(.*?)>", layout)
                                    if layout_match:
                                        layout_type = layout_match.group(1)  # Extract layout type (e.g., 'tile')
                                    else:
                                        layout_type = None  # If no layout type is found

                                    # Append the tuple (input_shape, input_dtype, input_layout)
                                    result.append((shape, dtype, layout_type))
                                else:
                                    count = count + 1
                                    print(count, "did not match")

    return result


def parse_input_2(data):
    result = []

    # Check if 'data' is a list
    if isinstance(data, list):
        for item in data:
            # Ensure each item is a dictionary and has the key 'input_shapes'
            if isinstance(item, dict) and "input_shapes" in item:
                for shape_str in item["input_shapes"]:
                    # Regular expression to capture the shape (dimensions) and dtype
                    # This captures a tensor of any dimensions, e.g., tensor<[1, 12, 10, 10, f32]>
                    match = re.match(r"tensor<\[(.*?)\],(\w+)>", shape_str)
                    if match:
                        # Extract shape dimensions as a list of integers
                        shape_str = match.group(1)  # Get the part with the dimensions
                        shape = tuple(map(int, shape_str.split(",")))  # Convert dimensions to integers
                        dtype = match.group(2)  # The dtype is the second captured group (f32, f64, etc.)

                        # Extract layout type from the 'attributes' dictionary
                        layout = item["attributes"].get("layout", "")
                        layout_match = re.search(r"<(.*?)>", layout)
                        if layout_match:
                            layout_type = layout_match.group(1)  # Extract layout type (e.g., 'tile')
                        else:
                            layout_type = None  # If no layout type is found

                        # Append the tuple (input_shape, input_dtype, input_layout)
                        result.append((shape, dtype, layout_type))

    return result


def write_parameters_to_json(parameter_list, filename="parameters.json"):
    with open(filename, "w") as f:
        json.dump(parameter_list, f, indent=2)


def write_parameters_to_file(parameter_list, filename="parameters.txt"):
    with open(filename, "w") as f:
        f.write("[\n")  # Start of list
        for param in parameter_list:
            # Format each parameter and write it to the file
            shape, dtype, layout = param
            f.write(f" ({shape}, {dtype}, {layout}),\n")
        f.write("]")  # End of list


# Load the original JSON file
def load_and_process_json(input_filename, output_filename="parameters.json"):
    with open(input_filename, "r") as f:
        data = json.load(f)

    # Replace "NaN" with actual NaN values
    data = replace_nan_in_dict(data)

    # Extract the parameter list from the processed data
    parameter_list = parse_input(data)

    # Write the result to a JSON file
    write_parameters_to_file(parameter_list, output_filename)


# Call the function to process and save to a new file
load_and_process_json(
    "/home/ubuntu/work/sweeps/tt-metal/tests/sweep_framework/sweeps/data_movement/to_layout/ttnn.to_layout.json",
    "/home/ubuntu/work/sweeps/tt-metal/tests/sweep_framework/sweeps/data_movement/to_layout/params_2.txt",
)
