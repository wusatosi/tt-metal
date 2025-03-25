import pandas as pd
import click
import datetime
import os

# Define constants for column names
OP_CODE = "OP CODE"
DEVICE_KERNEL_DURATION = "DEVICE KERNEL DURATION [ns]"
PM_IDEAL = "PM IDEAL [ns]"
PM_COMPUTE = "PM COMPUTE [ns]"
PM_BANDWIDTH = "PM BANDWIDTH [ns]"
MATMUL_UTIL = "MATMUL UTIL"
MATMUL_DURATION = "MATMUL DURATION [ns]"
CONV_UTIL = "CONV UTIL"
CONV_DURATION = "CONV DURATION [ns]"

CONV2D_BEGIN = ["Conv2D begin", "Conv2D begin Transpose"]
CONV2D_END = ["Conv2D end", "Conv2D end Transpose"]

# Define constant for columns to keep
COLUMNS_TO_KEEP = [
    OP_CODE,
    "ATTRIBUTES",
    "MATH FIDELITY",
    "CORE COUNT",
    PM_IDEAL,
    PM_COMPUTE,
    PM_BANDWIDTH,
    DEVICE_KERNEL_DURATION,
    MATMUL_DURATION,
    MATMUL_UTIL,
    CONV_DURATION,
    CONV_UTIL,
    "INPUT_0_W",
    "INPUT_0_Z",
    "INPUT_0_Y",
    "INPUT_0_X",
    "INPUT_1_W",
    "INPUT_1_Z",
    "INPUT_1_Y",
    "INPUT_1_X",
    "OUTPUT_0_W",
    "OUTPUT_0_Z",
    "OUTPUT_0_Y",
    "OUTPUT_0_X",
]


class Conv2DMatmulProfiler:
    def __init__(self, input_csv, output_csv=None):
        """
        Initializes an instance of the Conv2DMatmulProfiler class.

        :param input_csv: Path to the input CSV file.
        :param output_csv: Path to the output CSV file.
        """
        self.input_csv = input_csv
        self.output_csv = output_csv or self.generate_output_filename()
        self.df = self.read_csv(self.input_csv)
        self.total_matmul_duration = 0
        self.total_conv_duration = 0
        self.total_matmul_ideal_duration = 0
        self.total_conv_ideal_duration = 0

    def generate_output_filename(self):
        """
        Generates the output file name with the current timestamp.

        :return: Output file name.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"output_file_{timestamp}.csv"

    def read_csv(self, file_path):
        """
        Reads a CSV file and returns a DataFrame.

        :param file_path: Path to the CSV file.
        :return: DataFrame with data from the CSV file.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise

    def write_csv(self, df, file_path):
        """
        Writes a DataFrame to a CSV file.

        :param df: DataFrame to write.
        :param file_path: Path to the output CSV file.
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"CSV file has been written to: {file_path}")
        except Exception as e:
            raise ValueError(f"Error writing CSV file: {e}")

    def process_conv2d_block(self, total_duration, total_ideal_duration, matmul_flag):
        """
        Processes a Conv2D block.

        :param total_duration: Total duration.
        :param total_ideal_duration: Total ideal duration.
        :param matmul_flag: Flag indicating whether it is a Matmul operation.
        """
        if matmul_flag:
            self.total_matmul_duration += total_duration
            self.total_matmul_ideal_duration += total_ideal_duration
        else:
            self.total_conv_duration += total_duration
            self.total_conv_ideal_duration += total_ideal_duration

    def calculate_conv2d_duration(self):
        """
        Calculates the duration of Conv2D operations.

        :return: List with total ideal duration of Matmul and Conv2D operations.
        """
        # Initialize variables
        inside_conv2d_block = False
        total_duration = 0
        total_ideal_duration = 0
        matmul_flag = False

        # Iterate through DataFrame rows
        for index, row in self.df.iterrows():
            op_code = row[OP_CODE]
            duration = row[DEVICE_KERNEL_DURATION]
            ideal_duration = row[PM_IDEAL]
            bandwith = row[PM_BANDWIDTH]
            compute = row[PM_COMPUTE]

            if op_code in CONV2D_BEGIN:
                inside_conv2d_block = True
            elif op_code in CONV2D_END:
                inside_conv2d_block = False
                self.process_conv2d_block(total_duration, total_ideal_duration, matmul_flag)
                total_duration = 0
                total_ideal_duration = 0
                matmul_flag = False
            elif inside_conv2d_block:
                if op_code == "Matmul":
                    matmul_flag = True
                total_duration += duration
                total_ideal_duration += max(ideal_duration, bandwith, compute)

        # Print total duration
        print(f"Total convolution duration: {self.total_conv_duration} ns")
        print(f"Total matmul duration: {self.total_matmul_duration} ns")
        print(f"Total ideal convolution duration: {self.total_conv_ideal_duration} ns")
        print(f"Total ideal matmul duration: {self.total_matmul_ideal_duration} ns")

        return [self.total_matmul_ideal_duration, self.total_conv_ideal_duration]

    def process_conv2d_util_block(self, index, conv2d_duration_sum, conv2d_ideal_sum, matmul_flag):
        """
        Processes a Conv2D utility block.

        :param index: Index of the current row.
        :param conv2d_duration_sum: Total Conv2D duration.
        :param conv2d_ideal_sum: Total ideal Conv2D duration.
        :param matmul_flag: Flag indicating whether it is a Matmul operation.
        """
        if matmul_flag:
            matmul_util = conv2d_ideal_sum / conv2d_duration_sum if conv2d_duration_sum != 0 else None
            self.df.at[index, MATMUL_UTIL] = matmul_util
            self.df.at[index, MATMUL_DURATION] = conv2d_duration_sum
        else:
            conv_util = conv2d_ideal_sum / conv2d_duration_sum if conv2d_duration_sum != 0 else None
            self.df.at[index, CONV_UTIL] = conv_util
            self.df.at[index, CONV_DURATION] = conv2d_duration_sum

    def add_new_colums(self, columns_to_keep):
        """
        Adds new columns to the DataFrame.

        :param columns_to_keep: List of columns to keep.
        """
        for column in columns_to_keep:
            if column not in self.df.columns:
                self.df[column] = None

    def add_conv_param_columns(self):
        """
        Adds columns for Conv2D parameters.
        """
        # Initialize variables
        inside_conv2d_block = False
        conv2d_duration_sum = 0
        conv2d_ideal_sum = 0
        matmul_flag = False

        # Iterate through DataFrame rows
        for index, row in self.df.iterrows():
            op_code = row[OP_CODE]
            duration = row[DEVICE_KERNEL_DURATION]
            ideal_duration = row[PM_IDEAL]
            bandwith = row[PM_BANDWIDTH]
            compute = row[PM_COMPUTE]

            if op_code in CONV2D_BEGIN:
                inside_conv2d_block = True
                conv2d_duration_sum = 0
                conv2d_ideal_sum = 0
                matmul_flag = False
            elif op_code in CONV2D_END:
                inside_conv2d_block = False
                self.process_conv2d_util_block(index, conv2d_duration_sum, conv2d_ideal_sum, matmul_flag)
            elif inside_conv2d_block:
                if op_code == "Matmul":
                    matmul_flag = True
                conv2d_duration_sum += duration
                conv2d_ideal_sum += max(ideal_duration, bandwith, compute)

    def reduce_csv_columns(self, columns_to_keep):
        """
        Keeps only the specified columns in the DataFrame.

        :param columns_to_keep: List of columns to keep.
        """
        self.df = self.df[columns_to_keep]

    def add_matmul_param_columns(self):
        """
        Adds columns for Matmul parameters.
        """
        # Initialize variables
        inside_conv2d_block = False

        # Iterate through DataFrame rows
        for index, row in self.df.iterrows():
            op_code = row[OP_CODE]

            if op_code == "Conv2D begin":
                inside_conv2d_block = True
            elif op_code == "Conv2D end":
                inside_conv2d_block = False
            elif op_code == "Matmul" and not inside_conv2d_block:
                matmul_util = row[PM_IDEAL] / row[DEVICE_KERNEL_DURATION] if row[DEVICE_KERNEL_DURATION] != 0 else None
                self.df.at[index, MATMUL_UTIL] = matmul_util
                self.df.at[index, MATMUL_DURATION] = row[DEVICE_KERNEL_DURATION]
                self.total_matmul_duration += row[DEVICE_KERNEL_DURATION]
                self.total_matmul_ideal_duration += max(row[PM_IDEAL], row[PM_BANDWIDTH], row[PM_COMPUTE])

    def calculate_actual_device_perf(self, batch):
        """
        Calculates the actual device performance.

        :param batch: Batch size.
        """
        # Calculate total execution duration
        total_duration = self.df[DEVICE_KERNEL_DURATION].sum()

        # Print total duration
        print(f"Total execution duration: {total_duration} ns")
        if total_duration == 0:
            print("Total duration is 0. Cannot calculate device performance.")
            return
        
        print(f"Current device perf: {batch / ((total_duration)/1_000_000_000)} samples/s")

    def calculate_cc_cs(self, batch, matmul_ideal_execution_time, conv_ideal_execution_time, e2e_ratio):
        """
        Calculates the CS and CC models.

        :param batch: Batch size.
        :param matmul_ideal_execution_time: Ideal duration of Matmul operations.
        :param conv_ideal_execution_time: Ideal duration of Conv2D operations.
        :param e2e_ratio: End-to-end ratio.
        """
        if(matmul_ideal_execution_time == 0 and conv_ideal_execution_time == 0):
            print("Matmul or Conv2D ideal execution time is 0. Cannot calculate CS and CC models. Please check if Conv2D begin/end are added in files.")
            return
        
        cs_model = (
            (batch / ((matmul_ideal_execution_time / 0.4 + conv_ideal_execution_time / 0.3) / 1_000_000_000))
            * e2e_ratio
            / 2
        )
        cc_model = cs_model / 2
        print(f"CS model: {cs_model:.2f} [samples/s]")
        print(f"CC model: {cc_model:.2f} [samples/s]")


@click.command()
@click.option("-i", "--input_csv", type=str, required=True, help="Input CSV file path, output of the Tracy profiler")
@click.option("-o", "--output_csv", type=str, default=None, help="Output CSV file path")
@click.option("-b", "--batch", type=int, default=1, help="Batch size")
def main(input_csv, output_csv, batch):
    """
    Main function that runs the profiling of Conv2D and Matmul operations.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file.
    """
    profiler = Conv2DMatmulProfiler(input_csv, output_csv)

    profiler.add_new_colums(COLUMNS_TO_KEEP)
    profiler.add_matmul_param_columns()
    profiler.add_conv_param_columns()
    profiler.reduce_csv_columns(COLUMNS_TO_KEEP)
    profiler.write_csv(profiler.df, profiler.output_csv)

    # Calculate Conv2D operation duration
    ttmi, ttci = profiler.calculate_conv2d_duration()

    # Calculate actual device performance
    profiler.calculate_actual_device_perf(batch)

    # Calculate CC and CS models
    profiler.calculate_cc_cs(batch, ttmi, ttci, 0.85)

    return 0

if __name__ == "__main__":
    
    print("Please note that this script requires adjustments to function properly.")
    print("You must insert blocks with specific labels (as shown in the files within this commit):")
    print("<Conv2D begin> and <Conv2D end> in the file /tt-metal/ttnn/ttnn/operations/conv2d.py::conv2d")
    print("<Conv2D begin Transpose> and <Conv2D end Transpose> in the file /tt-metal/ttnn/ttnn/operations/transpose_conv2d.py::conv_transpose2d")

    main()
    