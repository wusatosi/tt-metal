import pandas as pd
import click
import datetime


def calculate_conv2d_duration(csv_file):
    # Učitajte CSV datoteku
    df = pd.read_csv(csv_file)

    # Inicijalizirajte varijable
    inside_conv2d_block = False
    total_duration = 0
    total_matmul_duration = 0
    total_conv_duration = 0
    total_ideal_duration = 0
    matmul_flag = False
    total_matmul_ideal_duration = 0
    total_conv_ideal_duration = 0
    # Iterirajte kroz redove DataFrame-a
    for index, row in df.iterrows():
        op_code = row["OP CODE"]
        duration = row["DEVICE KERNEL DURATION [ns]"]
        ideal_duration = row["PM IDEAL [ns]"]
        bandwith = row["PM BANDWIDTH [ns]"]
        compute = row["PM COMPUTE [ns]"]

        if op_code == "Conv2D begin":
            inside_conv2d_block = True
        elif op_code == "Conv2D end":
            inside_conv2d_block = False
            if matmul_flag:
                total_matmul_duration += total_duration
                total_matmul_ideal_duration += total_ideal_duration
            else:
                total_conv_duration += total_duration
                total_conv_ideal_duration += total_ideal_duration
            total_duration = 0
            total_ideal_duration = 0
            matmul_flag = False
        elif inside_conv2d_block:
            if op_code == "Matmul":
                matmul_flag = True
            total_duration += duration
            total_ideal_duration += max(ideal_duration, bandwith, compute)

    # Ispišite ukupno trajanje
    print(f"Ukupno trajanje konvolucije: {total_conv_duration} ns")
    print(f"Ukupno trajanje matmula: {total_matmul_duration} ns")
    print(f"Idealno ukupno trajanje konvolucije: {total_conv_ideal_duration} ns")
    print(f"Idealno ukupno trajanje matmula: {total_matmul_ideal_duration} ns")

    return [total_matmul_ideal_duration, total_conv_ideal_duration]


def reduce_csv_columns(input_csv, output_csv, columns_to_keep):
    # Učitajte ulazni CSV fajl
    df = pd.read_csv(input_csv)

    # Zadržite samo specificirane kolone
    reduced_df = df[columns_to_keep]

    # Sačuvajte novi CSV fajl sa redukovanim brojem kolona
    reduced_df.to_csv(output_csv, index=False)

    # Primer poziva funkcije
    # reduce_csv_columns('input.csv', 'output.csv', ['OP CODE', 'DEVICE KERNEL DURATION [ns]'])


def add_matmul_util_column(csv_file, output_csv):
    # Učitajte CSV datoteku
    df = pd.read_csv(csv_file)

    # Inicijalizirajte varijable
    inside_conv2d_block = False

    # Iterirajte kroz redove DataFrame-a
    for index, row in df.iterrows():
        op_code = row["OP CODE"]

        if op_code == "Conv2D begin":
            inside_conv2d_block = True
        elif op_code == "Conv2D end":
            inside_conv2d_block = False
        elif op_code == "Matmul" and not inside_conv2d_block:
            matmul_util = (
                row["PM IDEAL [ns]"] / row["DEVICE KERNEL DURATION [ns]"]
                if row["DEVICE KERNEL DURATION [ns]"] != 0
                else None
            )
            df.at[index, "MATMUL UTIL"] = matmul_util
            df.at[index, "MATMUL DURATION [ns]"] = row["DEVICE KERNEL DURATION [ns]"]

    # Sačuvajte novi CSV fajl sa dodatom kolonom
    df.to_csv(output_csv, index=False)


def add_conv_util_column(csv_file, output_csv):
    # Učitajte CSV datoteku
    df = pd.read_csv(csv_file)

    # Inicijalizirajte varijable
    inside_conv2d_block = False
    conv2d_duration_sum = 0
    conv2d_ideal_sum = 0

    # Iterirajte kroz redove DataFrame-a
    for index, row in df.iterrows():
        op_code = row["OP CODE"]
        duration = row["DEVICE KERNEL DURATION [ns]"]
        ideal_duration = row["PM IDEAL [ns]"]
        bandwith = row["PM BANDWIDTH [ns]"]
        compute = row["PM COMPUTE [ns]"]

        if op_code == "Conv2D begin":
            inside_conv2d_block = True
            conv2d_duration_sum = 0
            conv2d_ideal_sum = 0
            matmul_flag = False
        elif op_code == "Conv2D end":
            inside_conv2d_block = False
            if matmul_flag:
                matmul_flag = False
                matmul_util = conv2d_ideal_sum / conv2d_duration_sum if conv2d_duration_sum != 0 else None
                df.at[index, "MATMUL UTIL"] = matmul_util
                df.at[index, "MATMUL DURATION [ns]"] = conv2d_duration_sum
            else:
                conv_util = conv2d_ideal_sum / conv2d_duration_sum if conv2d_duration_sum != 0 else None
                df.at[index, "CONV UTIL"] = conv_util
                df.at[index, "CONV DURATION [ns]"] = conv2d_duration_sum
        elif inside_conv2d_block:
            if op_code == "Matmul":
                matmul_flag = True
            conv2d_duration_sum += duration
            conv2d_ideal_sum += max(ideal_duration, bandwith, compute)

    # Sačuvajte novi CSV fajl sa dodatom kolonom
    df.to_csv(output_csv, index=False)


def calculate_actual_device_perf(batch, csv_file):
    # Učitajte CSV datoteku
    df = pd.read_csv(csv_file)

    # Izračunajte ukupno trajanje izvršavanja
    total_duration = df["DEVICE KERNEL DURATION [ns]"].sum()

    # Ispišite ukupno trajanje
    print(f"Ukupno trajanje izvršavanja: {total_duration} ns")
    print(f"Trenutni device perf: {batch / (total_duration/1_000_000_000)} samples/s")


def calculate_cc_cs(batch, matmul_ideal_execution_time, conv_ideal_execution_time, e2e_ratio):
    cs_model = (
        (batch / ((matmul_ideal_execution_time / 0.4 + conv_ideal_execution_time / 0.3) / 1_000_000_000))
        * e2e_ratio
        / 2
    )
    cc_model = cs_model / 2
    print(f"CS model: {cs_model:.2f} [samples/s]")
    print(f"CC model: {cc_model:.2f} [samples/s]")


@click.command()
@click.option("-i", "--input_csv", type=str, required=True, help="Input CSV file path")
@click.option("-o", "--output_csv", type=str, default=None, help="Output CSV file path")
def main(input_csv, output_csv):
    # Zadržite samo specificirane kolone
    columns_to_keep = [
        "OP CODE",
        "ATTRIBUTES",
        "MATH FIDELITY",
        "CORE COUNT",
        # "PARALLELIZATION STRATEGY",
        # "HOST START TS",
        # "HOST END TS",
        # "HOST DURATION [ns]",
        # "DEVICE FW START CYCLE",
        # "DEVICE FW END CYCLE",
        # "OP TO OP LATENCY [ns]",
        # "DEVICE FW DURATION [ns]",
        "PM IDEAL [ns]",
        "PM COMPUTE [ns]",
        "PM BANDWIDTH [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "MATMUL DURATION [ns]",
        "MATMUL UTIL",
        "CONV DURATION [ns]",
        "CONV UTIL",
        # "DEVICE KERNEL DURATION PER CORE MIN [ns]",
        # "DEVICE KERNEL DURATION PER CORE MAX [ns]",
        # "DEVICE KERNEL DURATION PER CORE AVG [ns]",
        # "DEVICE KERNEL FIRST TO LAST START [ns]",
        # "DEVICE BRISC KERNEL DURATION [ns]",
        # "DEVICE NCRISC KERNEL DURATION [ns]",
        # "DEVICE TRISC0 KERNEL DURATION [ns]",
        # "DEVICE TRISC1 KERNEL DURATION [ns]",
        # "DEVICE TRISC2 KERNEL DURATION [ns]",
        # "DEVICE ERISC KERNEL DURATION [ns]",
        # "DEVICE COMPUTE CB WAIT FRONT [ns]",
        # "DEVICE COMPUTE CB RESERVE BACK [ns]",
        "INPUT_0_W",
        "INPUT_0_Z",
        "INPUT_0_Y",
        "INPUT_0_X",
        # "INPUT_0_LAYOUT",
        # "INPUT_0_DATATYPE",
        # "INPUT_0_MEMORY",
        "INPUT_1_W",
        "INPUT_1_Z",
        "INPUT_1_Y",
        "INPUT_1_X",
        # "INPUT_1_LAYOUT",
        # "INPUT_1_DATATYPE",
        # "INPUT_1_MEMORY",
        # "INPUT_2_W",
        # "INPUT_2_Z",
        # "INPUT_2_Y",
        # "INPUT_2_X",
        # "INPUT_2_LAYOUT",
        # "INPUT_2_DATATYPE",
        # "INPUT_2_MEMORY",
        # "INPUT_3_W",
        # "INPUT_3_Z",
        # "INPUT_3_Y",
        # "INPUT_3_X",
        # "INPUT_3_LAYOUT",
        # "INPUT_3_DATATYPE",
        # "INPUT_3_MEMORY",
        "OUTPUT_0_W",
        "OUTPUT_0_Z",
        "OUTPUT_0_Y",
        "OUTPUT_0_X",
        # "OUTPUT_0_LAYOUT",
        # "OUTPUT_0_DATATYPE",
        # "OUTPUT_0_MEMORY",
        # "METAL TRACE ID",
        # "METAL TRACE REPLAY SESSION ID",
        # "COMPUTE KERNEL SOURCE",
        # "COMPUTE KERNEL HASH",
        # "DATA MOVEMENT KERNEL SOURCE",
        # "DATA MOVEMENT KERNEL HASH",
        # "BRISC MAX KERNEL SIZE [B]",
        # "NCRISC MAX KERNEL SIZE [B]",
        # "TRISC 0 MAX KERNEL SIZE [B]",
        # "TRISC 1 MAX KERNEL SIZE [B]",
        # "TRISC 2 MAX KERNEL SIZE [B]",
        # "ERISC MAX KERNEL SIZE [B]",
        # "PM REQ I BW",
        # "PM REQ O BW",
        # "CompileProgram_TT_HOST_FUNC [ns]",
        # "HWCommandQueue_write_buffer_TT_HOST_FUNC [ns]"
    ]

    # Ako output_csv nije specificiran, generišite podrazumevani naziv fajla
    if output_csv is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"output_file_{timestamp}.csv"

    # Redukujte kolone u CSV fajlu

    print(f"Output CSV file: {output_csv}")
    add_matmul_util_column(input_csv, output_csv)
    print(f"Output CSV file: {output_csv}")
    add_conv_util_column(output_csv, output_csv)
    print(f"Output CSV file: {output_csv}")
    # Izračunajte trajanje Conv2D operacija
    ttmi, ttci = calculate_conv2d_duration(input_csv)
    # izračunajte actual device perf
    reduce_csv_columns(output_csv, output_csv, columns_to_keep)

    # Dodajte poziv funkcije u main
    batch = 1
    calculate_actual_device_perf(batch, input_csv)

    calculate_cc_cs(batch, ttmi, ttci, 0.85)


if __name__ == "__main__":
    input_csv = "/localdev/bjanjic/tt-metal/generated/profiler/yolov4/reports/2025_03_05_16_28_47/ops_perf_results_2025_03_05_16_28_47.csv"
    output_csv = "mojcsv.csv"
    main(["--input_csv", input_csv, "--output_csv", output_csv])
