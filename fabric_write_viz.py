import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

coordinates = {
    4: (3, 6),
    5: (3, 5),
    6: (2, 5),
    7: (2, 6),
    8: (1, 6),
    9: (1, 7),
    10: (2, 7),
    11: (3, 7),
    12: (0, 7),
    13: (0, 6),
    14: (0, 5),
    15: (1, 5),
    16: (1, 4),
    17: (2, 4),
    18: (3, 4),
    19: (3, 3),
    20: (2, 3),
    21: (1, 3),
    22: (1, 2),
    23: (2, 2),
    24: (3, 2),
    25: (3, 1),
    26: (2, 1),
    27: (1, 1),
    28: (1, 0),
    29: (2, 0),
    30: (3, 0),
    31: (0, 0),
    32: (0, 1),
    33: (0, 2),
    34: (0, 3),
    35: (0, 4),
}


def visualize_coordinates_with_wait_times(
    coordinates,
    wait_duration_df,
    run_id,
    zone_name,
    risc_processor_type,
    filename="generated/zone_duration_coordinates.png",
):
    """
    Visualize coordinates with different color schemes for multiple cores.
    """
    if wait_duration_df is None or wait_duration_df.empty:
        print("No data to visualize.")
        return

    core_groups = wait_duration_df.groupby(["core_x", "core_y"])
    colors = sns.color_palette("husl", len(core_groups))  # Get distinct colors

    plt.figure(figsize=(12, 12))
    max_x, max_y = max([c[0] for c in coordinates.values()]), max([c[1] for c in coordinates.values()])

    for i in range(max_x + 2):
        plt.axvline(x=i, color="lightgray", linestyle="-", alpha=0.7)
    for i in range(max_y + 2):
        plt.axhline(y=i, color="lightgray", linestyle="-", alpha=0.7)

    for (core_x, core_y), group in core_groups:
        core_color = colors.pop(0)  # Assign a unique color
        avg_durations = group.groupby("PCIe slot")[f"{zone_name} Duration (us)"].mean().to_dict()

        valid_nodes = {node_id: coordinates[node_id] for node_id in coordinates if node_id in avg_durations}

        if valid_nodes:
            x_values = [coord[0] for coord in valid_nodes.values()]
            y_values = [coord[1] for coord in valid_nodes.values()]
            durations = [avg_durations[node_id] for node_id in valid_nodes.keys()]
            sizes = [
                50 + 750 * (d - min(durations)) / (max(durations) - min(durations))
                if max(durations) > min(durations)
                else 200
                for d in durations
            ]

            scatter = plt.scatter(
                x_values,
                y_values,
                s=sizes,
                alpha=0.7,
                c=[core_color] * len(x_values),
                label=f"Core ({core_x}, {core_y})",
            )

            for node_id, (x, y) in valid_nodes.items():
                plt.text(x, y, str(node_id), fontsize=10, ha="center", va="center", color="black", fontweight="bold")

    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.title(
        f"Coordinate Visualization with {zone_name} Durations\nRun ID: {run_id}, Processor: {risc_processor_type}",
        fontsize=14,
    )

    plt.legend()
    plt.grid(False)
    plt.xticks(range(max_x + 2))
    plt.yticks(range(max_y + 2))
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close()


def analyze_wait_duration(csv_file, run_id, zone_name, risc_processor_type, cores, clock_frequency):
    """
    Analyze and plot wait durations for each PCIe slot for multiple cores.

    Parameters:
    -----------
    cores : list of tuples
        List of (x, y) coordinates of the cores to filter for.
    """
    try:
        df = pd.read_csv(csv_file, skiprows=1)
        df = df.rename(columns=lambda x: x.strip())  # Remove extra spaces
        print(f"Successfully loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Check for required columns
    required_columns = [
        "PCIe slot",
        "time[cycles since reset]",
        "run ID",
        "zone name",
        "type",
        "RISC processor type",
        "core_x",
        "core_y",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None

    # Filter for multiple cores
    filtered_df = df[
        (df["run ID"] == run_id)
        & (df["zone name"] == zone_name)
        & (df["RISC processor type"] == risc_processor_type)
        & df.apply(lambda row: (row["core_x"], row["core_y"]) in cores, axis=1)
    ]

    if filtered_df.empty:
        print(f"No data found for the specified filters.")
        return None

    results = []
    for (core_x, core_y), core_group in filtered_df.groupby(["core_x", "core_y"]):
        for pcie_slot, group in core_group.groupby("PCIe slot"):
            starts = group[group["type"] == "ZONE_START"]["time[cycles since reset]"].reset_index(drop=True)
            ends = group[group["type"] == "ZONE_END"]["time[cycles since reset]"].reset_index(drop=True)
            min_length = min(len(starts), len(ends))

            if min_length > 0:
                durations = (
                    ends[:min_length].values - starts[:min_length].values
                ) / clock_frequency  # Convert cycles to us
                for duration in durations:
                    results.append(
                        {
                            "PCIe slot": pcie_slot,
                            f"{zone_name} Duration (us)": duration,
                            "core_x": core_x,
                            "core_y": core_y,
                        }
                    )

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        plot_wait_durations(result_df, run_id, zone_name, risc_processor_type)

    return result_df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_wait_durations(df, run_id, zone_name, risc_processor_type):
    """
    Plot wait durations for multiple cores as a grid of subplots.
    """
    sns.set(style="whitegrid")

    unique_cores = df[["core_x", "core_y"]].drop_duplicates().values
    num_cores = len(unique_cores)

    # Determine grid size (rows x cols)
    cols = int(np.ceil(np.sqrt(num_cores)))  # Square-like layout
    rows = int(np.ceil(num_cores / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of 1D array

    for i, (core_x, core_y) in enumerate(unique_cores):
        core_df = df[(df["core_x"] == core_x) & (df["core_y"] == core_y)]

        ax = axes[i]
        sns.boxplot(x="PCIe slot", y=f"{zone_name} Duration (us)", data=core_df, ax=ax)
        ax.set_title(f"Core ({core_x}, {core_y})", fontsize=12)
        ax.set_xlabel("PCIe Slot")
        ax.set_ylabel(f"{zone_name} Duration (us)")

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{zone_name} Duration by Core\nRun ID: {run_id}, Processor: {risc_processor_type}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    filename = f"generated/{zone_name}_durations_run_{run_id}.png"
    plt.savefig(filename)
    print(f"Saved grid plot to {filename}")
    plt.close()


# Example usage
if __name__ == "__main__":
    # These parameters would be replaced with actual values
    csv_file = "generated/profiler/.logs/profile_log_device.csv"

    # You would need to specify these based on your data
    run_id = 30 * 2  # Replace with actual run ID
    zone_name = "MAIN-WRITE-LOOP"  # Replace with actual zone name
    risc_processor_type = "BRISC"  # Replace with actual RISC processor type
    cores = [(21, 18), (21, 19), (21, 20), (21, 21)]
    # cores = [(23, 23), (23, 24), (24, 23)]
    clock_frequency = 900  # Clock frequency in MHz

    # Analyze wait durations and get the dataframe
    df = analyze_wait_duration(csv_file, run_id, zone_name, risc_processor_type, cores, clock_frequency)

    # Create the new visualization with wait times (without core information)
    if df is not None and not df.empty:
        visualize_coordinates_with_wait_times(coordinates, df, run_id, zone_name, risc_processor_type)
    else:
        print("No data available for wait time visualization")
