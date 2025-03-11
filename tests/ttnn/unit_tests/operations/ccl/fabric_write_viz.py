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
    Visualize coordinates with dot sizes scaled according to average wait durations for each PCIe slot.
    Coordinate system starts at (0,0) in the top left corner.

    Parameters:
    -----------
    coordinates : dict
        Dictionary mapping node IDs to (x, y) coordinates
    wait_duration_df : pandas.DataFrame
        DataFrame containing the PCIe slot and wait duration data
    run_id : str or int
        Run ID for the title
    zone_name : str
        Zone name for the title
    risc_processor_type : str
        RISC processor type for the title
    filename : str
        Filename to save the plot
    """
    # Calculate average wait durations by PCIe slot
    avg_durations = wait_duration_df.groupby("PCIe slot")[f"{zone_name} Duration (us)"].mean().to_dict()

    # Create a mapping from node IDs to PCIe slots (assuming node ID = PCIe slot)
    # Filter to include only node IDs that have wait duration data
    valid_nodes = {node_id: coords for node_id, coords in coordinates.items() if node_id in avg_durations}

    if not valid_nodes:
        print("No matching nodes found with wait duration data")
        return

    # Extract x and y values
    x_values = [coord[0] for coord in valid_nodes.values()]
    y_values = [coord[1] for coord in valid_nodes.values()]

    # Get durations for scaling dot sizes
    durations = [avg_durations[node_id] for node_id in valid_nodes.keys()]

    # Normalize durations for more prominent dot sizes (min 50, max 800)
    if max(durations) > min(durations):
        sizes = [50 + 750 * (d - min(durations)) / (max(durations) - min(durations)) for d in durations]
    else:
        sizes = [200 for _ in durations]  # Use uniform size if all durations are equal

    # Create the plot
    plt.figure(figsize=(12, 12))

    # Find the maximum y value to invert the axis
    max_y = max(y_values)

    # Draw a grid
    max_x = max(x_values)
    for i in range(max_x + 2):
        plt.axvline(x=i, color="lightgray", linestyle="-", alpha=0.7)
    for i in range(max_y + 2):
        plt.axhline(y=i, color="lightgray", linestyle="-", alpha=0.7)

    # Plot the nodes with scaled sizes using green to red color map
    scatter = plt.scatter(x_values, y_values, s=sizes, alpha=0.7, c=durations, cmap="RdYlGn_r")

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"Average {zone_name} Duration (us)", fontsize=12)

    # Add labels to the points with black text
    for node_id, (x, y) in valid_nodes.items():
        plt.text(x, y, str(node_id), fontsize=10, ha="center", va="center", color="black", fontweight="bold")

    # Set the axis labels and title
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.title(
        f"Coordinate Visualization with {zone_name} Durations\n"
        f"Run ID: {run_id}, Zone: {zone_name}, Processor: {risc_processor_type}",
        fontsize=14,
    )

    # Ensure the grid is shown properly
    plt.grid(False)
    plt.xticks(range(max_x + 2))
    plt.yticks(range(max_y + 2))

    # Invert the y-axis to have (0,0) at the top left
    plt.gca().invert_yaxis()

    # Set axis limits to start from 0
    plt.xlim(-0.5, max_x + 0.5)
    plt.ylim(max_y + 0.5, -0.5)

    # Make sure the aspect ratio is equal
    plt.axis("equal")

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close()


def analyze_wait_duration(csv_file, run_id, zone_name, risc_processor_type, cores, clock_frequency):
    """
    Analyze and plot wait durations for each PCIe slot for a given run ID, zone name, and RISC processor type.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    run_id : str or int
        The run ID to filter for
    zone_name : str
        The zone name to filter for
    risc_processor_type : str
        The RISC processor type to filter for
    cores : tuple
        (x, y) coordinates of the core to filter for
    clock_frequency : int
        Clock frequency in MHz for converting cycles to microseconds

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the PCIe slot and wait duration data, or None if analysis fails
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file, skiprows=1)
        df = df.rename(columns=lambda x: x.strip())  # Remove leading/trailing spaces from column names
        print(f"Successfully loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Check if all required columns exist
    required_columns = ["PCIe slot", "time[cycles since reset]", "run ID", "zone name", "type", "RISC processor type"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Filter data for the specified run ID, zone name, and RISC processor type
    filtered_df = df[
        (df["run ID"] == run_id)
        & (df["zone name"] == zone_name)
        & (df["RISC processor type"] == risc_processor_type)
        & (df["core_x"] == cores[0])
        & (df["core_y"] == cores[1])
    ]

    if len(filtered_df) == 0:
        print(
            f"No data found for run ID '{run_id}', zone name '{zone_name}', and RISC processor type '{risc_processor_type}'"
        )
        print(f"Available run IDs: {df['run ID'].unique().tolist()}")
        print(f"Available zone names: {df['zone name'].unique().tolist()}")
        print(f"Available RISC processor types: {df['RISC processor type'].unique().tolist()}")
        return None

    # Group by PCIe slot and calculate wait durations
    results = []
    for pcie_slot, group in filtered_df.groupby("PCIe slot"):
        # Get start and end times
        starts = group[group["type"] == "ZONE_START"]["time[cycles since reset]"].reset_index(drop=True)
        ends = group[group["type"] == "ZONE_END"]["time[cycles since reset]"].reset_index(drop=True)

        # Check if we have matching pairs
        min_length = min(len(starts), len(ends))
        if min_length == 0:
            print(f"PCIe slot {pcie_slot} has no matching pairs of ZONE_START and ZONE_END")
            continue

        # Calculate wait durations
        durations = (
            ends[:min_length].values - starts[:min_length].values
        ) / clock_frequency  # Convert cycles to microseconds

        # Add to results
        for duration in durations:
            results.append({"PCIe slot": pcie_slot, f"{zone_name} Duration (us)": duration})

    if not results:
        print("No valid wait durations could be calculated")
        return None

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    # Print summary statistics
    print("\nSummary statistics for wait durations by PCIe slot:")
    summary = result_df.groupby("PCIe slot")[f"{zone_name} Duration (us)"].agg(["count", "mean", "std", "min", "max"])
    print(summary)

    # Create visualizations
    plot_wait_durations(result_df, run_id, zone_name, risc_processor_type)

    # Return the DataFrame for further analysis
    return result_df


def plot_wait_durations(df, run_id, zone_name, risc_processor_type):
    """
    Create visualizations for the wait duration data.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the PCIe slot and wait duration data
    run_id : str or int
        Run ID used for the title
    zone_name : str
        Zone name used for the title
    risc_processor_type : str
        RISC processor type used for the title
    """
    # Set the style
    sns.set(style="whitegrid")

    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Box plot
    sns.boxplot(x="PCIe slot", y=f"{zone_name} Duration (us)", data=df, ax=ax1)
    ax1.set_title(
        f"{zone_name} Duration Distribution by PCIe Slot\nRun ID: {run_id}, Zone: {zone_name}, Processor: {risc_processor_type}"
    )
    ax1.set_xlabel("PCIe Slot")
    ax1.set_ylabel(f"{zone_name} Duration (us)")

    # 2. Bar plot - Fixed to avoid the yerr shape mismatch error
    summary = df.groupby("PCIe slot")[f"{zone_name} Duration (us)"].agg(["mean", "std"]).reset_index()

    # Create bar plot without error bars first
    bars = ax2.bar(summary["PCIe slot"], summary["mean"])

    ax2.set_title(
        f"Average {zone_name} Duration by PCIe Slot\nRun ID: {run_id}, Zone: {zone_name}, Processor: {risc_processor_type}"
    )
    ax2.set_xlabel("PCIe Slot")
    ax2.set_ylabel(f"Average {zone_name} Duration (us)")

    # Adjust layout and save
    filename_base = f'generated/{zone_name}_duration_runid_{run_id}_zone_{zone_name}_processor_{risc_processor_type.replace(" ", "_")}'
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png")
    print(f"Saved plot to {filename_base}.png")

    # Create a histogram of all wait durations
    plt.figure(figsize=(10, 6))

    # Use factorplot instead of multiple histograms if there are many PCIe slots
    if len(df["PCIe slot"].unique()) > 5:
        sns.histplot(data=df, x=f"{zone_name} Duration (us)", hue="PCIe slot", kde=True, alpha=0.5, common_norm=False)
    else:
        # Plot separate histograms for each PCIe slot
        for pcie_slot, group in df.groupby("PCIe slot"):
            sns.histplot(group[f"{zone_name} Duration (us)"], label=f"PCIe Slot {pcie_slot}", kde=True, alpha=0.6)

    plt.title(
        f"{zone_name} Duration Distribution\nRun ID: {run_id}, Zone: {zone_name}, Processor: {risc_processor_type}"
    )
    plt.xlabel(f"{zone_name} Duration (us)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_base}_histogram.png")
    print(f"Saved histogram to {filename_base}_histogram.png")


# Example usage
if __name__ == "__main__":
    # These parameters would be replaced with actual values
    csv_file = "generated/profiler/.logs/profile_log_device.csv"

    # You would need to specify these based on your data
    run_id = 39 * 2  # Replace with actual run ID
    zone_name = "writer"  # Replace with actual zone name
    risc_processor_type = "BRISC"  # Replace with actual RISC processor type
    cores = (1, 3)
    clock_frequency = 900  # Clock frequency in MHz

    # Analyze wait durations and get the dataframe
    df = analyze_wait_duration(csv_file, run_id, zone_name, risc_processor_type, cores, clock_frequency)

    # Create the new visualization with wait times (without core information)
    if df is not None and not df.empty:
        visualize_coordinates_with_wait_times(coordinates, df, run_id, zone_name, risc_processor_type)
    else:
        print("No data available for wait time visualization")
