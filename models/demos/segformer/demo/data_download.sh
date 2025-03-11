#!/bin/bash

# Check if evaluate is installed, install it if not
if ! python -c "import evaluate" &> /dev/null; then
    echo "'evaluate' library not found, installing..."
    pip install evaluate
else
    echo "'evaluate' library is already installed."
fi

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "wget is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y wget
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    echo "unzip is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y unzip
fi

# Define the OneDrive zip file direct download link
ONEDRIVE_ZIP_LINK="https://tinyurl.com/4xtuxr3k"
# "https://multicorewareinc1-my.sharepoint.com/:u:/g/personal/venkatesh_guduru_multicorewareinc_com/EQD-v8YWNt1NpI5Az9SUbeQBgmehxW5EE_bjCWB8GXRP5Q?e=brm265&download=1"

# Define the output path for saving the zip file
OUTPUT_FILE="models/demos/segformer/demo/validation_data.zip"

# Create the directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Download the zip file using wget
echo "Downloading file from OneDrive..."
wget --trust-server-names --content-disposition -O "$OUTPUT_FILE" "$ONEDRIVE_ZIP_LINK"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download complete: $OUTPUT_FILE"
else
    echo "Download failed!"
    exit 1
fi

# Verify if the downloaded file is a valid zip file
echo "Verifying the downloaded file..."
file "$OUTPUT_FILE"

# Unzip the downloaded zip file into the target directory
echo "Unzipping the downloaded file..."
unzip "$OUTPUT_FILE" -d "models/demos/segformer/demo/"

# Check if unzip was successful
if [ $? -eq 0 ]; then
    echo "Unzip complete: Files extracted to models/demos/segformer/demo/"
else
    echo "Unzip failed!"
    exit 1
fi

# Clean up any __MACOSX and ._* files (MacOS metadata)
echo "Cleaning up __MACOSX and ._* files..."
find "models/demos/segformer/demo/" -type d -name "__MACOSX" -exec rm -rf {} +
find "models/demos/segformer/demo/" -type f -name "._*" -exec rm -f {} +

echo "Cleanup complete."
