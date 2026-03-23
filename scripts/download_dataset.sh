#!/bin/bash

URL="https://data.mendeley.com/public-api/zip/j4y5w2c8w9/download/1"
ZIP_FILE="asl_dataset.zip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../data"

echo "================================================"
echo "  ASL Dataset Downloader"
echo "================================================"
echo "Source : Mendeley Data"
echo "Output : $OUTPUT_DIR/"
echo "Downloading dataset (1.13 GB)..."

mkdir -p "$OUTPUT_DIR"

# Step 1: Download
curl -L --progress-bar -o "$SCRIPT_DIR/$ZIP_FILE" "$URL"

if [ $? -ne 0 ]; then
    echo "Download failed."
    exit 1
fi

echo "Download complete: $ZIP_FILE"

# Step 2: Extract outer zip to temp dir
TEMP_DIR="$SCRIPT_DIR/tmp_asl"
mkdir -p "$TEMP_DIR"

echo "Extracting outer zip..."
unzip -q "$SCRIPT_DIR/$ZIP_FILE" -d "$TEMP_DIR"

if [ $? -ne 0 ]; then
    echo "Outer extraction failed."
    rm -f "$SCRIPT_DIR/$ZIP_FILE"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Step 3: Extract nested processed images zip
NESTED_ZIP="$TEMP_DIR/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/ASL_Processed_Images.zip"

echo "Extracting processed images..."
unzip -q "$NESTED_ZIP" -d "$TEMP_DIR"

if [ $? -ne 0 ]; then
    echo "Nested extraction failed."
    rm -f "$SCRIPT_DIR/$ZIP_FILE"
    rm -rf "$TEMP_DIR"
    exit 1
fi

mv "$TEMP_DIR/asl_processed/train" "$OUTPUT_DIR"
mv "$TEMP_DIR/asl_processed/test" "$OUTPUT_DIR"

# Step 6: Cleanup
echo "Cleaning up..."
rm -f "$SCRIPT_DIR/$ZIP_FILE"
rm -rf "$TEMP_DIR"

echo "Dataset saved to: $OUTPUT_DIR"