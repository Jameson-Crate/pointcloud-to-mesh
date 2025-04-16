#!/bin/bash

# PointCloud to Mesh Pipeline Script
# This script processes a point cloud through multiple stages to create a textured mesh

# Check if data directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <data_directory>"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="${DATA_DIR}/output"
mkdir -p "$OUTPUT_DIR"

echo "===== Starting pipeline processing ====="
echo "Input directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Step 1: Preprocess point cloud with single_preprocess.py
echo "===== Step 1: Preprocessing point cloud ====="
python3 single_preprocess.py --data_dir "$DATA_DIR" --combine_frames --frame_step 10 --extend_mask_kernel 23 --depth_gradient_thresh 0.7

# Check if point cloud was generated
POINT_CLOUD="${DATA_DIR}/point_clouds/scene.ply"
if [ ! -f "$POINT_CLOUD" ]; then
    echo "Error: Failed to generate point cloud at ${POINT_CLOUD}"
    exit 1
fi

echo "Point cloud generated: $POINT_CLOUD"

# Step 2: Run NKSR reconstruction
echo "===== Step 2: Running NKSR reconstruction ====="
NKSR_OUTPUT="${OUTPUT_DIR}/nksr_mesh.ply"
python3 run_nksr.py "$POINT_CLOUD" --output "$NKSR_OUTPUT" --detail 0.3

if [ ! -f "$NKSR_OUTPUT" ]; then
    echo "Error: Failed to generate NKSR mesh at ${NKSR_OUTPUT}"
    exit 1
fi

echo "NKSR mesh generated: $NKSR_OUTPUT"

# Step 3: Run Alpha Wrap to create watertight mesh
echo "===== Step 3: Running Alpha Wrap ====="
ALPHA_WRAP_OUTPUT="${OUTPUT_DIR}/alpha_wrapped"
mkdir -p "$ALPHA_WRAP_OUTPUT"

# Build Alpha Wrap if needed
if [ ! -f "alpha_wrap/build/AlphaWrap" ]; then
    echo "Building Alpha Wrap..."
    mkdir -p alpha_wrap/build
    cd alpha_wrap/build
    cmake ..
    make
    cd ../..
fi

# Run Alpha Wrap
alpha_wrap/build/AlphaWrap --alpha 100 --outdir "$ALPHA_WRAP_OUTPUT" "$NKSR_OUTPUT"

# Check if Alpha Wrap output was generated (assuming it creates output.off)
ALPHA_WRAP_MESH="${ALPHA_WRAP_OUTPUT}/output.off"
if [ ! -f "$ALPHA_WRAP_MESH" ]; then
    echo "Error: Failed to generate Alpha Wrap mesh at ${ALPHA_WRAP_MESH}"
    exit 1
fi

echo "Alpha Wrap mesh generated: $ALPHA_WRAP_MESH"

# Step 4: Texture mapping from original point cloud to final mesh
echo "===== Step 4: Applying texture mapping ====="
FINAL_OUTPUT="${OUTPUT_DIR}/final_textured_mesh.ply"
python3 texture_map.py "$POINT_CLOUD" "$ALPHA_WRAP_MESH" --output "$FINAL_OUTPUT"

if [ ! -f "$FINAL_OUTPUT" ]; then
    echo "Error: Failed to generate textured mesh at ${FINAL_OUTPUT}"
    exit 1
fi

echo "===== Pipeline completed successfully! ====="
echo "Final textured mesh: $FINAL_OUTPUT"