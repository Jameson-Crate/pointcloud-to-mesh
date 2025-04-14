#!/usr/bin/env python3

import numpy as np
import argparse
import os
from pathlib import Path

def load_pointcloud_txt(file_path):
    """
    Load a pointcloud from a text file.
    Expected format: x y z [other columns]
    """
    data = np.loadtxt(file_path, delimiter=' ')
    points = data[:, :3]  # Extract xyz coordinates
    
    # If there are additional columns (like RGB), keep them
    extra_data = data[:, 3:] if data.shape[1] > 3 else None
    
    return points, extra_data

def load_pointcloud_ply(file_path):
    """
    Load a pointcloud from a PLY file using Open3D.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: open3d package not found. Install with 'pip install open3d'")
        exit(1)
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(str(file_path))
    
    # Extract points
    points = np.asarray(pcd.points)
    
    # Extract colors if available
    extra_data = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)  # Colors are in range [0, 1]
        extra_data = colors
    
    return points, extra_data

def load_pointcloud(file_path):
    """
    Load a pointcloud from a file based on its extension.
    Supports .txt, .xyz, .csv, and .ply files.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext == '.ply':
        return load_pointcloud_ply(file_path)
    elif ext in ['.txt', '.xyz', '.csv']:
        return load_pointcloud_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are .txt, .xyz, .csv, and .ply")

def rotate_pointcloud_x_180(points):
    """
    Rotate the pointcloud 180 degrees around the X-axis to convert between
    COLMAP/OpenCV convention and OpenGL/Blender convention.
    
    COLMAP/OpenCV:
    - Forward: +Z
    - Up: -Y
    - Right: +X
    
    OpenGL/Blender:
    - Forward: -Z
    - Up: +Y
    - Right: +X
    """
    # Rotation matrix for 180 degrees around X-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    # Apply rotation
    rotated_points = points @ rotation_matrix.T
    
    return rotated_points

def save_pointcloud(file_path, points, extra_data=None):
    """
    Save the pointcloud to a file based on its extension.
    Supports .txt, .xyz, .csv, and .ply files.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in ['.txt', '.xyz', '.csv']:
        if extra_data is not None:
            data = np.hstack((points, extra_data))
        else:
            data = points
        
        np.savetxt(file_path, data, delimiter=' ')
    
    elif ext == '.ply':
        try:
            import open3d as o3d
        except ImportError:
            print("Error: open3d package not found. Install with 'pip install open3d'")
            exit(1)
        
        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if available
        if extra_data is not None and extra_data.shape[1] >= 3:
            # Make sure colors are in range [0, 1]
            colors = extra_data[:, :3]
            if np.max(colors) > 1.0:
                colors = colors / 255.0
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save the point cloud
        o3d.io.write_point_cloud(str(file_path), pcd)
    
    else:
        raise ValueError(f"Unsupported output file extension: {ext}. Supported extensions are .txt, .xyz, .csv, and .ply")

def main():
    parser = argparse.ArgumentParser(description='Rotate a pointcloud by 180 degrees around the X-axis')
    parser.add_argument('input_file', help='Path to input pointcloud file (.txt, .xyz, .csv, or .ply)')
    parser.add_argument('--output_file', help='Path to output pointcloud file. If not specified, will use input_file_rotated[.ext]')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_rotated{input_path.suffix}")
    
    print(f"Loading pointcloud from {input_path}")
    points, extra_data = load_pointcloud(input_path)
    
    print(f"Rotating pointcloud 180 degrees around X-axis")
    rotated_points = rotate_pointcloud_x_180(points)
    
    print(f"Saving rotated pointcloud to {output_path}")
    save_pointcloud(output_path, rotated_points, extra_data)
    
    print("Done")

if __name__ == "__main__":
    main()
