#!/usr/bin/env python3

import numpy as np
import argparse
import os
from pathlib import Path
import open3d as o3d

def load_pointcloud(file_path):
    """
    Load a pointcloud from a file based on its extension.
    Supports .ply, .pcd, .xyz, .txt, and .csv files.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in ['.ply', '.pcd']:
        # Use Open3D to load the point cloud
        pcd = o3d.io.read_point_cloud(str(file_path))
        return pcd
    elif ext in ['.txt', '.xyz', '.csv']:
        # Load from text file and convert to Open3D format
        data = np.loadtxt(file_path, delimiter=' ')
        points = data[:, :3]  # Extract xyz coordinates
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # If there are RGB values (assuming they follow XYZ)
        if data.shape[1] >= 6:
            colors = data[:, 3:6]
            # Normalize colors if they're in range [0, 255]
            if np.max(colors) > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
        return pcd
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def estimate_normals(pcd, radius=None, max_nn=30, orient_consistent=True):
    """
    Estimate normals for a point cloud.
    
    Args:
        pcd: Open3D point cloud
        radius: Search radius for normal estimation. If None, it's estimated automatically.
        max_nn: Maximum number of nearest neighbors to use
        orient_consistent: Whether to orient normals consistently
        
    Returns:
        Point cloud with estimated normals
    """
    # If radius is not provided, estimate it based on the point cloud scale
    if radius is None:
        # Compute the bounding box and use a fraction of its diagonal as radius
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        radius = np.linalg.norm(bbox_size) * 0.02  # 2% of the bounding box diagonal
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    # Orient normals consistently if requested
    if orient_consistent:
        # Try to orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    
    return pcd

def save_pointcloud_with_normals(pcd, output_path):
    """
    Save a point cloud with normals to a file.
    """
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    
    if ext in ['.ply', '.pcd']:
        # Direct save using Open3D
        o3d.io.write_point_cloud(str(output_path), pcd)
    elif ext in ['.txt', '.xyz', '.csv']:
        # Save as text file with normals
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Combine points and normals
        data = np.hstack((points, normals))
        
        # Add colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            data = np.hstack((data, colors))
        
        np.savetxt(output_path, data, delimiter=' ')
    else:
        raise ValueError(f"Unsupported output file extension: {ext}")

def main():
    parser = argparse.ArgumentParser(description='Estimate normals for a point cloud')
    parser.add_argument('input_file', help='Path to input point cloud file')
    parser.add_argument('--output_file', help='Path to output file. If not specified, will use input_file_normals[.ext]')
    parser.add_argument('--radius', type=float, help='Search radius for normal estimation. If not specified, it will be estimated automatically.')
    parser.add_argument('--max_nn', type=int, default=30, help='Maximum number of nearest neighbors to use')
    parser.add_argument('--no_orient', action='store_true', help='Disable consistent normal orientation')
    parser.add_argument('--visualize', action='store_true', help='Visualize the point cloud with normals')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_normals{input_path.suffix}")
    
    print(f"Loading point cloud from {input_path}")
    pcd = load_pointcloud(input_path)
    
    print(f"Estimating normals")
    pcd = estimate_normals(
        pcd, 
        radius=args.radius, 
        max_nn=args.max_nn, 
        orient_consistent=not args.no_orient
    )
    
    print(f"Saving point cloud with normals to {output_path}")
    save_pointcloud_with_normals(pcd, output_path)
    
    if args.visualize:
        print("Visualizing point cloud with normals")
        # Visualize the point cloud with normals using draw_geometries
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    print("Done")

if __name__ == "__main__":
    main()
