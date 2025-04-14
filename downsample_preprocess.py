#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path


def downsample_pointcloud(input_file, output_file, percentage, method='random', voxel_size=None, visualize=False):
    """
    Downsample a point cloud by a specified percentage.
    
    Args:
        input_file: Path to input point cloud file
        output_file: Path to save the downsampled point cloud
        percentage: Percentage of points to keep (0-100)
        method: Downsampling method ('random', 'voxel', or 'uniform')
        voxel_size: Voxel size for voxel-based downsampling (auto-calculated if None)
        visualize: Whether to visualize the result
    """
    # 1. Load point cloud
    print(f"Loading point cloud from {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load point cloud or file is empty: {input_file}")
    
    original_points = len(pcd.points)
    print(f"Original point cloud has {original_points} points")
    
    # 2. Downsample based on the selected method
    if method == 'random':
        # Random downsampling
        target_points = int(original_points * percentage / 100)
        print(f"Randomly downsampling to {target_points} points ({percentage}%)")
        
        points = np.asarray(pcd.points)
        indices = np.random.choice(original_points, target_points, replace=False)
        
        # Create new downsampled point cloud
        pcd_down = o3d.geometry.PointCloud()
        pcd_down.points = o3d.utility.Vector3dVector(points[indices])
        
        # Copy colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            pcd_down.colors = o3d.utility.Vector3dVector(colors[indices])
    
    elif method == 'voxel':
        # Voxel-based downsampling
        if voxel_size is None:
            # Auto-calculate voxel size to achieve target percentage
            # This is an approximation and may require multiple attempts
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
            volume = np.prod(bbox_size)
            
            # Start with an estimated voxel size
            voxel_size = (volume / original_points * (100 / percentage)) ** (1/3)
        
        print(f"Voxel downsampling with voxel size {voxel_size}")
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Adjust voxel size if needed to get closer to target percentage
        current_percentage = len(pcd_down.points) / original_points * 100
        print(f"Achieved {current_percentage:.2f}% of original points")
    
    elif method == 'uniform':
        # Uniform downsampling
        every_k_points = max(1, int(100 / percentage))
        print(f"Uniform downsampling: keeping 1 point for every {every_k_points} points")
        pcd_down = pcd.uniform_down_sample(every_k_points)
        
        current_percentage = len(pcd_down.points) / original_points * 100
        print(f"Achieved {current_percentage:.2f}% of original points")
    
    else:
        raise ValueError(f"Unknown downsampling method: {method}")
    
    # 3. Save the result
    print(f"Saving downsampled point cloud to {output_file}")
    o3d.io.write_point_cloud(output_file, pcd_down)
    print(f"Downsampled cloud has {len(pcd_down.points)} points")
    
    # 4. Visualize if requested
    if visualize:
        print("Visualizing downsampled point cloud...")
        o3d.visualization.draw_geometries([pcd_down])
    
    return pcd_down


def main():
    parser = argparse.ArgumentParser(description='Downsample a point cloud by a specified percentage')
    parser.add_argument('input_file', help='Path to input point cloud file')
    parser.add_argument('--output_file', help='Path to save the downsampled point cloud')
    parser.add_argument('--percentage', type=float, default=10.0, 
                        help='Percentage of points to keep (0-100)')
    parser.add_argument('--method', choices=['random', 'voxel', 'uniform'], default='random',
                        help='Downsampling method to use')
    parser.add_argument('--voxel_size', type=float, default=None,
                        help='Voxel size for voxel-based downsampling (auto-calculated if not specified)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the downsampled point cloud')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_downsampled{input_path.suffix}")
    
    try:
        downsample_pointcloud(
            input_path, 
            output_path, 
            args.percentage, 
            args.method, 
            args.voxel_size, 
            args.visualize
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
