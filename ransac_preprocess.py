#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path


def filter_table_scene(input_file, output_file, min_dist=-0.02, max_dist=0.20, 
                       distance_threshold=0.01, ransac_n=3, num_iterations=1000, 
                       visualize=False):
    """
    Filter a point cloud to extract objects on a table surface using RANSAC.
    
    Args:
        input_file: Path to input point cloud file
        output_file: Path to save the filtered point cloud
        min_dist: Minimum distance from plane to keep points (negative = below plane)
        max_dist: Maximum distance from plane to keep points
        distance_threshold: RANSAC distance threshold for plane detection
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        visualize: Whether to visualize the result
    """
    # 1. Load point cloud
    print(f"Loading point cloud from {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load point cloud or file is empty: {input_file}")
    
    # 2. RANSAC plane detection
    print("Detecting table plane using RANSAC...")
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    # plane_model = [a, b, c, d] such that ax + by + cz + d = 0
    [a, b, c, d] = plane_model
    plane_normal_magnitude = np.sqrt(a*a + b*b + c*c)
    
    # 3. Calculate each point's distance to the plane
    points = np.asarray(pcd.points)
    distances = (a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / plane_normal_magnitude
    
    # 4. Select points that fall within [min_dist, max_dist]
    print(f"Filtering points between {min_dist}m and {max_dist}m from the plane...")
    indices_to_keep = np.where((distances >= min_dist) & (distances <= max_dist))[0]
    pcd_filtered = pcd.select_by_index(indices_to_keep)
    
    # 5. Save the result
    print(f"Saving filtered point cloud to {output_file}")
    o3d.io.write_point_cloud(output_file, pcd_filtered)
    print(f"Filtered cloud has {len(pcd_filtered.points)} points.")
    
    # 6. Visualize if requested
    if visualize:
        print("Visualizing filtered point cloud...")
        o3d.visualization.draw_geometries([pcd_filtered])
    
    return pcd_filtered


def main():
    parser = argparse.ArgumentParser(description='Filter a point cloud to extract objects on a table surface')
    parser.add_argument('input_file', help='Path to input point cloud file')
    parser.add_argument('--output_file', help='Path to save the filtered point cloud')
    parser.add_argument('--min_dist', type=float, default=-0.02, 
                        help='Minimum distance from plane to keep points (negative = below plane)')
    parser.add_argument('--max_dist', type=float, default=0.20, 
                        help='Maximum distance from plane to keep points')
    parser.add_argument('--distance_threshold', type=float, default=0.01, 
                        help='RANSAC distance threshold for plane detection')
    parser.add_argument('--ransac_n', type=int, default=3, 
                        help='Number of points to sample for RANSAC')
    parser.add_argument('--num_iterations', type=int, default=1000, 
                        help='Number of RANSAC iterations')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize the filtered point cloud')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1
    
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = input_path.with_name(f"{input_path.stem}_filtered{input_path.suffix}")
    
    try:
        filter_table_scene(
            input_file=args.input_file,
            output_file=output_path,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            distance_threshold=args.distance_threshold,
            ransac_n=args.ransac_n,
            num_iterations=args.num_iterations,
            visualize=args.visualize
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())