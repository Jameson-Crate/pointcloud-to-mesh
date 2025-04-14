#!/usr/bin/env python3
"""
Script to map texture (color) from a point cloud onto a corresponding mesh.
This script uses nearest triangle and barycentric coordinate mapping to transfer
the colors from the point cloud to the mesh.
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm
import copy


def load_pointcloud(pointcloud_path):
    """
    Load a point cloud from file.
    
    Args:
        pointcloud_path: Path to the point cloud file
        
    Returns:
        open3d.geometry.PointCloud: Loaded point cloud
    """
    print(f"Loading point cloud from {pointcloud_path}")
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load point cloud or file is empty: {pointcloud_path}")
    
    if not pcd.has_colors():
        raise ValueError(f"Point cloud has no color information: {pointcloud_path}")
    
    print(f"Loaded point cloud with {len(pcd.points)} points")
    return pcd


def load_mesh(mesh_path):
    """
    Load a mesh from file.
    
    Args:
        mesh_path: Path to the mesh file
        
    Returns:
        open3d.geometry.TriangleMesh: Loaded mesh
    """
    print(f"Loading mesh from {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh or file is empty: {mesh_path}")
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh

def inverse_texture_mapping(pcd, mesh, max_distance=None, num_neighbors=3):
    """
    Inverse method to map textures using interpolation of nearest points.
    This can be more robust for sparse point clouds.
    
    Args:
        pcd: Point cloud with color information
        mesh: Mesh to apply texture to
        max_distance: Maximum distance for color transfer
        num_neighbors: Number of nearest neighbors to consider for interpolation
    
    Returns:
        open3d.geometry.TriangleMesh: Mesh with texture mapped from point cloud
    """
    mesh_vertices = np.asarray(mesh.vertices)
    cloud_points = np.asarray(pcd.points)
    cloud_colors = np.asarray(pcd.colors)
    
    # Create a KD-tree for the point cloud
    pcd_tree = KDTree(cloud_points)
    
    # Initialize mesh vertex colors
    vertex_colors = np.zeros((len(mesh_vertices), 3))
    
    # If no max distance is specified, estimate one from the data
    if max_distance is None:
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        mesh_extent = np.linalg.norm(mesh_bbox.get_max_bound() - mesh_bbox.get_min_bound())
        max_distance = mesh_extent * 0.05  # Use 5% of the bounding box size as default
    
    print(f"Mapping colors using k-nearest neighbors (k={num_neighbors}, max distance: {max_distance:.6f})")
    
    # Find nearest neighbors for each vertex and interpolate colors
    for i, vertex in tqdm(enumerate(mesh_vertices), total=len(mesh_vertices), desc="Processing vertices"):
        # Find nearest points in the point cloud
        distances, indices = pcd_tree.query(vertex, k=num_neighbors)
        
        # Weight colors by inverse distance
        weights = np.zeros(len(indices))
        valid_indices = []
        
        for j, (idx, dist) in enumerate(zip(indices, distances)):
            if dist <= max_distance:
                weights[j] = 1.0 / (dist + 1e-10)  # Avoid division by zero
                valid_indices.append(j)
        
        if valid_indices:
            # Normalize weights
            weights = weights[valid_indices] / np.sum(weights[valid_indices])
            # Apply weighted colors
            for w, idx in zip(weights, indices[valid_indices]):
                vertex_colors[i] += cloud_colors[idx] * w
    
    # Apply colors to the mesh
    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    colored_mesh.triangles = mesh.triangles
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Set any other mesh properties
    if mesh.has_vertex_normals():
        colored_mesh.vertex_normals = mesh.vertex_normals
    if mesh.has_triangle_normals():
        colored_mesh.triangle_normals = mesh.triangle_normals
    
    return colored_mesh


def transform_coordinates_for_format(mesh, target_format):
    """
    Transform mesh coordinates to match the convention of the target file format.
    
    Different file formats use different coordinate system conventions:
    - PLY (photogrammetry/computer vision): Y-up or Z-up depending on scanner
    - OBJ/GLB/GLTF (3D modeling/graphics): Typically Y-up (Blender) or Z-up (other)
    
    This fixes the common 90-degree rotation when going from PLY to OBJ/GLB.
    
    Args:
        mesh: Open3D triangle mesh
        target_format: Target file format extension (e.g., '.obj', '.glb')
    
    Returns:
        open3d.geometry.TriangleMesh: Mesh with transformed coordinates
    """
    target_format = target_format.lower()
    if target_format not in ['.obj', '.glb', '.gltf']:
        # No transformation needed for other formats
        return mesh
    
    print(f"Applying coordinate system transformation for {target_format} format...")
    transformed_mesh = copy.deepcopy(mesh)
    
    # Get vertices as numpy array for transformation
    vertices = np.asarray(transformed_mesh.vertices).copy()
    
    # Apply 90-degree rotation around X-axis to convert from Y-up to Z-up
    # This is the transformation matrix:
    # [ 1  0  0 ]   [ x ]   [ x  ]
    # [ 0  0 -1 ] * [ y ] = [ -z ]
    # [ 0  1  0 ]   [ z ]   [ y  ]
    y_temp = vertices[:, 1].copy()
    vertices[:, 1] = vertices[:, 2]  # y = z
    vertices[:, 2] = -y_temp           # z = -y
    
    # Update the mesh with transformed vertices
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # If the mesh has normals, transform them too
    if transformed_mesh.has_vertex_normals():
        normals = np.asarray(transformed_mesh.vertex_normals).copy()
        y_temp = normals[:, 1].copy()
        normals[:, 1] = normals[:, 2]
        normals[:, 2] = -y_temp
        transformed_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    
    if transformed_mesh.has_triangle_normals():
        tri_normals = np.asarray(transformed_mesh.triangle_normals).copy()
        y_temp = tri_normals[:, 1].copy()
        tri_normals[:, 1] = tri_normals[:, 2]
        tri_normals[:, 2] = -y_temp
        transformed_mesh.triangle_normals = o3d.utility.Vector3dVector(tri_normals)
    
    return transformed_mesh


def main():
    parser = argparse.ArgumentParser(description='Map texture from a point cloud to a mesh')
    parser.add_argument('pointcloud', help='Path to the point cloud file with color information')
    parser.add_argument('mesh', help='Path to the mesh file to texture')
    parser.add_argument('--output', '-o', help='Path to save the textured mesh')
    parser.add_argument('--max_distance', type=float, default=None,
                        help='Maximum distance between mesh and point cloud points for color mapping')
    parser.add_argument('--neighbors', type=int, default=3,
                        help='Number of neighbors')
    parser.add_argument('--upsample', type=float, default=0,
                        help='Upsample mesh to match a percentage of the point cloud size (0 to disable, e.g. 0.8 for 80%%)')
    parser.add_argument('--fix-rotation', action='store_true',
                        help='Fix coordinate system rotation when saving to OBJ/GLB formats')
    
    args = parser.parse_args()
    
    try:
        # Load input files
        pcd = load_pointcloud(args.pointcloud)
        mesh = load_mesh(args.mesh)
        
        # Upsample mesh if requested
        if args.upsample > 0:
            target_vertices = int(len(pcd.points) * args.upsample)
            current_vertices = len(mesh.vertices)
            
            if target_vertices > current_vertices:
                print(f"Upsampling mesh from {current_vertices} to approximately {target_vertices} vertices...")
                
                # Calculate how many subdivision iterations we need
                # Each subdivision roughly quadruples the number of vertices
                iterations = 0
                estimated_vertices = current_vertices
                while estimated_vertices < target_vertices:
                    estimated_vertices *= 4
                    iterations += 1
                
                # We might have overshot, so go back one iteration if needed
                if iterations > 0 and estimated_vertices / 4 >= target_vertices:
                    iterations -= 1
                    estimated_vertices /= 4
                
                if iterations > 0:
                    mesh = mesh.subdivide_midpoint(number_of_iterations=iterations)
                    print(f"Performed {iterations} subdivision(s), resulting in {len(mesh.vertices)} vertices")
                else:
                    print("No upsampling needed as mesh already has sufficient vertices")
            else:
                print(f"No upsampling needed: mesh has {current_vertices} vertices, target was {target_vertices}")
        
        # Map texture from point cloud to mesh
        colored_mesh = inverse_texture_mapping(
            pcd, mesh, args.max_distance, args.neighbors
        )
        
        # Save output
        if args.output:
            output_path = args.output
        else:
            input_mesh_path = Path(args.mesh)
            output_path = str(input_mesh_path.with_name(f"{input_mesh_path.stem}_textured{input_mesh_path.suffix}"))
        
        # Check file format
        file_ext = Path(output_path).suffix.lower()
        mesh_to_save = colored_mesh
        
        # Handle different file formats
        if file_ext == '.stl':
            print("WARNING: STL format does not support color information. The mesh geometry will be saved,")
            print("         but all color/texture information will be lost.")
            print("         Consider using .ply, .obj, or .glb formats to preserve colors.")
            save_anyway = input("Do you want to save as STL anyway? (y/n): ")
            if save_anyway.lower() != 'y':
                alt_format = '.ply'
                output_path = str(Path(output_path).with_suffix(alt_format))
                file_ext = alt_format
                print(f"Saving with {alt_format} format instead: {output_path}")
        
        # Fix coordinate system differences if needed
        if file_ext.lower() in ['.obj', '.glb', '.gltf'] and args.fix_rotation:
            mesh_to_save = transform_coordinates_for_format(colored_mesh, file_ext)
        
        print(f"Saving textured mesh to {output_path}")
        success = o3d.io.write_triangle_mesh(output_path, mesh_to_save)
        
        if not success:
            print(f"Error: Failed to save mesh to {output_path}")
            # Try to save in a different format as fallback
            if file_ext != '.ply':
                fallback_path = str(Path(output_path).with_suffix('.ply'))
                print(f"Trying to save as PLY format instead: {fallback_path}")
                fallback_success = o3d.io.write_triangle_mesh(fallback_path, colored_mesh)
                if fallback_success:
                    print(f"Successfully saved to {fallback_path}")
                else:
                    print("Failed to save fallback file as well.")
            return 1
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
