#!/usr/bin/env python3
"""
Script to convert mesh files between various formats.
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from plyfile import PlyData
import copy


def transform_coordinates_for_format(mesh, target_format):
    """
    Transform mesh coordinates to match the convention of the target file format.
    
    Different file formats use different coordinate system conventions:
    - PLY (photogrammetry/computer vision): Y-up or Z-up depending on scanner
    - OBJ/GLB/GLTF (3D modeling/graphics): Typically Y-up (Blender) or Z-up (other)
    
    This fixes the common 90-degree rotation when going from PLY to OBJ/GLB.
    
    Args:
        mesh: trimesh mesh object
        target_format: Target file format extension (e.g., '.obj', '.glb')
    
    Returns:
        trimesh mesh: Mesh with transformed coordinates
    """
    target_format = target_format.lower() if not target_format.startswith('.') else target_format.lower()
    target_format = f".{target_format}" if not target_format.startswith('.') else target_format
    
    if target_format not in ['.obj', '.glb', '.gltf']:
        # No transformation needed for other formats
        return mesh
    
    print(f"Applying coordinate system transformation for {target_format} format...")
    transformed_mesh = copy.deepcopy(mesh)
    
    # Get vertices as numpy array for transformation
    vertices = np.array(transformed_mesh.vertices).copy()
    
    # Apply 90-degree rotation around X-axis to convert from Y-up to Z-up
    # This is the transformation matrix:
    # [ 1  0  0 ]   [ x ]   [ x  ]
    # [ 0  0 -1 ] * [ y ] = [ -z ]
    # [ 0  1  0 ]   [ z ]   [ y  ]
    y_temp = vertices[:, 1].copy()
    vertices[:, 1] = vertices[:, 2]  # y = z
    vertices[:, 2] = -y_temp         # z = -y
    
    # Update the mesh with transformed vertices
    transformed_mesh.vertices = vertices
    
    # If the mesh has vertex normals, transform them too
    if hasattr(transformed_mesh, 'vertex_normals') and transformed_mesh.vertex_normals is not None:
        normals = np.array(transformed_mesh.vertex_normals).copy()
        y_temp = normals[:, 1].copy()
        normals[:, 1] = normals[:, 2]
        normals[:, 2] = -y_temp
        transformed_mesh.vertex_normals = normals
    
    # If the mesh has face normals, transform them too
    if hasattr(transformed_mesh, 'face_normals') and transformed_mesh.face_normals is not None:
        tri_normals = np.array(transformed_mesh.face_normals).copy()
        y_temp = tri_normals[:, 1].copy()
        tri_normals[:, 1] = tri_normals[:, 2]
        tri_normals[:, 2] = -y_temp
        transformed_mesh.face_normals = tri_normals
    
    return transformed_mesh


def convert_mesh(input_file, output_file=None, output_format=None):
    """
    Convert a mesh file from one format to another.
    
    Args:
        input_file (str): Path to the input mesh file
        output_file (str, optional): Path to the output mesh file. If not provided,
                                    will use the same name as input with new extension.
        output_format (str, optional): Format for the output file. If not provided,
                                      will be inferred from output_file extension.
    
    Returns:
        str: Path to the output mesh file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine output format
    if output_format is None:
        if output_file is None:
            output_format = 'glb'  # Default format
        else:
            _, ext = os.path.splitext(output_file)
            output_format = ext[1:] if ext else 'glb'
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + f'.{output_format}'
    
    # Load the input mesh file
    try:
        # Try using trimesh
        mesh = trimesh.load(input_file)
    except Exception as e:
        print(f"Trimesh loading failed: {e}")
        # Handle PLY files specifically with plyfile as fallback
        if input_file.lower().endswith('.ply'):
            try:
                # Fallback to manual loading with plyfile
                ply_data = PlyData.read(input_file)
                
                # Extract vertices
                vertices = np.vstack([
                    ply_data['vertex']['x'],
                    ply_data['vertex']['y'],
                    ply_data['vertex']['z']
                ]).T
                
                # Extract faces if available
                if 'face' in ply_data:
                    faces = np.vstack([f[0] for f in ply_data['face']['vertex_indices']])
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                else:
                    # Create point cloud if no faces
                    mesh = trimesh.PointCloud(vertices)
            except Exception as e2:
                raise RuntimeError(f"Failed to load file {input_file}: {e2}")
        else:
            raise RuntimeError(f"Failed to load file {input_file}: {e}")
    
    # Apply coordinate transformation based on output format
    input_ext = os.path.splitext(input_file)[1].lower()
    output_ext = f".{output_format}"
    
    # Only transform if going from PLY to OBJ/GLB/GLTF
    if input_ext == '.ply' and output_ext in ['.obj', '.glb', '.gltf']:
        mesh = transform_coordinates_for_format(mesh, output_ext)
    
    # Export to the output format
    try:
        mesh.export(output_file, file_type=output_format)
        print(f"Successfully converted {input_file} to {output_file}")
        return output_file
    except Exception as e:
        raise RuntimeError(f"Failed to export to {output_format} format: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert mesh files between formats')
    parser.add_argument('input', help='Input mesh file or directory containing mesh files')
    parser.add_argument('-o', '--output', help='Output mesh file or directory')
    parser.add_argument('-f', '--format', help='Output format (e.g., glb, obj, stl, ply)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('-t', '--types', default='.ply,.obj,.stl,.glb,.gltf',
                      help='Comma-separated list of mesh file extensions to process (default: .ply,.obj,.stl,.glb,.gltf)')
    args = parser.parse_args()
    
    # Parse file types to process
    valid_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                        for ext in args.types.split(',')]
    
    if os.path.isfile(args.input):
        # Process single file
        convert_mesh(args.input, args.output, args.format)
    elif os.path.isdir(args.input):
        # Process directory
        if args.output and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
        
        for root, dirs, files in os.walk(args.input):
            if not args.recursive and root != args.input:
                continue
                
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in valid_extensions:
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, args.input)
                    
                    if args.output:
                        if args.format:
                            output_name = f"{os.path.splitext(os.path.basename(rel_path))[0]}.{args.format}"
                        else:
                            output_name = os.path.basename(rel_path)
                        output_path = os.path.join(args.output, 
                                                  os.path.dirname(rel_path),
                                                  output_name)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_path = None
                        
                    try:
                        convert_mesh(input_path, output_path, args.format)
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")
    else:
        print(f"Input path does not exist: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
