#!/usr/bin/env python3
import pybullet as p
import argparse
import os.path
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Generate convex decomposition from a mesh")
    parser.add_argument("input", help="Input mesh file (.obj or .ply)")
    parser.add_argument("output", help="Output convex decomposition file (.obj)")
    parser.add_argument("--log", help="Log file path", default="logs/vhacd_log.log")
    parser.add_argument("--resolution", type=int, default=300000, help="Maximum number of voxels generated")
    parser.add_argument("--max_hulls", "--depth", type=int, default=20, help="Maximum number of convex hulls / recursion depth")
    parser.add_argument("--concavity", type=float, default=0.0025, help="Maximum concavity")
    parser.add_argument("--planeDownsampling", type=int, default=4, help="Plane downsampling")
    parser.add_argument("--convexhullDownsampling", type=int, default=4, help="Convex hull downsampling")
    parser.add_argument("--maxNumVerticesPerCH", type=int, default=64, help="Maximum vertices per convex hull")
    parser.add_argument("--minVolumePerCH", type=float, default=0.0005, help="Minimum volume per convex hull")
    parser.add_argument("--convexhullApproximation", type=int, default=1, help="Convex hull approximation")
    parser.add_argument("--pca", type=int, default=1, help="PCA (Principal Component Analysis)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    input_file = args.input
    temp_file = None
    
    # Check if input is a PLY file and convert if needed
    if input_file.lower().endswith('.ply'):
        try:
            import trimesh
            print("Converting .ply to .obj for processing...")
            mesh = trimesh.load(input_file)
            
            # Create a temporary file for the converted .obj
            temp_fd, temp_file = tempfile.mkstemp(suffix='.obj')
            os.close(temp_fd)
            
            # Export to OBJ
            mesh.export(temp_file, file_type='obj')
            print("Temporary conversion to .obj successful")
            
            # Use the temporary file for processing
            input_file = temp_file
        except ImportError:
            print("Error: trimesh package not found. Install with 'pip install trimesh'")
            sys.exit(1)
        except Exception as e:
            print(f"Error converting .ply to .obj: {e}")
            sys.exit(1)
        
    print(f"Generating convex decomposition from {args.input}")
    
    # PyBullet VHACD parameters
    try:
        # Check if output is a PLY file
        output_is_ply = args.output.lower().endswith('.ply')
        output_file = args.output
        
        # If output is PLY, create a temporary OBJ file for VHACD
        temp_output_file = None
        if output_is_ply:
            temp_fd, temp_output_file = tempfile.mkstemp(suffix='.obj')
            os.close(temp_fd)
            output_file = temp_output_file
        
        # Use PyBullet for decomposition
        p.vhacd(
            input_file, 
            output_file, 
            args.log,
            depth=args.max_hulls,
            resolution=args.resolution,
            concavity=args.concavity,
            planeDownsampling=args.planeDownsampling,
            convexhullDownsampling=args.convexhullDownsampling,
            pca=args.pca,
            maxNumVerticesPerCH=args.maxNumVerticesPerCH,
            minVolumePerCH=args.minVolumePerCH,
            convexhullApproximation=args.convexhullApproximation,
            oclAcceleration=0  # OpenCL acceleration off by default
        )
        
        # Convert OBJ to PLY if needed
        if output_is_ply and temp_output_file:
            try:
                import trimesh
                print("Converting output .obj to .ply...")
                mesh = trimesh.load(temp_output_file)
                mesh.export(args.output, file_type='ply')
                print(f"Convex decomposition completed. Output saved to {args.output}")
            except ImportError:
                print("Error: trimesh package not found. Install with 'pip install trimesh'")
                sys.exit(1)
            except Exception as e:
                print(f"Error converting output .obj to .ply: {e}")
                sys.exit(1)
        else:
            print(f"Convex decomposition completed. Output saved to {args.output}")
    finally:
        # Clean up temporary files if they were created
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        if temp_output_file and os.path.exists(temp_output_file):
            os.remove(temp_output_file)

if __name__ == "__main__":
    main()
