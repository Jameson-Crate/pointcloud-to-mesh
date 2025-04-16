import os
import argparse
import numpy as np
import torch
import open3d as o3d
import nksr

def load_pointcloud(file_path):
    """Load a pointcloud from a file using Open3D."""
    print(f"Loading pointcloud from {file_path}")
    
    # Determine file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Load the pointcloud based on file type
    if ext in ['.ply', '.pcd', '.xyz', '.pts']:
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    if not pcd.has_points():
        raise ValueError("Loaded pointcloud has no points")
    
    # Estimate normals if they don't exist
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    return pcd

def load_bunny_example():
    """Load the Stanford bunny example."""
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    # Sample the mesh to get a pointcloud
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    return pcd

def main():
    parser = argparse.ArgumentParser(description="Run NKSR reconstruction on a pointcloud file")
    parser.add_argument("input_file", type=str, help="Path to the input pointcloud file")
    parser.add_argument("--output", type=str, default="output.ply", help="Output mesh file path")
    parser.add_argument("--detail", type=float, default=1.0, help="Detail level for reconstruction")
    parser.add_argument("--mise_iter", type=int, default=1, help="MISE iterations")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Load pointcloud
    if args.input_file.lower() == "bunny":
        print("Loading Stanford bunny example")
        pcd = load_bunny_example()
    else:
        pcd = load_pointcloud(args.input_file)
    
    # Convert to torch tensors
    input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)
    
    print(f"Pointcloud loaded: {input_xyz.shape[0]} points")
    
    # Run NKSR reconstruction
    print("Running NKSR reconstruction...")
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=args.detail)
    
    # Extract mesh
    print(f"Extracting mesh with MISE iterations: {args.mise_iter}")
    mesh = field.extract_dual_mesh(mise_iter=args.mise_iter)
    
    # Save the mesh
    print(f"Saving mesh to {args.output}")
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.v.cpu().numpy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.f.cpu().numpy())
    o3d.io.write_triangle_mesh(args.output, o3d_mesh)
    
    print("Done!")

if __name__ == "__main__":
    main()
