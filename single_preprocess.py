"""
This is the script to remove the MegaSaM point cloud based on SAM2F2 masks for single folder.
"""

import glob
import open3d as o3d
import argparse
import cv2
import imageio
import numpy as np
from pathlib import Path
import random
from scipy import ndimage


def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = depth < 0.001
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(
            np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij"
        )
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def extend_mask(mask, kernel_size=5, method='max'):
    """
    Extend a binary mask using convolution.
    
    Args:
        mask (numpy.ndarray): Binary mask to extend
        kernel_size (int): Size of the convolution kernel
        method (str): Method to use ('max' or 'mode')
        
    Returns:
        numpy.ndarray: Extended binary mask
    """
    if method == 'max':
        # Use dilation operation for max method
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        extended_mask = cv2.dilate(mask, kernel, iterations=1)
    elif method == 'mode':
        # Use mode filter for mode method
        kernel = np.ones((kernel_size, kernel_size))
        extended_mask = ndimage.convolve(mask, kernel, mode='constant', cval=0)
        extended_mask = (extended_mask > (kernel_size**2) / 2).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported extension method: {method}")
    
    return extended_mask


class MegaSamReader:
    def __init__(
        self, video_dir, downscale=1, shorter_side=None, zfar=np.inf, object_names=[]
    ):
        self.video_dir = video_dir
        self.file_name = "sgd_cvd_hr.npz"
        data = np.load(str(self.video_dir / self.file_name))
        self.downscale = downscale
        self.zfar = zfar
        self.color_files = data["images"]  # (N, H, W, 3) uint8
        self.depth_files = data["depths"]  # (N, H, W) float16
        self.K = data["intrinsic"].astype(np.float64)  # (3, 3) float32
        self.camera_poses = data["cam_c2w"]  # (N, 4, 4) float32
        self.masks = {}
        for object_name in object_names:
            self.masks[object_name] = sorted(
                glob.glob(f"{self.video_dir}/masks/segment_{object_name}*.png")
            )
        self.id_strs = [f"{idx:04d}" for idx in range(len(self.color_files))]
        self.H, self.W = self.color_files.shape[1:3]
        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)
        assert self.downscale == 1

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i):
        color = self.color_files[i]
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_depth(self, i):
        depth = self.depth_files[i]
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_extrinsic(self, i):
        return self.camera_poses[i]

    def get_mask(self, i, object_name):
        if object_name not in self.masks:
            raise ValueError(f"Object '{object_name}' not found in mask list.")
        mask_files = self.masks[object_name]
        if i >= len(mask_files):
            raise IndexError(
                f"Frame index {i} out of range for object '{object_name}'."
            )
        mask = cv2.imread(mask_files[i], -1)
        if mask is None:
            raise FileNotFoundError(f"Mask file {mask_files[i]} could not be loaded.")
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = (
            cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            .astype(bool)
            .astype(np.uint8)
        )
        return mask

    def save_images_and_depths(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(self)):
            color = self.get_color(i)
            depth = self.get_depth(i)
            imageio.imwrite(output_dir / f"frame_{i:04d}.png", color)
            np.save(output_dir / f"frame_{i:04d}.npy", depth)


def save_point_cloud(xyz_map, color, masks, output_file="point_cloud.ply", extend_mask_params=None):
    """
    Save a masked point cloud with color.
    Args:
        xyz_map (numpy.ndarray): 3D point map of shape [H, W, 3].
        color (numpy.ndarray): Corresponding color image of shape [H, W, 3].
        masks (list of numpy.ndarray): List of binary masks of shape [H, W].
        output_file (str): Output filename for the point cloud.
        extend_mask_params (dict, optional): Parameters for mask extension.
    """
    if not masks:
        raise ValueError("At least one mask must be provided.")
    
    # Process masks with extension if requested
    processed_masks = []
    for mask in masks:
        if extend_mask_params:
            mask = extend_mask(
                mask, 
                kernel_size=extend_mask_params['kernel_size'], 
                method=extend_mask_params['method']
            )
        processed_masks.append(mask)
    
    # Combine all masks using logical OR
    combined_mask = np.logical_or.reduce(processed_masks)
    valid_mask = ~combined_mask
    
    # Extract valid 3D points and corresponding colors
    points = xyz_map[valid_mask]
    colors = color[valid_mask] / 255.0  # Normalize to [0,1] for Open3D
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to file
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved point cloud to {output_file}")
    
    return pcd


def monte_carlo_average_point_clouds(reader, object_names, output_file, 
                                     num_frames=None, sampling_ratio=0.1, 
                                     extend_mask_params=None):
    """
    Combine point clouds from multiple frames using Monte Carlo averaging.
    
    Args:
        reader: MegaSamReader instance
        object_names: List of object names to mask out
        output_file: Output filename for the combined point cloud
        num_frames: Number of frames to sample (None for all)
        sampling_ratio: Ratio of points to sample from each frame
        extend_mask_params: Parameters for mask extension
    """
    print(f"Creating Monte Carlo average point cloud from multiple frames...")
    
    # Create an empty combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_points = []
    combined_colors = []
    
    # Determine frames to process
    total_frames = len(reader)
    frames_to_process = range(total_frames) if num_frames is None else \
                        random.sample(range(total_frames), min(num_frames, total_frames))
    
    # Process each frame
    for i in frames_to_process:
        print(f"Processing frame {i+1}/{len(frames_to_process)}...")
        color = reader.get_color(i)
        xyz_map = reader.get_xyz_map(i)
        
        # Get masks for all objects
        masks = []
        try:
            for obj in object_names:
                masks.append(reader.get_mask(i, obj))
        except (IndexError, FileNotFoundError) as e:
            print(f"Warning: {e}. Skipping frame {i}.")
            continue
            
        # Process masks with extension if requested
        processed_masks = []
        for mask in masks:
            if extend_mask_params:
                mask = extend_mask(
                    mask, 
                    kernel_size=extend_mask_params['kernel_size'], 
                    method=extend_mask_params['method']
                )
            processed_masks.append(mask)
        
        # Combine all masks using logical OR
        combined_mask = np.logical_or.reduce(processed_masks)
        valid_mask = ~combined_mask
        
        # Extract valid 3D points and corresponding colors
        points = xyz_map[valid_mask]
        colors = color[valid_mask] / 255.0
        
        # Skip if no valid points
        if len(points) == 0:
            continue
            
        # Randomly sample points
        if sampling_ratio < 1.0:
            num_samples = max(1, int(len(points) * sampling_ratio))
            indices = np.random.choice(len(points), num_samples, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # Add to combined lists
        combined_points.append(points)
        combined_colors.append(colors)
    
    # Concatenate all points and colors
    if combined_points:
        all_points = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)
        
        # Create and save the combined point cloud
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # Save to file
        o3d.io.write_point_cloud(output_file, combined_pcd)
        print(f"Saved combined point cloud to {output_file}")
        return combined_pcd
    else:
        print("No valid points found across frames.")
        return None


if __name__ == "__main__":
    # python get_scene_pc.py --video_dir /home/balen/workspace/pc_recon/MeshRecon/datasets/new_kitchen_pick_bottle --object_names arm bottle --output_file new_pick_bottle_pc.ply
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate a masked-out point cloud.")
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Path to the video directory."
    )
    parser.add_argument(
        "--object_names",
        type=str,
        nargs="+",
        required=True,
        help="List of object names for masks.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output filename for the point cloud.",
    )
    parser.add_argument(
        "--monte_carlo_average",
        action="store_true",
        help="Combine point clouds from all frames using Monte Carlo averaging.",
    )
    parser.add_argument(
        "--mc_sampling_ratio",
        type=float,
        default=0.1,
        help="Ratio of points to sample from each frame for Monte Carlo averaging.",
    )
    parser.add_argument(
        "--mc_num_frames",
        type=int,
        default=None,
        help="Number of frames to use for Monte Carlo averaging (default: all frames).",
    )
    parser.add_argument(
        "--extend_mask",
        action="store_true",
        help="Use extended masking with convolution.",
    )
    parser.add_argument(
        "--mask_kernel_size",
        type=int,
        default=5,
        help="Kernel size for mask extension (default: 5).",
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        choices=["max", "mode"],
        default="max",
        help="Method for mask extension: 'max' or 'mode' (default: max).",
    )
    
    args = parser.parse_args()
    video_dir = Path(args.video_dir)
    object_names = args.object_names
    output_file = args.output_file
    
    # Initialize the reader
    reader = MegaSamReader(video_dir, object_names=object_names)
    
    # Set up mask extension parameters if enabled
    extend_mask_params = None
    if args.extend_mask:
        extend_mask_params = {
            'kernel_size': args.mask_kernel_size,
            'method': args.mask_method
        }
    
    # Process based on selected method
    if args.monte_carlo_average:
        # Process multiple frames with Monte Carlo averaging
        monte_carlo_average_point_clouds(
            reader,
            object_names,
            output_file,
            num_frames=args.mc_num_frames,
            sampling_ratio=args.mc_sampling_ratio,
            extend_mask_params=extend_mask_params
        )
    else:
        # Process single frame (original functionality)
        color = reader.get_color(0)  # [H, W, 3]
        xyz_map = reader.get_xyz_map(0)  # [H, W, 3]
        masks = [reader.get_mask(0, obj) for obj in object_names]  # [h, w] * len
        save_point_cloud(xyz_map, color, masks, output_file, extend_mask_params=extend_mask_params)
