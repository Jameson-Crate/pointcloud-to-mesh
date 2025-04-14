import json
import argparse
import glob
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = (depth < 0.001)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W),
                             sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N, 3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

def extend_mask(mask, kernel_size=5, method='max'):
    """
    Extend the edges of a mask using convolution.
    
    Args:
        mask (numpy.ndarray): Binary mask to extend [H, W].
        kernel_size (int): Size of the convolution kernel.
        method (str): Method for extension ('max', 'mode').
        
    Returns:
        numpy.ndarray: Extended mask.
    """
    if method not in ['max', 'mode']:
        raise ValueError("Method must be 'max' or 'mode'")
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if method == 'max':
        # Dilate the mask using a max operation
        extended_mask = cv2.dilate(mask, kernel, iterations=1)
    else:  # mode
        # Use a mode filter (most common value in neighborhood)
        extended_mask = mask.copy()
        pad_size = kernel_size // 2
        padded = np.pad(mask, pad_size, mode='constant')
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                values, counts = np.unique(window, return_counts=True)
                extended_mask[i, j] = values[np.argmax(counts)]
    
    return extended_mask

class MegaSamReader:
    """
    Used to read sgd_cvd_hr.npz and load corresponding segmentation masks based on the given object name list.
    """
    def __init__(self, video_dir, downscale=1, shorter_side=None, zfar=np.inf, object_names=[]):
        self.video_dir = Path(video_dir)
        self.file_name = "sgd_cvd_hr.npz"
        data_path = self.video_dir / self.file_name
        if not data_path.exists():
            raise FileNotFoundError(f"sgd_cvd_hr.npz file not found at {data_path}.")
        data = np.load(str(data_path))
        self.downscale = downscale
        self.zfar = zfar
        # Read color / depth / camera parameters / poses
        self.color_files = data['images']   # (N, H, W, 3) uint8
        self.depth_files = data['depths']   # (N, H, W) float16
        self.K = data['intrinsic'].astype(np.float64)  # (3, 3) float32
        self.camera_poses = data['cam_c2w'] # (N, 4, 4) float32
        # Get corresponding object segmentation masks from the file system
        self.masks = {}
        for object_name in object_names:
            # Assume mask filenames are in the format segment_{object_name}*.png
            mask_pattern = f"{self.video_dir}/masks/segment_{object_name}*.png"
            self.masks[object_name] = sorted(glob.glob(mask_pattern))
        self.id_strs = [f"{idx:04d}" for idx in range(len(self.color_files))]
        self.H, self.W = self.color_files.shape[1:3]
        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)
        # In this example, no scaling is applied, just check
        assert self.downscale == 1

    def get_video_name(self):
        return self.video_dir.name
    
    def __len__(self):
        return len(self.color_files)
    
    def get_color(self, i):
        color = self.color_files[i]
        # Scaling operation (here it's 1, so size doesn't change)
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

    def get_mask(self, i, object_name, extend_mask_kernel=None, extend_mask_method='max'):
        if object_name not in self.masks:
            raise ValueError(f"Object '{object_name}' not found in mask list.")
        
        # Ensure we have the correct mask for the frame index
        expected_mask_filename = f"{self.video_dir}/masks/segment_{object_name}_{i:04d}.png"
        mask_path = Path(expected_mask_filename)
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found for frame {i}: {expected_mask_filename}")
        
        mask = cv2.imread(str(mask_path), -1)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask file: {mask_path}")
        
        # If it's a three-channel image, take the first channel with content
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
        # Apply mask extension if specified
        if extend_mask_kernel is not None and extend_mask_kernel > 0:
            mask = extend_mask(mask, extend_mask_kernel, extend_mask_method)
            
        return mask

def save_point_cloud(xyz_map, color, masks, output_file="point_cloud.ply"):
    """
    Save a point cloud as .ply after removing specified objects.
    Args:
        xyz_map (numpy.ndarray): [H, W, 3] 3D coordinates.
        color (numpy.ndarray): [H, W, 3] image colors.
        masks (list of numpy.ndarray): Masks for each target object [H, W].
        output_file (str): Output filename.
    """
    if not masks:
        raise ValueError("Please provide at least one mask.")
    # Combine multiple masks with logical OR to represent all areas to be removed
    combined_mask = np.logical_or.reduce(masks)
    valid_mask = ~combined_mask  # Only keep areas that are not masked
    # Extract valid points and colors
    points = xyz_map[valid_mask]
    colors = color[valid_mask] / 255.0  # Convert to [0,1] range
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Write PLY file
    o3d.io.write_point_cloud(str(output_file), pcd)
    print(f"Saved point cloud to {output_file}")

def combine_point_clouds(reader, frame_indices, object_names, sampling_factor=None, 
                         extend_mask_kernel=None, extend_mask_method='max'):
    """
    Combine multiple frames into a single downsampled point cloud.
    
    Args:
        reader (MegaSamReader): The data reader.
        frame_indices (list): Indices of frames to combine.
        object_names (list): Names of objects to exclude.
        sampling_factor (float, optional): Sampling factor for each frame (0-1). If None, defaults to 1/len(frame_indices).
        extend_mask_kernel (int, optional): Kernel size for mask extension.
        extend_mask_method (str): Method for mask extension ('max', 'mode').
        
    Returns:
        open3d.geometry.PointCloud: The combined point cloud.
    """
    combined_pcd = o3d.geometry.PointCloud()
    
    # Set default sampling rate based on number of frames if not specified
    if sampling_factor is None:
        sampling_factor = 1.0 / len(frame_indices) if len(frame_indices) > 0 else 1.0
    
    for idx in tqdm(frame_indices):
        # Get data for this frame
        xyz_map = reader.get_xyz_map(idx)
        color = reader.get_color(idx)
        # Get masks and ensure they are 2D
        masks = []
        for obj in object_names:
            mask = reader.get_mask(idx, obj, extend_mask_kernel, extend_mask_method)
            # Ensure mask is 2D
            if len(mask.shape) > 2:
                # If mask has more dimensions, convert to 2D binary mask
                mask = (mask > 0).astype(np.uint8)
                if len(mask.shape) > 2:
                    # If still not 2D, take the first channel or sum across channels
                    mask = mask[..., 0] if mask.shape[-1] > 0 else np.sum(mask, axis=-1) > 0
            masks.append(mask)
        
        # Skip if we have no valid masks
        if not masks:
            continue
            
        # Combine masks to get a single removal mask
        combined_mask = np.logical_or.reduce(masks)
        valid_mask = ~combined_mask
        
        # Extract valid points and colors
        # Reshape the mask to match xyz_map dimensions if needed
        if valid_mask.ndim == 2 and xyz_map.ndim == 3:
            # Get indices of valid points
            valid_indices = np.where(valid_mask)
            points = xyz_map[valid_indices]
            colors = color[valid_indices] / 255.0
        else:
            # Fallback to original approach
            points = xyz_map[valid_mask]
            colors = color[valid_mask] / 255.0
        
        # Skip this frame if we have no valid points
        if len(points) == 0:
            continue
        
        # Create point cloud for this frame
        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(points)
        frame_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Downsample this frame's point cloud before combining
        num_points = len(frame_pcd.points)
        if num_points > 0 and sampling_factor < 1.0:
            target_num = max(1, int(num_points * sampling_factor))
            frame_pcd = frame_pcd.random_down_sample(target_num / num_points)
        
        # Combine with the accumulated point cloud
        combined_pcd += frame_pcd
    
    return combined_pcd

def main():
    """
    Usage:
    python pointcloud_preprocess.py --parent_dir /home/.../new_kitchen_pick [--combine_frames] 
                                    [--frame_step 1] [--sampling_factor 0.1] 
                                    [--extend_mask_kernel 5] [--extend_mask_method max]
    Logic:
      - Iterate through all subdirectories under parent_dir
      - For each subdirectory:
         * When sgd_cvd_hr.npz + config.json exist, read objects_label from config.json
         * Use MegaSamReader to load images, depths and masks
         * Generate point cloud after removing all objects in objects_label
         * Output to <subdirectory>/point_clouds/scene.ply
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, required=True,
                        help="include multiple parent folders, each child folder has sgd_cvd_hr.npz and config.json.")
    parser.add_argument("--combine_frames", action="store_true",
                        help="Combine multiple frames into a single point cloud")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Step size for frame sampling when combining frames")
    parser.add_argument("--sampling_factor", type=float, default=None,
                        help="Sampling factor for each frame (0-1). If None, defaults to 1/num_frames.")
    parser.add_argument("--extend_mask_kernel", type=int, default=None,
                        help="Kernel size for extending object masks (if not specified, no extension)")
    parser.add_argument("--extend_mask_method", type=str, default="max", choices=["max", "mode"],
                        help="Method for extending mask edges ('max' or 'mode')")
    
    args = parser.parse_args()
    parent_path = Path(args.parent_dir)
    if not parent_path.is_dir():
        print(f"The given path {parent_path} is not a directory.")
        return
    
    # Iterate through all subdirectories under the parent directory
    for subfolder in parent_path.iterdir():
        if not subfolder.is_dir():
            continue
        npz_file = subfolder / "sgd_cvd_hr.npz"
        json_file = subfolder / "config.json"
        if not npz_file.exists() or not json_file.exists():
            # Skip if the folder doesn't have sgd_cvd_hr.npz or config.json
            continue
        
        # Read objects_label from JSON file as targets to remove
        with open(json_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        object_names = config_data.get("objects", {}).get("labels", [])
        if not object_names:
            print(f"[{subfolder.name}] No objects.labels found in config.json, skipping.")
            continue
        
        # Construct output file path
        meshes_dir = subfolder / "point_clouds"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        output_file = meshes_dir / "scene.ply"
        
        try:
            # Initialize reader
            reader = MegaSamReader(subfolder, object_names=object_names)
            
            if args.combine_frames:
                # Process multiple frames and combine them
                print(f"[{subfolder.name}] Combining frames into a single point cloud...")
                frame_indices = list(range(0, len(reader), args.frame_step))
                if not frame_indices:
                    frame_indices = [0]  # Fallback to just frame 0
                
                # Create combined point cloud
                combined_pcd = combine_point_clouds(
                    reader, 
                    frame_indices, 
                    object_names,
                    sampling_factor=args.sampling_factor,
                    extend_mask_kernel=args.extend_mask_kernel,
                    extend_mask_method=args.extend_mask_method
                )
                
                # Save the combined point cloud
                o3d.io.write_point_cloud(str(output_file), combined_pcd)
                print(f"[{subfolder.name}] Combined {len(frame_indices)} frames into point cloud: {output_file}")
                
            else:
                # Process just frame 0
                color = reader.get_color(0)
                xyz_map = reader.get_xyz_map(0)
                masks = [reader.get_mask(0, obj, args.extend_mask_kernel, args.extend_mask_method) 
                         for obj in object_names]
                
                # Save point cloud after removing objects
                save_point_cloud(xyz_map, color, masks, output_file)
                print(f"[{subfolder.name}] Processing complete, output: {output_file}")
                
        except Exception as e:
            print(f"[{subfolder.name}] Error during processing: {e}")
            continue
        
if __name__ == "__main__":
    main()