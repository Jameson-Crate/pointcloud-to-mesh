# 3D Mesh and Point Cloud Processing Tools

This repository contains a collection of Python scripts for processing 3D meshes and point clouds, with a focus on texture mapping and coordinate system conversions.

## Scripts

### texture_map.py
Maps colors from a point cloud to a mesh using nearest neighbor interpolation.

**Usage:**
```
python texture_map.py pointcloud.ply mesh.obj --output textured_mesh.obj --max_distance 0.01 --neighbors 3
```

**Options:**
- `--output`, `-o`: Path to save the textured mesh
- `--max_distance`: Maximum distance between mesh and point cloud points for color mapping
- `--neighbors`: Number of neighbors to consider for interpolation
- `--upsample`: Upsample mesh to match a percentage of the point cloud size
- `--fix-rotation`: Fix coordinate system rotation when saving to OBJ/GLB formats

### complex_texture_map.py
Advanced texture mapping script with multiple mapping methods and barycentric coordinate support.

**Usage:**
```
python complex_texture_map.py pointcloud.ply mesh.obj --output textured_mesh.obj --method weighted
```

**Options:**
- `--output`, `-o`: Path to save the textured mesh
- `--max_distance`: Maximum distance for color transfer
- `--method`: Method for texture mapping ('nearest', 'weighted', or 'alternative')
- `--fix-rotation`: Fix coordinate system rotation when saving to OBJ/GLB formats

### convert_mesh.py
Converts mesh files between various formats with proper coordinate system transformations.

**Usage:**
```
python convert_mesh.py input_mesh.ply -o output_mesh.obj -f obj
```

**Options:**
- `-o`, `--output`: Output mesh file or directory
- `-f`, `--format`: Output format (e.g., glb, obj, stl, ply)
- `-r`, `--recursive`: Process directories recursively
- `-t`, `--types`: Comma-separated list of mesh file extensions to process

### change_convention.py
Rotates a point cloud 180 degrees around the X-axis to convert between different coordinate system conventions (COLMAP/OpenCV and OpenGL/Blender).

**Usage:**
```
python change_convention.py input_pointcloud.ply --output_file output_pointcloud.ply
```

**Options:**
- `--output_file`: Path to output pointcloud file. If not specified, will use input_file_rotated[.ext]

### convex_decomposition.py
Generates a convex decomposition from a mesh using PyBullet's VHACD algorithm.

**Usage:**
```
python convex_decomposition.py input_mesh.obj output_decomp.obj --max_hulls 20 --resolution 300000
```

**Options:**
- `--log`: Log file path (default: logs/vhacd_log.log)
- `--resolution`: Maximum number of voxels generated (default: 300000)
- `--max_hulls`, `--depth`: Maximum number of convex hulls / recursion depth (default: 20)
- `--concavity`: Maximum concavity (default: 0.0025)
- `--planeDownsampling`: Plane downsampling (default: 4)
- `--convexhullDownsampling`: Convex hull downsampling (default: 4)
- `--maxNumVerticesPerCH`: Maximum vertices per convex hull (default: 64)
- `--minVolumePerCH`: Minimum volume per convex hull (default: 0.0005)
- `--convexhullApproximation`: Convex hull approximation (default: 1)
- `--pca`: PCA (Principal Component Analysis) (default: 1)

### predict_normals.py
Estimates and adds normals to a point cloud.

**Usage:**
```
python predict_normals.py input_pointcloud.ply --output_file output_with_normals.ply
```

**Options:**
- `--output_file`: Path to output file. If not specified, will use input_file_normals[.ext]
- `--radius`: Search radius for normal estimation. If not specified, it will be estimated automatically
- `--max_nn`: Maximum number of nearest neighbors to use (default: 30)
- `--no_orient`: Disable consistent normal orientation
- `--visualize`: Visualize the point cloud with normals

### single_preprocess.py
Preprocesses a single point cloud file to prepare it for mesh reconstruction.

**Usage:**
```
python single_preprocess.py input_pointcloud.ply --output_file processed.ply
```

### pointcloud_preprocess.py
Batch preprocessing for point clouds to prepare them for mesh reconstruction.

**Usage:**
```
python pointcloud_preprocess.py input_directory --output_dir processed_clouds
```

### ransac_preprocess.py
Preprocesses point clouds using RANSAC for plane segmentation.

**Usage:**
```
python ransac_preprocess.py input_pointcloud.ply --output_file processed.ply
```

### downsample_preprocess.py
Downsamples point clouds to reduce size while preserving structure.

**Usage:**
```
python downsample_preprocess.py input_pointcloud.ply --output_file downsampled.ply --voxel_size 0.05
```
