# Alpha Wrap Mesh Processing

This tool applies CGAL's alpha wrap algorithm to a 3D mesh, creating a watertight surface mesh.

## Description

Alpha wrapping is useful for creating clean, watertight meshes from potentially noisy, incomplete, or non-manifold input data. It's particularly useful for:
- Creating collision meshes for physics simulations
- Preparing meshes for 3D printing
- Simplifying complex geometries
- Filling holes in surfaces

## Prerequisites

- CMake (3.3 or higher)
- C++ compiler with C++17 support (recommended)
- CGAL library (with Core component)

## Building

```bash
mkdir -p alpha_wrap/build
cd alpha_wrap/build
cmake ..
make
```

## Usage

Run the program with an input mesh:

```bash
./AlphaWrap [options] <input_mesh>
```

### Options

- `--alpha <value>`: Sets the relative alpha value (default: 20.0)
- `--offset <value>`: Sets the relative offset value (default: 600.0)
- `--outdir <dir>`: Sets the output directory (default: 'output')
- `--help, -h`: Shows help message

### Supported Mesh Formats

The tool supports all formats recognized by CGAL, including:
- PLY (.ply)
- OBJ (.obj)
- OFF (.off)
- STL (.stl)
- GLTF/GLB (.gltf, .glb)

Output is always saved in the OFF format.

## Examples

```bash
# Basic usage
./AlphaWrap input.ply

# Custom alpha and offset values
./AlphaWrap --alpha 30 --offset 800 input.obj

# Custom output directory
./AlphaWrap --outdir results input.stl
```

## Parameter Explanation

- **Alpha**: Controls the size of details that can be captured. Smaller values capture more fine details but might be more sensitive to noise.
- **Offset**: Controls how far the alpha shape is offset. Larger values create a smoother output. 