#include "output_helper.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/IO/polygon_mesh_io.h>

#include <iostream>
#include <string>
#include <utility>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <fstream>

// Types
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Vector = Kernel::Vector_3;
using Pwn = std::pair<Point, Vector>;  // Point with normal
using Polyhedron = CGAL::Polyhedron_3<Kernel>;
using Mesh = CGAL::Surface_mesh<Point>;

// Function to convert string to lowercase
std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

// Function to get file extension
std::string get_extension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
        return to_lower(filename.substr(pos + 1));
    }
    return "";
}

int main(int argc, char** argv)
{
    // Parse command line arguments
    std::string filename;
    double sm_angle = 20.0;       // Min triangle angle in degrees
    double sm_radius = 30.0;      // Max triangle size w.r.t. point set average spacing
    double sm_distance = 0.375;   // Surface approximation error w.r.t. point set average spacing
    std::string output_dir = "output";
    
    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] <input_point_cloud>\n"
                      << "Options:\n"
                      << "  --angle <value>     Set min triangle angle in degrees (default: 20.0)\n"
                      << "  --radius <value>    Set max triangle size w.r.t. average spacing (default: 30.0)\n"
                      << "  --distance <value>  Set surface approximation error w.r.t. average spacing (default: 0.375)\n"
                      << "  --outdir <dir>      Set output directory (default: 'output')\n"
                      << "  --help, -h          Show this help message\n";
            return EXIT_SUCCESS;
        } else if (arg == "--angle" && i + 1 < argc) {
            sm_angle = std::stod(argv[++i]);
        } else if (arg == "--radius" && i + 1 < argc) {
            sm_radius = std::stod(argv[++i]);
        } else if (arg == "--distance" && i + 1 < argc) {
            sm_distance = std::stod(argv[++i]);
        } else if (arg == "--outdir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (filename.empty()) {
            filename = arg;
        }
    }
    
    // Check if input file was provided
    if (filename.empty()) {
        std::cerr << "Error: No input file specified. Use --help for usage information." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Reading " << filename << "..." << std::endl;
    
    // Check file extension
    std::string extension = get_extension(filename);
    if (extension != "ply" && extension != "xyz" && extension != "off") {
        std::cerr << "Error: Input file must be a .ply, .xyz, or .off file." << std::endl;
        return EXIT_FAILURE;
    }
    
    // Read the point set file
    std::vector<Pwn> points;
    if (!CGAL::IO::read_points(
            filename, 
            std::back_inserter(points),
            CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>())
                            .normal_map(CGAL::Second_of_pair_property_map<Pwn>())))
    {
        std::cerr << "Error: Cannot read point cloud from " << filename << std::endl;
        return EXIT_FAILURE;
    }
    
    // Check if we have points
    if (points.empty()) {
        std::cerr << "Error: No points found in the input file." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Input point cloud has " << points.size() << " points with normals." << std::endl;
    
    // Compute average spacing
    double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(
        points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()));
    std::cout << "Average spacing: " << average_spacing << std::endl;
    
    // Create the output mesh
    Polyhedron output_mesh;
    
    // Poisson reconstruction
    std::cout << "Reconstructing surface..." << std::endl;
    
    // Set up message with parameters
    std::cout << "Parameters: angle=" << sm_angle << ", radius=" << sm_radius 
              << ", distance=" << sm_distance << std::endl;
    
    bool success = CGAL::poisson_surface_reconstruction_delaunay(
        points.begin(), points.end(),
        CGAL::First_of_pair_property_map<Pwn>(),
        CGAL::Second_of_pair_property_map<Pwn>(),
        output_mesh,
        average_spacing);
    
    if (!success) {
        std::cerr << "Error: Surface reconstruction failed." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Reconstruction completed successfully." << std::endl;
    std::cout << "Result: " << output_mesh.size_of_vertices() << " vertices, " 
              << output_mesh.size_of_facets() << " faces" << std::endl;
    
    // Save the result
    std::string output_name = generate_output_name(filename, sm_angle, sm_radius, output_dir);
    std::cout << "Writing to " << output_name << std::endl;
    
    std::ofstream out(output_name);
    if (!out) {
        std::cerr << "Error: Cannot open output file " << output_name << std::endl;
        return EXIT_FAILURE;
    }
    
    out << output_mesh;
    out.close();
    
    std::cout << "Reconstruction process completed." << std::endl;
    
    return EXIT_SUCCESS;
}