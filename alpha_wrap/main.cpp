#include "output_helper.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Real_timer.h>

#include <iostream>
#include <string>

#include <cstdlib>

namespace PMP = CGAL::Polygon_mesh_processing;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;

using Mesh = CGAL::Surface_mesh<Point_3>;

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
    double relative_alpha = 20.0;
    double relative_offset = 600.0;
    std::string output_dir = "output";
    
    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] <input_mesh>\n"
                      << "Options:\n"
                      << "  --alpha <value>     Set relative alpha value (default: 20.0)\n"
                      << "  --offset <value>    Set relative offset value (default: 600.0)\n"
                      << "  --outdir <dir>      Set output directory (default: 'output')\n"
                      << "  --help, -h          Show this help message\n";
            return EXIT_SUCCESS;
        } else if (arg == "--alpha" && i + 1 < argc) {
            relative_alpha = std::stod(argv[++i]);
        } else if (arg == "--offset" && i + 1 < argc) {
            relative_offset = std::stod(argv[++i]);
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
    
    // Load the mesh based on its extension
    Mesh mesh;
    if (!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh)) {
        std::cerr << "Invalid input: " << filename << std::endl;
        return EXIT_FAILURE;
    }
    
    // Ensure the mesh is triangulated
    if (!is_triangle_mesh(mesh)) {
        std::cerr << "Input mesh is not triangulated. Please triangulate the mesh first." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Input: " << num_vertices(mesh) << " vertices, " << num_faces(mesh) << " faces" << std::endl;
    
    // Compute the alpha and offset values
    CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(mesh);
    const double diag_length = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                        CGAL::square(bbox.ymax() - bbox.ymin()) +
                                        CGAL::square(bbox.zmax() - bbox.zmin()));
    
    const double alpha = diag_length / relative_alpha;
    const double offset = diag_length / relative_offset;
    std::cout << "alpha: " << alpha << ", offset: " << offset << std::endl;
    
    // Construct the wrap
    CGAL::Real_timer t;
    t.start();
    
    Mesh wrap;
    CGAL::alpha_wrap_3(mesh, alpha, offset, wrap);
    
    t.stop();
    std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces" << std::endl;
    std::cout << "Took " << t.time() << " s." << std::endl;
    
    // Save the result
    const std::string output_name = generate_output_name(filename, relative_alpha, relative_offset, output_dir);
    std::cout << "Writing to " << output_name << std::endl;
    CGAL::IO::write_polygon_mesh(output_name, wrap, CGAL::parameters::stream_precision(17));
    
    return EXIT_SUCCESS;
}