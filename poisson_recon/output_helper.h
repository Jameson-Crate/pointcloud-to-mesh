#ifndef OUTPUT_HELPER_H
#define OUTPUT_HELPER_H

#include <string>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <libgen.h>
    #include <cstring>
    #include <unistd.h>

    // Simple directory creation for systems without std::filesystem
    inline void create_directory(const std::string& path) {
        #ifdef _WIN32
            _mkdir(path.c_str());
        #else
            mkdir(path.c_str(), 0755);
        #endif
    }

    // Extract filename without extension
    inline std::string get_stem(const std::string& path) {
        char* path_copy = strdup(path.c_str());
        char* filename = basename(path_copy);
        std::string name(filename);
        free(path_copy);
        
        size_t last_dot = name.find_last_of('.');
        if (last_dot != std::string::npos) {
            return name.substr(0, last_dot);
        }
        return name;
    }
#endif

// Generate an output filename based on the input file and parameters
inline std::string generate_output_name(const std::string& input_filename, 
                                       double relative_alpha, 
                                       double relative_offset,
                                       const std::string& output_dir = "output") {
    // Create output directory if it doesn't exist
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    fs::path output_path(output_dir);
    if (!fs::exists(output_path)) {
        fs::create_directories(output_path);
    }

    // Extract the base filename without extension
    fs::path input_path(input_filename);
    std::string base_name = input_path.stem().string();
#else
    // Create directory if it doesn't exist
    create_directory(output_dir);
    
    // Get the stem (filename without extension)
    std::string base_name = get_stem(input_filename);
#endif
    
    // Create the output filename with parameters
    std::ostringstream oss;
    oss << output_dir << "/" << base_name 
        << "_alpha" << std::fixed << std::setprecision(1) << relative_alpha
        << "_offset" << std::fixed << std::setprecision(1) << relative_offset
        << ".off"; // Always use .off for the output format
    
    return oss.str();
}

#endif // OUTPUT_HELPER_H 