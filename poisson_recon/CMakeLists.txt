cmake_minimum_required(VERSION 3.3)
project(PoissonReconstruction)

# Find CGAL and its components
find_package(CGAL REQUIRED COMPONENTS Core)

# Boost is required by CGAL
find_package(Boost REQUIRED)

# Use C++17 if available
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp)

# Include directories
include_directories(${CGAL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Link CGAL and system libraries
target_link_libraries(${PROJECT_NAME} CGAL::CGAL CGAL::CGAL_Core)