cmake_minimum_required(VERSION 3.28)
project(AlphaWrap)

find_package(CGAL REQUIRED COMPONENTS Core)

# Use C++17 if available
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp)

# Link CGAL and system libraries
target_link_libraries(${PROJECT_NAME} CGAL::CGAL CGAL::CGAL_Core)