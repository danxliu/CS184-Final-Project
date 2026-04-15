cmake_minimum_required(VERSION 3.15)
project(RepulsiveShells CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(OpenMesh REQUIRED)
add_subdirectory(ext/nanogui)

add_executable(RepulsiveShells src/main.cpp src/MeshViewer.cpp)

target_include_directories(RepulsiveShells PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}
    ${OPENMESH_INCLUDE_DIRS}
    ${OPENMESH_INCLUDE_DIR}
    ${OpenMesh_INCLUDE_DIR}
)

target_link_libraries(RepulsiveShells PRIVATE 
    OpenMeshCore
    OpenMeshTools
    nanogui
)
