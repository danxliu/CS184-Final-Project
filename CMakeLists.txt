cmake_minimum_required(VERSION 3.15)
project(RepulsiveShells CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -- dependencies -------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(OpenMesh REQUIRED)

# NanoGUI viewer is optional. Auto-enabled if the submodule has been
# initialized; force-disable with -DRSH_ENABLE_VIEWER=OFF for headless builds
# (CI, remote boxes without a GL stack, etc.).
option(RSH_ENABLE_VIEWER "Build the NanoGUI + OpenGL viewer" ON)
set(RSH_HAVE_NANOGUI OFF)
if(RSH_ENABLE_VIEWER AND EXISTS "${CMAKE_SOURCE_DIR}/ext/nanogui/CMakeLists.txt")
    # We only want the C++ NanoGUI library; skip the Python bindings and demos,
    # which pull in nanobind and don't always compile cleanly against the host
    # C++ toolchain (and we never invoke them).
    set(NANOGUI_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(NANOGUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NANOGUI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    add_subdirectory(ext/nanogui)
    set(RSH_HAVE_NANOGUI ON)
endif()

# -- rsh_core: the pipeline library ------------------------------------------
# Everything in src/core/ goes here. Test binaries and the viewer all link
# against rsh_core and pick up its include directory automatically.
set(RSH_CORE_SOURCES
    src/core/MeshData.cpp
    src/core/GradCheck.cpp
    src/core/TestMeshes.cpp
    src/core/FaceGeom.cpp
    src/core/TPE.cpp
    src/core/BVH.cpp
    src/core/BCT.cpp
    src/core/HsPreconditioner.cpp
)

add_library(rsh_core STATIC ${RSH_CORE_SOURCES})
target_include_directories(rsh_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/core
    ${EIGEN3_INCLUDE_DIRS}
    ${OPENMESH_INCLUDE_DIRS}
    ${OPENMESH_INCLUDE_DIR}
    ${OpenMesh_INCLUDE_DIR}
)
target_link_libraries(rsh_core PUBLIC
    OpenMeshCore
    OpenMeshTools
)

# -- viewer (NanoGUI + OpenGL) ------------------------------------------------
if(RSH_HAVE_NANOGUI)
    add_executable(RepulsiveShells
        src/viewer/main.cpp
        src/viewer/MeshViewer.cpp
    )
    target_link_libraries(RepulsiveShells PRIVATE rsh_core nanogui)
else()
    message(STATUS
        "ext/nanogui not found; skipping viewer. "
        "Run `git submodule update --init --recursive` to enable it.")
endif()

# -- test / utility binaries --------------------------------------------------
add_executable(test_phase0 tests/test_phase0.cpp)
target_link_libraries(test_phase0 PRIVATE rsh_core)

add_executable(test_phase1 tests/test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE rsh_core)

add_executable(dump_test_meshes tests/dump_test_meshes.cpp)
target_link_libraries(dump_test_meshes PRIVATE rsh_core)

add_executable(demo_phase1_flow tests/demo_phase1_flow.cpp)
target_link_libraries(demo_phase1_flow PRIVATE rsh_core)

add_executable(demo_phase2_hs_flow tests/demo_phase2_hs_flow.cpp)
target_link_libraries(demo_phase2_hs_flow PRIVATE rsh_core)
