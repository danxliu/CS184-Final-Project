cmake_minimum_required(VERSION 3.15)
project(RepulsiveShells CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -- dependencies -------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(OpenMesh REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# GLFW subprojects cache these as absolute paths on the Nix glibc in some
# mixed host/devenv shells. Keep them as regular linker flags so viewer
# executables do not get a Nix-glibc runtime rpath mixed with host libstdc++.
set(RT_LIBRARY rt CACHE STRING "" FORCE)
set(MATH_LIBRARY m CACHE STRING "" FORCE)

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
    src/core/ShellEnergy.cpp
    src/core/PathEnergy.cpp
    src/core/TrustRegionSolver.cpp
    src/core/ExtrapolationSolver.cpp
    src/core/Constraints.cpp
    src/core/Obstacle.cpp
    src/core/Remesh.cpp
    src/core/OptimizeTPE.cpp
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
    OpenMP::OpenMP_CXX
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

# -- Polyscope viewer (optional) ---------------------------------------------
option(RSH_ENABLE_POLYSCOPE "Build Polyscope-based viewer" OFF)
if(RSH_ENABLE_POLYSCOPE)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/ext/polyscope/CMakeLists.txt")
        message(FATAL_ERROR
            "RSH_ENABLE_POLYSCOPE=ON but ext/polyscope is missing. "
            "Run `git submodule update --init --recursive`.")
    endif()
    # Keep the opt-in viewer on the GLFW backend; do not require EGL headers in
    # the default Nix/dev shell.
    set(POLYSCOPE_BACKEND_OPENGL3_EGL OFF CACHE STRING "" FORCE)
    add_subdirectory(ext/polyscope)

    add_executable(polyscope_viewer tools/polyscope_viewer.cpp)
    target_link_libraries(polyscope_viewer PRIVATE rsh_core polyscope)
    file(GLOB RSH_DBUS_CANDIDATES
        "/nix/store/*dbus*-lib/lib/libdbus-1.so"
        "/nix/store/*dbus*/lib/libdbus-1.so")
    if(RSH_DBUS_CANDIDATES)
        list(SORT RSH_DBUS_CANDIDATES)
        list(GET RSH_DBUS_CANDIDATES 0 RSH_DBUS_LIBRARY)
        target_link_libraries(polyscope_viewer PRIVATE ${RSH_DBUS_LIBRARY})
    endif()
endif()

# -- test / utility binaries --------------------------------------------------
add_executable(test_phase0 tests/test_phase0.cpp)
target_link_libraries(test_phase0 PRIVATE rsh_core)

add_executable(test_phase1 tests/test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE rsh_core)

add_executable(test_phase2 tests/test_phase2.cpp)
target_link_libraries(test_phase2 PRIVATE rsh_core)

add_executable(test_phase3 tests/test_phase3.cpp)
target_link_libraries(test_phase3 PRIVATE rsh_core)

add_executable(test_hs_kernel_exponent tests/test_hs_kernel_exponent.cpp)
target_link_libraries(test_hs_kernel_exponent PRIVATE rsh_core)

add_executable(dump_test_meshes tests/dump_test_meshes.cpp)
target_link_libraries(dump_test_meshes PRIVATE rsh_core)

add_executable(demo_phase1_flow tests/demo_phase1_flow.cpp)
target_link_libraries(demo_phase1_flow PRIVATE rsh_core)

add_executable(demo_phase2_hs_flow tests/demo_phase2_hs_flow.cpp)
target_link_libraries(demo_phase2_hs_flow PRIVATE rsh_core)

add_executable(demo_canonical_genus0 tests/demo_canonical_genus0.cpp)
target_link_libraries(demo_canonical_genus0 PRIVATE rsh_core)

add_executable(demo_gallery_genus0 tests/demo_gallery_genus0.cpp)
target_link_libraries(demo_gallery_genus0 PRIVATE rsh_core)

add_executable(demo_gallery_genus1 tests/demo_gallery_genus1.cpp)
target_link_libraries(demo_gallery_genus1 PRIVATE rsh_core)

function(add_connected_gallery_target GENUS)
    set(TARGET_NAME "demo_gallery_genus${GENUS}")
    add_executable(${TARGET_NAME} tests/demo_gallery_genus2.cpp)
    target_link_libraries(${TARGET_NAME} PRIVATE rsh_core)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        RSH_GALLERY_DEFAULT_GENUS=${GENUS})
endfunction()

add_connected_gallery_target(2)
add_connected_gallery_target(3)
add_connected_gallery_target(4)
add_connected_gallery_target(5)

add_custom_target(gallery_genus0 DEPENDS demo_gallery_genus0)
add_custom_target(gallery_genus1 DEPENDS demo_gallery_genus1)
add_custom_target(gallery_genus2 DEPENDS demo_gallery_genus2)
add_custom_target(gallery_genus3 DEPENDS demo_gallery_genus3)
add_custom_target(gallery_genus4 DEPENDS demo_gallery_genus4)
add_custom_target(gallery_genus5 DEPENDS demo_gallery_genus5)

add_executable(diagnose_bh_cancellation tools/diagnose_bh_cancellation.cpp)
target_link_libraries(diagnose_bh_cancellation PRIVATE rsh_core)

add_executable(diagnose_bh_leaf tools/diagnose_bh_leaf.cpp)
target_link_libraries(diagnose_bh_leaf PRIVATE rsh_core)

# add_executable(demo_interpolation tests/demo_interpolation.cpp)
# target_link_libraries(demo_interpolation PRIVATE rsh_core)

add_executable(demo_compression tests/demo_compression.cpp)
target_link_libraries(demo_compression PRIVATE rsh_core)

add_executable(bench_openmp_optimize tests/bench_openmp_optimize.cpp)
target_link_libraries(bench_openmp_optimize PRIVATE rsh_core)

# add_executable(demo_extrapolation tests/demo_extrapolation.cpp)
# target_link_libraries(demo_extrapolation PRIVATE rsh_core)

# -- opt-in external validation ----------------------------------------------
# Repulsor is a sibling checkout used only for cross-validation; keep it out of
# the normal build so teammates without ../Repulsor are not blocked.
option(RSH_HAVE_REPULSOR "Build Repulsor cross-validation tool" OFF)
if(RSH_HAVE_REPULSOR)
    include(CheckCXXSourceCompiles)

    get_filename_component(RSH_REPULSOR_DIR
        "${CMAKE_SOURCE_DIR}/../Repulsor" ABSOLUTE)
    if(NOT EXISTS "${RSH_REPULSOR_DIR}/Repulsor.hpp")
        message(FATAL_ERROR
            "RSH_HAVE_REPULSOR=ON but ${RSH_REPULSOR_DIR}/Repulsor.hpp "
            "does not exist")
    endif()
    if(NOT EXISTS "${RSH_REPULSOR_DIR}/submodules/Tensors/OpenBLAS.hpp")
        message(FATAL_ERROR
            "RSH_HAVE_REPULSOR=ON but Repulsor submodules are missing. "
            "Run `git submodule update --init --recursive` in ../Repulsor.")
    endif()

    find_path(RSH_REPULSOR_CBLAS_INCLUDE_DIR cblas.h
        PATHS /usr/include /usr/include/x86_64-linux-gnu
        PATH_SUFFIXES x86_64-linux-gnu)
    find_path(RSH_REPULSOR_LAPACKE_INCLUDE_DIR lapacke.h
        PATHS
            ${EIGEN3_INCLUDE_DIRS}/Eigen/src/misc
            ${EIGEN3_INCLUDE_DIRS}
        PATH_SUFFIXES Eigen/src/misc)
    find_library(RSH_REPULSOR_OPENBLAS_LIBRARY NAMES openblas
        PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib /lib/x86_64-linux-gnu)
    if(NOT RSH_REPULSOR_OPENBLAS_LIBRARY)
        file(GLOB RSH_REPULSOR_OPENBLAS_CANDIDATES
            "/nix/store/*openblas*/lib/libopenblas.so")
        if(RSH_REPULSOR_OPENBLAS_CANDIDATES)
            list(SORT RSH_REPULSOR_OPENBLAS_CANDIDATES)
            list(GET RSH_REPULSOR_OPENBLAS_CANDIDATES 0
                RSH_REPULSOR_OPENBLAS_LIBRARY)
        endif()
    endif()
    find_library(RSH_REPULSOR_BLAS_LIBRARY NAMES blas
        PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib /lib/x86_64-linux-gnu)
    find_library(RSH_REPULSOR_LAPACK_LIBRARY NAMES lapack
        PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib /lib/x86_64-linux-gnu)

    if(NOT RSH_REPULSOR_CBLAS_INCLUDE_DIR)
        message(FATAL_ERROR "RSH_HAVE_REPULSOR=ON but cblas.h was not found")
    endif()
    if(NOT RSH_REPULSOR_LAPACKE_INCLUDE_DIR)
        message(FATAL_ERROR "RSH_HAVE_REPULSOR=ON but lapacke.h was not found")
    endif()
    if(RSH_REPULSOR_OPENBLAS_LIBRARY)
        set(RSH_REPULSOR_BLAS_LIBRARIES ${RSH_REPULSOR_OPENBLAS_LIBRARY})
    elseif(RSH_REPULSOR_BLAS_LIBRARY AND RSH_REPULSOR_LAPACK_LIBRARY)
        set(RSH_REPULSOR_BLAS_LIBRARIES
            ${RSH_REPULSOR_BLAS_LIBRARY}
            ${RSH_REPULSOR_LAPACK_LIBRARY})
    else()
        message(FATAL_ERROR
            "RSH_HAVE_REPULSOR=ON but neither OpenBLAS nor BLAS+LAPACK "
            "libraries were found")
    endif()

    check_cxx_source_compiles(
        "#include \"${RSH_REPULSOR_CBLAS_INCLUDE_DIR}/cblas.h\"
         int main() { blasint x = 0; return static_cast<int>(x); }"
        RSH_REPULSOR_CBLAS_HAS_BLASINT)

    add_executable(cross_validate_repulsor
        tools/cross_validate_repulsor.cpp)
    set_property(TARGET cross_validate_repulsor PROPERTY CXX_STANDARD 20)
    target_include_directories(cross_validate_repulsor PRIVATE
        ${RSH_REPULSOR_DIR})
    target_compile_definitions(cross_validate_repulsor PRIVATE
        RSH_REPULSOR_CBLAS_HEADER="${RSH_REPULSOR_CBLAS_INCLUDE_DIR}/cblas.h"
        RSH_REPULSOR_LAPACKE_HEADER="${RSH_REPULSOR_LAPACKE_INCLUDE_DIR}/lapacke.h")
    target_link_libraries(cross_validate_repulsor PRIVATE
        rsh_core
        ${RSH_REPULSOR_BLAS_LIBRARIES}
        Threads::Threads)
    if(NOT RSH_REPULSOR_CBLAS_HAS_BLASINT)
        target_compile_definitions(cross_validate_repulsor PRIVATE
            RSH_REPULSOR_NEEDS_BLASINT_ALIAS=1)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(cross_validate_repulsor PRIVATE -fpermissive)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(cross_validate_repulsor PRIVATE -fenable-matrix)
    endif()
endif()
