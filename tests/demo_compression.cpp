#include "MeshData.h"
#include "PathEnergy.h"
#include "TrustRegionSolver.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace rsh;

MeshData merge_meshes(const MeshData& a, const MeshData& b) {
    MeshData out;
    out.V.resize(a.n_vertices() + b.n_vertices(), 3);
    out.V << a.V, b.V;
    
    Eigen::MatrixXi bF = b.F.array() + a.n_vertices();
    out.F.resize(a.n_faces() + b.n_faces(), 3);
    out.F << a.F, bF;
    return out;
}

int main(int argc, char** argv) {
    std::cout << "=== Compression Demo ===\n";
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh1.obj> [mesh2.obj]\n";
        return 1;
    }
    
    MeshData mesh1 = MeshData::load_obj(argv[1]);
    mesh1.normalize();
    
    MeshData mesh2 = (argc >= 3) ? MeshData::load_obj(argv[2]) : mesh1;
    if (argc >= 3) mesh2.normalize();
    
    // Initial configuration: mesh1 at x=-0.5, mesh2 at x=0.5
    MeshData m1_start = mesh1;
    m1_start.V.col(0).array() -= 0.5;
    
    MeshData m2_start = mesh2;
    m2_start.V.col(0).array() += 0.5;
    // Rotate mesh2 so they face each other
    for (int i = 0; i < m2_start.n_vertices(); ++i) {
        double x = m2_start.V(i, 0) - 0.5; // relative to center
        double z = m2_start.V(i, 2);
        m2_start.V(i, 0) = -x + 0.5;
        m2_start.V(i, 2) = -z;
    }
    
    MeshData x0 = merge_meshes(m1_start, m2_start);
    
    // Final configuration: they move towards each other
    MeshData m1_end = mesh1;
    m1_end.V.col(0).array() -= 0.1; // intersect
    
    MeshData m2_end = mesh2;
    for (int i = 0; i < m2_end.n_vertices(); ++i) {
        double x = m2_end.V(i, 0);
        double z = m2_end.V(i, 2);
        m2_end.V(i, 0) = -x;
        m2_end.V(i, 2) = -z;
    }
    m2_end.V.col(0).array() += 0.1;
    
    MeshData x_end = merge_meshes(m1_end, m2_end);
    
    const int num_frames = 6; 
    std::vector<MeshData> frames(num_frames, x0);
    
    // Linear initialization
    for (int i = 0; i < num_frames; ++i) {
        float t = (float)i / (num_frames - 1);
        frames[i].V = (1.0f - t) * x0.V + t * x_end.V;
    }
    
    // Set up free vertices
    // We want to pin the far ends of the meshes to force compression.
    // For mesh1, pin vertices with x < -0.7. For mesh2, pin vertices with x > 0.7.
    std::vector<bool> free_vertices(x0.n_vertices(), true);
    for (int i = 0; i < x0.n_vertices(); ++i) {
        if (x0.V(i, 0) < -0.7 || x0.V(i, 0) > 0.7) {
            free_vertices[i] = false;
        }
    }
    
    PathEnergyParams ep;
    ep.tpe_adaptive.enabled = true;
    ep.tpe_adaptive.max_depth = 2;
    ep.tpe_theta = 0.5; // Ensure BCT clustering is fine enough
    
    TrustRegionParams tp;
    tp.max_iters = 30; // 30 is usually enough for a demo
    tp.max_cg_iters = 50;
    tp.free_vertices = free_vertices;
    
    std::cout << "Interpolating " << num_frames << " frames...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    TrustRegionResult res = interpolate_geodesic_trust_region(frames, ep, tp);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Interpolation finished in " << diff.count() << " seconds. Converged: " << res.converged 
              << ", iters: " << res.outer_iterations << "\n";
              
    std::cout << "Final Energy: " << path_energy(res.frames, ep).terms.total << "\n";

    for (size_t i = 0; i < res.frames.size(); ++i) {
        std::string filename = "out_compress_" + std::to_string(i) + ".obj";
        res.frames[i].save_obj(filename);
        std::cout << "Saved " << filename << "\n";
    }
    
    return 0;
}