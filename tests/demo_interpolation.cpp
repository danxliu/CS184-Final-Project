#include "MeshData.h"
#include "TestMeshes.h"
#include "PathEnergy.h"
#include "TrustRegionSolver.h"

#include <iostream>
#include <string>
#include <vector>

using namespace rsh;

int main(int argc, char **argv) {
    std::cout << "=== Interpolation Demo ===\n";

    // Load two meshes or create procedural ones
    MeshData x0 = make_icosphere(1);
    x0.normalize();
    
    MeshData x_end = x0;
    x_end.V.col(0) *= 1.2;
    x_end.V.col(1) *= 0.8;
    x_end.V.rowwise() += Eigen::RowVector3d(0.5, 0.0, 0.0);

    const int num_frames = 5; // 0, 1, 2, 3, 4
    std::vector<MeshData> frames(num_frames, x0);
    frames.back() = x_end;
    
    // Piecewise constant initialization
    for (int i = 1; i < num_frames - 1; ++i) {
        if (i < num_frames / 2) {
            frames[i] = x0;
        } else {
            frames[i] = x_end;
        }
    }

    PathEnergyParams ep;
    ep.tpe_adaptive.enabled = true;
    ep.tpe_adaptive.max_depth = 2;

    TrustRegionParams tp;
    tp.max_iters = 20;

    std::cout << "Interpolating " << num_frames << " frames...\n";
    TrustRegionResult res = interpolate_geodesic_trust_region(frames, ep, tp);
    
    std::cout << "Interpolation finished. Converged: " << res.converged 
              << ", iters: " << res.outer_iterations << "\n";
              
    std::cout << "Final Energy: " << path_energy(res.frames, ep).terms.total << "\n";

    for (size_t i = 0; i < res.frames.size(); ++i) {
        std::string filename = "out_interp_" + std::to_string(i) + ".obj";
        res.frames[i].save_obj(filename);
        std::cout << "Saved " << filename << "\n";
    }

    return 0;
}
