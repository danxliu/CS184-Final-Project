#include "MeshData.h"
#include "TestMeshes.h"
#include "PathEnergy.h"
#include "ExtrapolationSolver.h"

#include <iostream>
#include <string>
#include <vector>

using namespace rsh;

int main(int argc, char **argv) {
    std::cout << "=== Extrapolation Demo ===\n";

    MeshData x0 = make_icosphere(1);
    x0.normalize();
    
    MeshData x1 = x0;
    // Initial velocity
    x1.V.rowwise() += Eigen::RowVector3d(0.1, 0.0, 0.0);
    // Twist a bit
    for (int i = 0; i < x1.n_vertices(); ++i) {
        double y = x1.V(i, 1);
        double z = x1.V(i, 2);
        double theta = 0.1 * x1.V(i, 0);
        x1.V(i, 1) = y * std::cos(theta) - z * std::sin(theta);
        x1.V(i, 2) = y * std::sin(theta) + z * std::cos(theta);
    }

    PathEnergyParams ep;
    ep.tpe_adaptive.enabled = true;
    ep.tpe_adaptive.max_depth = 2;

    ExtrapolationParams ext_p;

    const int num_frames = 5;
    std::vector<MeshData> frames;
    frames.push_back(x0);
    frames.push_back(x1);

    x0.save_obj("out_extrap_0.obj");
    x1.save_obj("out_extrap_1.obj");

    std::cout << "Extrapolating...\n";
    for (int i = 2; i < num_frames; ++i) {
        std::cout << "  Frame " << i << "...\n";
        ExtrapolationResult res = extrapolate_geodesic(frames[i - 2], frames[i - 1], ep, ext_p);
        frames.push_back(res.next_frame);
        
        std::string filename = "out_extrap_" + std::to_string(i) + ".obj";
        res.next_frame.save_obj(filename);
        std::cout << "    Saved " << filename << " (iters: " << res.newton_iters << ")\n";
    }

    return 0;
}
