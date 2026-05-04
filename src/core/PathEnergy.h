#ifndef PATH_ENERGY_H
#define PATH_ENERGY_H

#include "MeshData.h"
#include "ShellEnergy.h"
#include "TPE.h"

#include <Eigen/Dense>
#include <vector>

namespace rsh {

class Obstacle;

struct PathEnergyParams {
    ShellEnergyParams shell;
    double tpe_alpha = 6.0;
    double tpe_theta = 0.5;
    TpeAdaptiveParams tpe_adaptive;
    // Optional analytic obstacle. When non-null, its vertex barrier is added
    // to the graph-manifold potential:
    //   Phi_total = TPE + obstacle_weight * Phi_obstacle.
    // It is not an ordinary additive force term in the path energy.
    const Obstacle *obstacle = nullptr;
    double obstacle_weight = 1.0;
};

struct PathEnergyTermBreakdown {
    double total = 0.0;
    double shell_sum = 0.0;
    double repulsive_sum = 0.0;
    double obstacle_sum = 0.0;
};

struct PathEnergyResult {
    PathEnergyTermBreakdown terms;
    // Per-frame total graph potential Phi_total, not raw TPE.
    std::vector<double> phi_per_frame;
};

struct PathEnergyGradientResult {
    PathEnergyResult energy;
    std::vector<Eigen::MatrixXd> grad_frames;  // dE / d x_k
    // dPhi_total / d x_k.
    std::vector<Eigen::MatrixXd> grad_phi_per_frame;
};

struct PathEnergyFrameCache {
    BVH bvh;
    BlockPairs bp;
    TpeAdaptiveCache adaptive_cache;
    bool has_adaptive = false;
};

// Build frozen per-frame hierarchy/partition caches. These can be reused while
// evaluating energy/gradient on nearby geometry to avoid admissibility flips.
std::vector<PathEnergyFrameCache> build_path_energy_frame_cache(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams());

// Discrete path energy (RS Eq. 6 + Eq. 17 approximation):
// E_hat(x0..xn) = n * sum_k [ W_c(x_{k-1}, x_k) + (Phi(x_{k-1}) - Phi(x_k))^2 ].
PathEnergyResult path_energy(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams(),
    const std::vector<PathEnergyFrameCache> *frame_cache = nullptr);

// Energy + per-frame gradients.
PathEnergyGradientResult path_energy_with_gradient(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams(),
    const std::vector<PathEnergyFrameCache> *frame_cache = nullptr);

} // namespace rsh

#endif
