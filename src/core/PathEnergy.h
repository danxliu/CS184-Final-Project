#ifndef PATH_ENERGY_H
#define PATH_ENERGY_H

#include "MeshData.h"
#include "ShellEnergy.h"
#include "SurfaceBarrier.h"
#include "TPE.h"

#include <Eigen/Dense>
#include <vector>

namespace rsh {

class Obstacle;

struct PathEnergyParams {
    ShellEnergyParams shell;
    double tpe_alpha = 6.0;
    double tpe_theta = 0.5;
    // Global graph-metric strength. The official Repulsive Shells demo uses
    // separate graph coordinates for self TPE and obstacle TPE, then weights
    // their squared path differences independently.
    double graph_beta = 1.0;
    // Repulsor's "innerWeight": scales raw TPE values before differencing.
    double tpe_inner_weight = 1.0;
    // Weight of the self-TPE graph coordinate.
    double self_tpe_weight = 1.0;
    TpeAdaptiveParams tpe_adaptive;
    // Optional fixed obstacle surface. This is a separate graph coordinate,
    // not added into the self-TPE coordinate before squaring.
    const MeshData *tpe_barrier_mesh = nullptr;
    double tpe_barrier_weight = 1.0;
    // Optional analytic obstacle guard. This also lives as its own graph
    // coordinate when enabled; the paper demo leaves it disabled.
    const Obstacle *obstacle = nullptr;
    double obstacle_weight = 1.0;
    // RS Section 7.4.3 rigid-motion terms. The shell metric factors out rigid
    // motions, so translated endpoints need these terms to avoid zero-cost
    // teleporting paths.
    double rigid_translation_weight = 0.0;
    double rigid_rotation_weight = 0.0;
};

struct PathEnergyTermBreakdown {
    double total = 0.0;
    double shell_sum = 0.0;
    double repulsive_sum = 0.0;
    double obstacle_sum = 0.0;
    double rigid_sum = 0.0;
};

struct PathEnergyResult {
    PathEnergyTermBreakdown terms;
    // Compatibility aggregate for older scalar-Phi callers. Coordinates are
    // sqrt-weighted so a single active coordinate preserves the same weighted
    // squared difference as the vector graph energy. New code should prefer
    // the coordinate-specific arrays below.
    std::vector<double> phi_per_frame;
    std::vector<double> self_phi_per_frame;
    std::vector<double> barrier_phi_per_frame;
    std::vector<double> obstacle_phi_per_frame;
};

struct PathEnergyGradientResult {
    PathEnergyResult energy;
    std::vector<Eigen::MatrixXd> grad_frames;  // dE / d x_k
    // Compatibility aggregate gradient matching phi_per_frame.
    std::vector<Eigen::MatrixXd> grad_phi_per_frame;
    std::vector<Eigen::MatrixXd> grad_self_phi_per_frame;
    std::vector<Eigen::MatrixXd> grad_barrier_phi_per_frame;
    std::vector<Eigen::MatrixXd> grad_obstacle_phi_per_frame;
};

struct PathEnergyFrameCache {
    BVH bvh;
    BlockPairs bp;
    TpeAdaptiveCache adaptive_cache;
    bool has_adaptive = false;
    SurfaceBarrierCache barrier_cache;
    bool has_barrier_cache = false;
};

// Build frozen per-frame hierarchy/partition caches. These can be reused while
// evaluating energy/gradient on nearby geometry to avoid admissibility flips.
std::vector<PathEnergyFrameCache> build_path_energy_frame_cache(
    const std::vector<MeshData> &frames,
    const PathEnergyParams &params = PathEnergyParams());

// Discrete path energy (RS Eq. 6 + Eq. 17 approximation). Self TPE,
// fixed-surface TPE, and optional SDF guards are separate graph coordinates;
// they are not summed into one Phi before squaring.
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
