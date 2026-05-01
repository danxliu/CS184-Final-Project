#ifndef TRUST_REGION_SOLVER_H
#define TRUST_REGION_SOLVER_H

#include "PathEnergy.h"

#include <vector>

namespace rsh {

struct TrustRegionParams {
    int max_iters = 30;
    int max_cg_iters = 50;
    double grad_tol = 1e-6;
    double cg_tol = 1e-4;
    double initial_radius = 1e-2;
    double max_radius = 1.0;
    double accept_eta = 0.1;
    double hvp_eps = 1e-5;
    std::vector<bool> free_vertices;
    bool optimize_end_frame = false;
};

struct TrustRegionResult {
    std::vector<MeshData> frames;
    std::vector<double> accepted_energy;
    int outer_iterations = 0;
    int accepted_steps = 0;
    bool converged = false;
};

// Interpolate a geodesic-like trajectory by minimizing the discrete path energy
// with fixed endpoints using a trust-region method and Steihaug CG subproblems.
TrustRegionResult interpolate_geodesic_trust_region(
    const std::vector<MeshData> &initial_frames,
    const PathEnergyParams &energy_params = PathEnergyParams(),
    const TrustRegionParams &tr_params = TrustRegionParams());

} // namespace rsh

#endif
