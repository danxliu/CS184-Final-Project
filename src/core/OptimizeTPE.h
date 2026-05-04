#ifndef OPTIMIZETPE_H
#define OPTIMIZETPE_H

#include "HsPreconditioner.h"
#include "MeshData.h"

#include <string>

namespace rsh {

struct OptimizeTPEParams {
    OptimizeTPEParams();

    int max_iters = 200;
    double grad_tol = 1e-4;
    int remesh_every = 10;
    double armijo_c1 = 1e-4;
    double armijo_shrink = 0.5;
    int armijo_max_backtracks = 60;
    double initial_tau = 1.0;
    double tpe_alpha = 6.0;
    double bvh_theta = 0.5;
    double remesh_energy_max_factor = 1.0e300;
    double remesh_max_faces_factor = 1.15;
    double remesh_max_step_faces_factor = 1.05;
    HsPreconditionerParams hs_params;
    HsConstraints constraints;
    std::string out_dir = "";
    bool dump_every_iter = true;
};

struct OptimizeTPEResult {
    MeshData final_mesh;
    int iterations_completed = 0;
    int remeshes_completed = 0;
    int remeshes_rejected = 0;
    int remeshes_rejected_energy = 0;
    int remeshes_rejected_face_budget = 0;
    double final_energy = 0.0;
    double final_grad_norm = 0.0;
    std::string stop_reason = "max_iters";
};

OptimizeTPEResult optimize_tpe(const MeshData &initial,
                               const OptimizeTPEParams &params);

} // namespace rsh

#endif
