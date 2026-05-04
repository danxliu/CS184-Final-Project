#include "OptimizeTPE.h"

#include "BCT.h"
#include "BVH.h"
#include "Constraints.h"
#include "FaceGeom.h"
#include "Remesh.h"
#include "TPE.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rsh {

namespace {

std::string frame_path(const std::string &dir, int idx) {
    std::ostringstream oss;
    oss << dir << "/frame_" << std::setfill('0') << std::setw(4) << idx
        << ".obj";
    return oss.str();
}

void remove_stale_outputs(const std::string &dir) {
    if (!std::filesystem::exists(dir)) return;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if ((name.rfind("frame_", 0) == 0 &&
             entry.path().extension() == ".obj") ||
            name == "energy.csv") {
            std::filesystem::remove(entry.path());
        }
    }
}

void remove_scale_mode(const Eigen::MatrixXd &V, Eigen::MatrixXd &field) {
    const Eigen::RowVector3d c = V.colwise().mean();
    const Eigen::MatrixXd R = V.rowwise() - c;
    const double den = R.squaredNorm();
    if (den > 0.0) {
        const double num = R.cwiseProduct(field).sum();
        field -= (num / den) * R;
    }
}

void normalize_to_centroid(MeshData &mesh, const Eigen::RowVector3d &centroid) {
    mesh.normalize();
    mesh.V.rowwise() += centroid;
}

double tpe_energy_current(const MeshData &mesh,
                          const OptimizeTPEParams &params) {
    const FaceGeom g = compute_face_geom(mesh);
    const BVH bvh = build_bvh(mesh, g);
    const BlockPairs bp = build_bct_self(bvh, params.bvh_theta);
    return tpe_energy_bh(g, bvh, bp, params.tpe_alpha);
}

bool valid_params(const OptimizeTPEParams &p) {
    return p.max_iters >= 0 &&
           p.grad_tol >= 0.0 &&
           p.remesh_every >= 0 &&
           std::isfinite(p.armijo_c1) &&
           p.armijo_c1 >= 0.0 &&
           p.armijo_c1 < 1.0 &&
           std::isfinite(p.armijo_shrink) &&
           p.armijo_shrink > 0.0 &&
           p.armijo_shrink < 1.0 &&
           p.armijo_max_backtracks >= 0 &&
           std::isfinite(p.initial_tau) &&
           p.initial_tau > 0.0 &&
           std::isfinite(p.tpe_alpha) &&
           p.tpe_alpha > 0.0 &&
           std::isfinite(p.bvh_theta) &&
           p.bvh_theta >= 0.0;
}

} // namespace

OptimizeTPEParams::OptimizeTPEParams() {
    hs_params.s = 5.0 / 3.0;
    hs_params.sigma = 1.0;
    hs_params.mass_weight = 0.0;
    hs_params.theta = 0.25;
    constraints.pin_barycenter = true;
}

OptimizeTPEResult optimize_tpe(const MeshData &initial,
                               const OptimizeTPEParams &params) {
    if (!valid_params(params)) {
        throw std::runtime_error("optimize_tpe: invalid optimization parameter");
    }
    if (initial.n_vertices() <= 0 || initial.n_faces() <= 0) {
        throw std::runtime_error("optimize_tpe: initial mesh is empty");
    }
    if (params.constraints.pin_mask != nullptr &&
        static_cast<int>(params.constraints.pin_mask->size()) !=
            initial.n_vertices()) {
        throw std::runtime_error("optimize_tpe: pin mask size mismatch");
    }

    OptimizeTPEResult out;
    MeshData mesh = initial;
    out.final_mesh = mesh;
    const Eigen::RowVector3d target_centroid = mesh.V.colwise().mean();

    std::ofstream csv;
    if (!params.out_dir.empty()) {
        std::filesystem::create_directories(params.out_dir);
        remove_stale_outputs(params.out_dir);
        mesh.save_obj(frame_path(params.out_dir, 0));
        csv.open(params.out_dir + "/energy.csv");
        csv << "iter,energy,grad_norm,step_size,n_backtracks,did_remesh\n";
    }

    double current_energy = tpe_energy_current(mesh, params);
    double current_grad_norm = 0.0;
    if (csv.is_open()) {
        csv << "0," << current_energy << ",0,0,0,0\n";
        csv.flush();
    }

    double tau = params.initial_tau;
    int consecutive_armijo_failures = 0;

    for (int iter = 0; iter < params.max_iters; ++iter) {
        const FaceGeom g = compute_face_geom(mesh);
        const BVH bvh = build_bvh(mesh, g);
        const BlockPairs bp = build_bct_self(bvh, params.bvh_theta);

        current_energy = tpe_energy_bh(g, bvh, bp, params.tpe_alpha);
        Eigen::MatrixXd gradient =
            tpe_gradient_bh(mesh, g, bvh, bp, params.tpe_alpha);
        remove_scale_mode(mesh.V, gradient);

        HsDirectionResult hs =
            hs_preconditioned_direction(
                mesh, gradient, params.hs_params, params.constraints);
        Eigen::MatrixXd direction = hs.direction;
        remove_scale_mode(mesh.V, direction);
        if (params.constraints.pin_barycenter) {
            project_barycenter(direction);
        }
        if (params.constraints.pin_mask != nullptr) {
            apply_pin_mask(direction, *params.constraints.pin_mask);
        }

        const double g_dot_dir = (gradient.array() * direction.array()).sum();
        current_grad_norm = (std::isfinite(g_dot_dir) && g_dot_dir > 0.0)
                                ? std::sqrt(g_dot_dir)
                                : gradient.norm();
        if (current_grad_norm < params.grad_tol) {
            out.stop_reason = "grad_tol";
            break;
        }
        if (!std::isfinite(g_dot_dir) || g_dot_dir <= 0.0 ||
            !direction.allFinite()) {
            out.stop_reason = "armijo_failed";
            break;
        }

        tau = std::min(tau * 2.0, params.initial_tau);
        int n_backtracks = 0;
        MeshData trial = mesh;
        double trial_energy = current_energy;
        for (; n_backtracks < params.armijo_max_backtracks; ++n_backtracks) {
            trial = mesh;
            trial.V = mesh.V - tau * direction;
            normalize_to_centroid(trial, target_centroid);
            trial_energy = tpe_energy_current(trial, params);
            if (trial_energy <=
                current_energy - params.armijo_c1 * tau * g_dot_dir) {
                break;
            }
            tau *= params.armijo_shrink;
        }

        if (n_backtracks == params.armijo_max_backtracks) {
            ++consecutive_armijo_failures;
            if (consecutive_armijo_failures > 5) {
                out.stop_reason = "armijo_failed";
                break;
            }
            continue;
        }
        consecutive_armijo_failures = 0;

        mesh = trial;
        ++out.iterations_completed;

        bool did_remesh = false;
        if (params.remesh_every > 0 &&
            out.iterations_completed % params.remesh_every == 0) {
            MeshData remeshed = remesh_full(mesh);
            normalize_to_centroid(remeshed, target_centroid);
            const double remeshed_energy = tpe_energy_current(remeshed, params);
            mesh = remeshed;
            trial_energy = remeshed_energy;
            did_remesh = true;
            ++out.remeshes_completed;
        }

        current_energy = trial_energy;
        if (csv.is_open()) {
            csv << out.iterations_completed << "," << current_energy << ","
                << current_grad_norm << "," << tau << "," << n_backtracks
                << "," << (did_remesh ? 1 : 0) << "\n";
            csv.flush();
        }
        if (!params.out_dir.empty()) {
            const bool dump_frame =
                params.dump_every_iter || did_remesh ||
                params.max_iters <= 1 ||
                out.iterations_completed == params.max_iters;
            if (dump_frame) {
                mesh.save_obj(frame_path(params.out_dir,
                                         out.iterations_completed));
            }
        }
    }

    out.final_mesh = mesh;
    out.final_energy = tpe_energy_current(mesh, params);
    {
        const FaceGeom g = compute_face_geom(mesh);
        const BVH bvh = build_bvh(mesh, g);
        const BlockPairs bp = build_bct_self(bvh, params.bvh_theta);
        Eigen::MatrixXd gradient =
            tpe_gradient_bh(mesh, g, bvh, bp, params.tpe_alpha);
        remove_scale_mode(mesh.V, gradient);
        HsDirectionResult hs =
            hs_preconditioned_direction(
                mesh, gradient, params.hs_params, params.constraints);
        Eigen::MatrixXd direction = hs.direction;
        remove_scale_mode(mesh.V, direction);
        const double g_dot_dir = (gradient.array() * direction.array()).sum();
        out.final_grad_norm =
            (std::isfinite(g_dot_dir) && g_dot_dir > 0.0)
                ? std::sqrt(g_dot_dir)
                : gradient.norm();
    }
    if (out.iterations_completed >= params.max_iters &&
        out.stop_reason != "grad_tol" &&
        out.stop_reason != "armijo_failed") {
        out.stop_reason = "max_iters";
    }
    return out;
}

} // namespace rsh
