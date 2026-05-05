#include "ExtrapolationSolver.h"
#include "FaceGeom.h"
#include "BVH.h"
#include "BCT.h"
#include "Obstacle.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include <cmath>
#include <iostream>

namespace rsh {

// Forward declaration for traits specialization
struct JacobianOperator;

} // namespace rsh

namespace Eigen {
namespace internal {
template <>
struct traits<rsh::JacobianOperator> : public traits<Eigen::MatrixXd> {};
} // namespace internal
} // namespace Eigen

namespace rsh {

namespace {

Eigen::VectorXd pack_mesh(const MeshData &m) {
    const int nv = m.n_vertices();
    Eigen::VectorXd x(nv * 3);
    for (int i = 0; i < nv; ++i) {
        x.segment<3>(3 * i) = m.V.row(i).transpose();
    }
    return x;
}

void unpack_mesh(const Eigen::VectorXd &x, MeshData &m) {
    const int nv = m.n_vertices();
    for (int i = 0; i < nv; ++i) {
        m.V.row(i) = x.segment<3>(3 * i).transpose();
    }
}

std::pair<double, Eigen::MatrixXd> graph_phi_and_gradient(
    const MeshData &m,
    const PathEnergyParams &params) {
    const std::vector<MeshData> frames = {m};
    const std::vector<PathEnergyFrameCache> cache =
        build_path_energy_frame_cache(frames, params);
    const PathEnergyGradientResult pr =
        path_energy_with_gradient(frames, params, &cache);
    return {pr.energy.phi_per_frame[0], pr.grad_phi_per_frame[0]};
}

Eigen::VectorXd compute_f(
    const MeshData &x_km1,
    const MeshData &x_k,
    const MeshData &x_kp1,
    const PathEnergyParams &params,
    double &out_energy) {
    
    const int n_val = 2;
    const double scale = static_cast<double>(n_val);
    
    std::vector<MeshData> frames_12 = {x_km1, x_k};
    std::vector<MeshData> frames_23 = {x_k, x_kp1};
    
    // We need dE/dx_k. E = n * (W(x_km1, x_k) + W(x_k, x_kp1) + (phi_km1 - phi_k)^2 + (phi_k - phi_kp1)^2)
    // dE/dx_k = n * (dW(x_km1, x_k)/dx_k + dW(x_k, x_kp1)/dx_k + 2(phi_km1 - phi_k)(-dphi_k/dx_k) + 2(phi_k - phi_kp1)(dphi_k/dx_k))
    
    auto res_km1 = graph_phi_and_gradient(x_km1, params);
    auto res_k = graph_phi_and_gradient(x_k, params);
    auto res_kp1 = graph_phi_and_gradient(x_kp1, params);
    
    ShellEnergyGradientResult sw_12 = shell_energy_with_gradient(x_km1, x_k, params.shell);
    ShellEnergyGradientResult sw_23 = shell_energy_with_gradient(x_k, x_kp1, params.shell);
    
    out_energy = scale * (sw_12.energy.total + sw_23.energy.total +
                         params.graph_beta *
                             std::pow(res_km1.first - res_k.first, 2.0) +
                         params.graph_beta *
                             std::pow(res_k.first - res_kp1.first, 2.0));

    const int nv = x_k.n_vertices();
    Eigen::MatrixXd grad_k = sw_12.grad_def + sw_23.grad_ref;

    double dphi_12 = res_km1.first - res_k.first;
    double dphi_23 = res_k.first - res_kp1.first;

    grad_k += params.graph_beta * 2.0 * dphi_12 * (-res_k.second);
    grad_k += params.graph_beta * 2.0 * dphi_23 * (res_k.second);

    Eigen::VectorXd f(nv * 3);
    for (int i = 0; i < nv; ++i) {
        f.segment<3>(3 * i) = scale * grad_k.row(i).transpose();
    }
    return f;
}

Eigen::VectorXd compute_g_k_part(const MeshData &x_k, const MeshData &x_kp1, const PathEnergyParams &params) {
    ShellEnergyGradientResult sr_23 = shell_energy_with_gradient(x_k, x_kp1, params.shell);
    const int nv = x_k.n_vertices();
    Eigen::VectorXd g(nv * 3);
    for (int i = 0; i < nv; ++i) {
        g.segment<3>(3 * i) = sr_23.grad_ref.row(i).transpose();
    }
    return g;
}

} // namespace

struct JacobianOperator : public Eigen::EigenBase<JacobianOperator> {
    const MeshData& x_km1;
    const MeshData& x_k;
    const MeshData& x_kp1;
    const PathEnergyParams& params;
    const ExtrapolationParams& ext_params;
    Eigen::VectorXd grad_phi_k;
    double phi_km1;
    double phi_k;
    double phi_kp1;

    JacobianOperator(const MeshData& x_km1, const MeshData& x_k, const MeshData& x_kp1,
                     const PathEnergyParams& params, const ExtrapolationParams& ext_params)
        : x_km1(x_km1), x_k(x_k), x_kp1(x_kp1), params(params), ext_params(ext_params) {
        
        auto res_km1 = graph_phi_and_gradient(x_km1, params);
        auto res_k = graph_phi_and_gradient(x_k, params);
        auto res_kp1 = graph_phi_and_gradient(x_kp1, params);
        
        phi_km1 = res_km1.first;
        phi_k = res_k.first;
        phi_kp1 = res_kp1.first;
        
        const int nv = x_k.n_vertices();
        grad_phi_k.resize(nv * 3);
        for (int i = 0; i < nv; ++i) {
            grad_phi_k.segment<3>(3 * i) = res_k.second.row(i).transpose();
        }
    }

    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    typedef int Index;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return x_k.n_vertices() * 3; }
    Index cols() const { return x_k.n_vertices() * 3; }

    template<typename Rhs>
    Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& v) const {
        if (v.size() == 0) return Eigen::VectorXd();
        const double nv_norm = v.norm();
        if (!(nv_norm > 0.0)) return Eigen::VectorXd::Zero(v.size());
        
        const double h = ext_params.fd_eps / nv_norm;
        
        MeshData x_kp1_plus = x_kp1;
        MeshData x_kp1_minus = x_kp1;
        unpack_mesh(pack_mesh(x_kp1) + h * v, x_kp1_plus);
        unpack_mesh(pack_mesh(x_kp1) - h * v, x_kp1_minus);
        
        // f = n * (dW(x_km1, x_k)/dx_k + dW(x_k, x_kp1)/dx_k + repulsive_k)
        // Only dW(x_k, x_kp1)/dx_k and phi_kp1 in repulsive_k depend on x_kp1.
        // d/dx_kp1 ( dW(x_k, x_kp1)/dx_k ) is the mixed Hessian H_{xy} (where x=x_k, y=x_kp1).
        
        auto g_k_from_kp1 = [&](const MeshData &x_kp1_val) {
            ShellEnergyGradientResult sw_23 = shell_energy_with_gradient(x_k, x_kp1_val, params.shell);
            const int nv = x_k.n_vertices();
            Eigen::VectorXd g(nv * 3);
            for (int i = 0; i < nv; ++i) {
                g.segment<3>(3 * i) = sw_23.grad_ref.row(i).transpose();
            }
            return g;
        };
        
        Eigen::VectorXd g_plus = g_k_from_kp1(x_kp1_plus);
        Eigen::VectorXd g_minus = g_k_from_kp1(x_kp1_minus);
        
        const double n_val = 2.0;
        Eigen::VectorXd Hxy_v = n_val * (g_plus - g_minus) / (2.0 * h);
        
        // Repulsive part in f: 2n (phi_k - phi_kp1) grad_phi_k
        // d/dx_kp1 ( 2n (phi_k - phi_kp1) grad_phi_k ) = 2n (-grad_phi_kp1^T applied to v) grad_phi_k
        // = -2n (grad_phi_kp1 . v) grad_phi_k
        
        // For Jacobian check, we also need grad_phi_kp1.
        // I should have computed it in constructor.
        
        Eigen::MatrixXd gp_kp1_mat =
            graph_phi_and_gradient(x_kp1, params).second;
        Eigen::VectorXd g_phi_kp1(v.size());
        for(int i=0; i<x_k.n_vertices(); ++i) g_phi_kp1.segment<3>(3*i) = gp_kp1_mat.row(i).transpose();

        double dot_val = g_phi_kp1.dot(v);
        Eigen::VectorXd Jv =
            Hxy_v -
            params.graph_beta * 2.0 * n_val * grad_phi_k * dot_val;
        
        return Jv;
    }
};

ExtrapolationResult extrapolate_geodesic(
    const MeshData &x_km1,
    const MeshData &x_k,
    const PathEnergyParams &energy_params,
    const ExtrapolationParams &params) {
    
    ExtrapolationResult out;
    out.next_frame = x_k; 
    // Linear extrapolation guess: x_kp1 = x_k + (x_k - x_km1) = 2*x_k - x_km1
    out.next_frame.V = 2.0 * x_k.V - x_km1.V;
    
    double cur_energy = 0.0;
    Eigen::VectorXd f =
        compute_f(x_km1, x_k, out.next_frame, energy_params, cur_energy);
    if (!std::isfinite(cur_energy)) {
        double velocity_scale = 0.5;
        bool found_feasible_guess = false;
        for (int ls = 0; ls < params.armijo_max_steps; ++ls) {
            out.next_frame = x_k;
            out.next_frame.V =
                x_k.V + velocity_scale * (x_k.V - x_km1.V);
            f = compute_f(x_km1, x_k, out.next_frame, energy_params,
                          cur_energy);
            if (std::isfinite(cur_energy)) {
                found_feasible_guess = true;
                break;
            }
            velocity_scale *= 0.5;
        }
        if (!found_feasible_guess) {
            out.next_frame = x_k;
            f = compute_f(x_km1, x_k, out.next_frame, energy_params,
                          cur_energy);
        }
    }
    
    for (int it = 0; it < params.max_newton_iters; ++it) {
        out.newton_iters = it + 1;
        if (f.norm() < params.newton_tol) {
            out.converged = true;
            break;
        }
        
        JacobianOperator J(x_km1, x_k, out.next_frame, energy_params, params);
        Eigen::GMRES<JacobianOperator, Eigen::IdentityPreconditioner> gmres;
        gmres.compute(J);
        gmres.setTolerance(params.gmres_tol);
        gmres.setMaxIterations(params.max_gmres_iters);
        
        Eigen::VectorXd dx = gmres.solve(-f);
        if (gmres.info() != Eigen::Success) {
            std::cerr << "GMRES failed to converge in extrapolation step.\n";
        }
        
        // Armijo line search
        double alpha = 1.0;
        double f_norm_sq = f.squaredNorm();
        MeshData best_frame = out.next_frame;
        Eigen::VectorXd best_f = f;
        bool accepted_step = false;
        
        for (int ls = 0; ls < params.armijo_max_steps; ++ls) {
            MeshData trial = out.next_frame;
            unpack_mesh(pack_mesh(out.next_frame) + alpha * dx, trial);
            
            double trial_energy = 0.0;
            Eigen::VectorXd f_trial = compute_f(x_km1, x_k, trial, energy_params, trial_energy);
            
            if (std::isfinite(trial_energy) &&
                f_trial.squaredNorm() <=
                    f_norm_sq * (1.0 - 2.0 * params.armijo_c * alpha)) {
                best_frame = trial;
                best_f = f_trial;
                accepted_step = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted_step) {
            break;
        }
        
        out.next_frame = best_frame;
        f = best_f;
    }
    
    return out;
}

} // namespace rsh
