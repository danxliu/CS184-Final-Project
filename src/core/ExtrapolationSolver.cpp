#include "ExtrapolationSolver.h"
#include "FaceGeom.h"
#include "BVH.h"
#include "BCT.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
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
    
    const PathEnergyFrameCache cache_km1 = build_path_energy_frame_cache({x_km1}, params)[0];
    const PathEnergyFrameCache cache_k = build_path_energy_frame_cache({x_k}, params)[0];
    const PathEnergyFrameCache cache_kp1 = build_path_energy_frame_cache({x_kp1}, params)[0];
    
    auto get_phi_grad = [&](const MeshData &m, const PathEnergyFrameCache &c) {
        FaceGeom g = compute_face_geom(m);
        BVH bvh = c.bvh;
        update_bvh_aggregates(bvh, g);
        if (params.tpe_adaptive.enabled) {
            double phi = tpe_energy_bh(m, g, bvh, c.bp, params.tpe_adaptive, params.tpe_alpha, &c.adaptive_cache);
            Eigen::MatrixXd grad = tpe_gradient_bh(m, g, bvh, c.bp, params.tpe_adaptive, params.tpe_alpha, &c.adaptive_cache);
            return std::make_pair(phi, grad);
        } else {
            double phi = tpe_energy_bh(g, bvh, c.bp, params.tpe_alpha);
            Eigen::MatrixXd grad = tpe_gradient_bh(m, g, bvh, c.bp, params.tpe_alpha);
            return std::make_pair(phi, grad);
        }
    };
    
    auto res_km1 = get_phi_grad(x_km1, cache_km1);
    auto res_k = get_phi_grad(x_k, cache_k);
    auto res_kp1 = get_phi_grad(x_kp1, cache_kp1);
    
    ShellEnergyGradientResult sw_12 = shell_energy_with_gradient(x_km1, x_k, params.shell);
    ShellEnergyGradientResult sw_23 = shell_energy_with_gradient(x_k, x_kp1, params.shell);
    
    out_energy = scale * (sw_12.energy.total + sw_23.energy.total + 
                         std::pow(res_km1.first - res_k.first, 2.0) + 
                         std::pow(res_k.first - res_kp1.first, 2.0));
    
    const int nv = x_k.n_vertices();
    Eigen::MatrixXd grad_k = sw_12.grad_def + sw_23.grad_ref;
    
    double dphi_12 = res_km1.first - res_k.first;
    double dphi_23 = res_k.first - res_kp1.first;
    
    grad_k += 2.0 * dphi_12 * (-res_k.second);
    grad_k += 2.0 * dphi_23 * (res_k.second);
    
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
        
        auto get_phi_grad = [&](const MeshData &m) {
            FaceGeom g = compute_face_geom(m);
            BVH bvh = build_bvh(m, g);
            BlockPairs bp = build_bct_self(bvh, params.tpe_theta);
            if (params.tpe_adaptive.enabled) {
                auto cache = build_tpe_adaptive_cache(m, g, bvh, bp, params.tpe_adaptive);
                double phi = tpe_energy_bh(m, g, bvh, bp, params.tpe_adaptive, params.tpe_alpha, &cache);
                Eigen::MatrixXd grad = tpe_gradient_bh(m, g, bvh, bp, params.tpe_adaptive, params.tpe_alpha, &cache);
                return std::make_pair(phi, grad);
            } else {
                double phi = tpe_energy_bh(g, bvh, bp, params.tpe_alpha);
                Eigen::MatrixXd grad = tpe_gradient_bh(m, g, bvh, bp, params.tpe_alpha);
                return std::make_pair(phi, grad);
            }
        };
        
        auto res_km1 = get_phi_grad(x_km1);
        auto res_k = get_phi_grad(x_k);
        auto res_kp1 = get_phi_grad(x_kp1);
        
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
        
        auto get_phi_grad_simple = [&](const MeshData &m) {
            FaceGeom g = compute_face_geom(m);
            BVH bvh = build_bvh(m, g);
            BlockPairs bp = build_bct_self(bvh, params.tpe_theta);
            if (params.tpe_adaptive.enabled) {
                auto cache = build_tpe_adaptive_cache(m, g, bvh, bp, params.tpe_adaptive);
                return tpe_gradient_bh(m, g, bvh, bp, params.tpe_adaptive, params.tpe_alpha, &cache);
            } else {
                return tpe_gradient_bh(m, g, bvh, bp, params.tpe_alpha);
            }
        };
        Eigen::MatrixXd gp_kp1_mat = get_phi_grad_simple(x_kp1);
        Eigen::VectorXd g_phi_kp1(v.size());
        for(int i=0; i<x_k.n_vertices(); ++i) g_phi_kp1.segment<3>(3*i) = gp_kp1_mat.row(i).transpose();

        double dot_val = g_phi_kp1.dot(v);
        Eigen::VectorXd Jv = Hxy_v - 2.0 * n_val * grad_phi_k * dot_val;
        
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
    Eigen::VectorXd f = compute_f(x_km1, x_k, out.next_frame, energy_params, cur_energy);
    
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
        
        for (int ls = 0; ls < params.armijo_max_steps; ++ls) {
            MeshData trial = out.next_frame;
            unpack_mesh(pack_mesh(out.next_frame) + alpha * dx, trial);
            
            double trial_energy = 0.0;
            Eigen::VectorXd f_trial = compute_f(x_km1, x_k, trial, energy_params, trial_energy);
            
            if (f_trial.squaredNorm() <= f_norm_sq * (1.0 - 2.0 * params.armijo_c * alpha)) {
                best_frame = trial;
                best_f = f_trial;
                break;
            }
            alpha *= 0.5;
            best_frame = trial;
            best_f = f_trial;
        }
        
        out.next_frame = best_frame;
        f = best_f;
    }
    
    return out;
}

} // namespace rsh