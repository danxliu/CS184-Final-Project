#include "TrustRegionSolver.h"

#include "PathEnergy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace rsh {

namespace {

int interior_dim(const std::vector<MeshData> &frames) {
    const int n = static_cast<int>(frames.size()) - 1;
    if (n < 1) return 0;
    return std::max(0, (n - 1) * frames[0].n_vertices() * 3);
}

Eigen::VectorXd pack_interior_frames(const std::vector<MeshData> &frames) {
    const int n = static_cast<int>(frames.size()) - 1;
    const int nv = frames[0].n_vertices();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(std::max(0, (n - 1) * nv * 3));
    int off = 0;
    for (int k = 1; k <= n - 1; ++k) {
        for (int i = 0; i < nv; ++i) {
            z.segment<3>(off) = frames[k].V.row(i).transpose();
            off += 3;
        }
    }
    return z;
}

void unpack_interior_frames(const Eigen::VectorXd &z, std::vector<MeshData> &frames) {
    const int n = static_cast<int>(frames.size()) - 1;
    const int nv = frames[0].n_vertices();
    int off = 0;
    for (int k = 1; k <= n - 1; ++k) {
        for (int i = 0; i < nv; ++i) {
            frames[k].V.row(i) = z.segment<3>(off).transpose();
            off += 3;
        }
    }
}

Eigen::VectorXd pack_interior_gradient(const PathEnergyGradientResult &g) {
    const int n = static_cast<int>(g.grad_frames.size()) - 1;
    const int nv = static_cast<int>(g.grad_frames[0].rows());
    Eigen::VectorXd out = Eigen::VectorXd::Zero(std::max(0, (n - 1) * nv * 3));
    int off = 0;
    for (int k = 1; k <= n - 1; ++k) {
        for (int i = 0; i < nv; ++i) {
            out.segment<3>(off) = g.grad_frames[k].row(i).transpose();
            off += 3;
        }
    }
    return out;
}

double norm2(const Eigen::VectorXd &x) { return x.squaredNorm(); }

struct ModelEval {
    double energy = 0.0;
    Eigen::VectorXd grad;
    std::vector<double> phi_per_frame;
    std::vector<Eigen::MatrixXd> grad_phi_per_frame;
};

ModelEval eval_model(const std::vector<MeshData> &frames,
                     const PathEnergyParams &energy_params,
                     const std::vector<PathEnergyFrameCache> &cache) {
    ModelEval out;
    const PathEnergyGradientResult pr =
        path_energy_with_gradient(frames, energy_params, &cache);
    out.energy = pr.energy.terms.total;
    out.grad = pack_interior_gradient(pr);
    out.phi_per_frame = pr.energy.phi_per_frame;
    out.grad_phi_per_frame = pr.grad_phi_per_frame;
    return out;
}

Eigen::VectorXd shell_grad_only(const std::vector<MeshData> &frames,
                                const PathEnergyParams &energy_params,
                                const std::vector<PathEnergyFrameCache> &cache) {
    const PathEnergyGradientResult pr = path_energy_with_gradient(frames, energy_params, &cache);
    const int n = static_cast<int>(frames.size()) - 1;
    const double scale = static_cast<double>(n);
    
    std::vector<Eigen::MatrixXd> grad_frames = pr.grad_frames;
    for (int k = 1; k <= n; ++k) {
        const int km1 = k - 1;
        const double dphi = pr.energy.phi_per_frame[km1] - pr.energy.phi_per_frame[k];
        grad_frames[static_cast<size_t>(km1)] -= scale * (2.0 * dphi) * pr.grad_phi_per_frame[static_cast<size_t>(km1)];
        grad_frames[static_cast<size_t>(k)] -= scale * (-2.0 * dphi) * pr.grad_phi_per_frame[static_cast<size_t>(k)];
    }
    
    const int nv = static_cast<int>(grad_frames[0].rows());
    Eigen::VectorXd out = Eigen::VectorXd::Zero(std::max(0, (n - 1) * nv * 3));
    int off = 0;
    for (int k = 1; k <= n - 1; ++k) {
        for (int i = 0; i < nv; ++i) {
            out.segment<3>(off) = grad_frames[static_cast<size_t>(k)].row(i).transpose();
            off += 3;
        }
    }
    return out;
}

Eigen::VectorXd hvp(const std::vector<MeshData> &frames,
                    const PathEnergyParams &energy_params,
                    const std::vector<PathEnergyFrameCache> &cache,
                    const ModelEval &cur,
                    const Eigen::VectorXd &x,
                    const Eigen::VectorXd &v,
                    double eps) {
    if (v.size() == 0) return Eigen::VectorXd();
    const double nv_norm = v.norm();
    if (!(nv_norm > 0.0)) return Eigen::VectorXd::Zero(v.size());
    const double h = eps / nv_norm;

    std::vector<MeshData> fp = frames;
    std::vector<MeshData> fm = frames;
    unpack_interior_frames(x + h * v, fp);
    unpack_interior_frames(x - h * v, fm);

    const Eigen::VectorXd gp_shell = shell_grad_only(fp, energy_params, cache);
    const Eigen::VectorXd gm_shell = shell_grad_only(fm, energy_params, cache);
    Eigen::VectorXd Hs = (gp_shell - gm_shell) / (2.0 * h);

    const int n = static_cast<int>(frames.size()) - 1;
    const double scale = static_cast<double>(n);
    const int nv = frames[0].n_vertices();
    
    std::vector<double> w(static_cast<size_t>(n + 1), 0.0);
    std::vector<Eigen::MatrixXd> v_frames(static_cast<size_t>(n + 1), Eigen::MatrixXd::Zero(nv, 3));
    int off = 0;
    for (int k = 1; k <= n - 1; ++k) {
        for (int i = 0; i < nv; ++i) {
            v_frames[static_cast<size_t>(k)].row(i) = v.segment<3>(off).transpose();
            off += 3;
        }
    }
    
    for (int k = 1; k <= n; ++k) {
        const int km1 = k - 1;
        double dot_km1 = 0.0;
        double dot_k = 0.0;
        if (km1 > 0 && km1 < n) {
            dot_km1 = (cur.grad_phi_per_frame[static_cast<size_t>(km1)].array() * v_frames[static_cast<size_t>(km1)].array()).sum();
        }
        if (k > 0 && k < n) {
            dot_k = (cur.grad_phi_per_frame[static_cast<size_t>(k)].array() * v_frames[static_cast<size_t>(k)].array()).sum();
        }
        w[static_cast<size_t>(k)] = dot_km1 - dot_k;
    }
    
    std::vector<Eigen::MatrixXd> h_gn_frames(static_cast<size_t>(n + 1), Eigen::MatrixXd::Zero(nv, 3));
    for (int i = 1; i <= n - 1; ++i) {
        h_gn_frames[static_cast<size_t>(i)] = cur.grad_phi_per_frame[static_cast<size_t>(i)] * (w[static_cast<size_t>(i + 1)] - w[static_cast<size_t>(i)]);
    }
    
    Eigen::VectorXd h_gn_vec = Eigen::VectorXd::Zero(v.size());
    off = 0;
    for (int i = 1; i <= n - 1; ++i) {
        for (int j = 0; j < nv; ++j) {
            h_gn_vec.segment<3>(off) = h_gn_frames[static_cast<size_t>(i)].row(j).transpose();
            off += 3;
        }
    }
    
    Hs += scale * 2.0 * h_gn_vec;
    return Hs;
}

Eigen::VectorXd steihaug_cg(const std::vector<MeshData> &frames,
                            const PathEnergyParams &energy_params,
                            const std::vector<PathEnergyFrameCache> &cache,
                            const ModelEval &cur,
                            const Eigen::VectorXd &x,
                            const Eigen::VectorXd &g,
                            const TrustRegionParams &p,
                            double radius) {
    const int dim = g.size();
    Eigen::VectorXd s = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd r = g;
    Eigen::VectorXd d = -r;
    if (dim == 0) return s;
    const double r0 = r.norm();
    if (r0 < p.grad_tol) return s;

    auto tau_to_boundary = [&](const Eigen::VectorXd &s0, const Eigen::VectorXd &d0) {
        const double a = d0.squaredNorm();
        const double b = 2.0 * s0.dot(d0);
        const double c = s0.squaredNorm() - radius * radius;
        const double disc = std::max(0.0, b * b - 4.0 * a * c);
        return (-b + std::sqrt(disc)) / (2.0 * a);
    };

    for (int it = 0; it < p.max_cg_iters; ++it) {
        const Eigen::VectorXd Hd =
            hvp(frames, energy_params, cache, cur, x, d, p.hvp_eps);
        const double dHd = d.dot(Hd);
        if (dHd <= 0.0) {
            const double tau = tau_to_boundary(s, d);
            return s + tau * d;
        }
        const double alpha = r.dot(r) / dHd;
        const Eigen::VectorXd s_next = s + alpha * d;
        if (s_next.norm() >= radius) {
            const double tau = tau_to_boundary(s, d);
            return s + tau * d;
        }
        s = s_next;
        const Eigen::VectorXd r_next = r + alpha * Hd;
        if (r_next.norm() <= p.cg_tol * r0) {
            return s;
        }
        const double beta = r_next.dot(r_next) / r.dot(r);
        d = -r_next + beta * d;
        r = r_next;
    }
    return s;
}

} // namespace

TrustRegionResult interpolate_geodesic_trust_region(
    const std::vector<MeshData> &initial_frames,
    const PathEnergyParams &energy_params,
    const TrustRegionParams &tr_params) {
    if (initial_frames.size() < 2) {
        throw std::runtime_error(
            "interpolate_geodesic_trust_region: need at least 2 frames");
    }
    TrustRegionResult out;
    out.frames = initial_frames;
    const int dim = interior_dim(out.frames);
    if (dim == 0) {
        out.converged = true;
        out.accepted_energy.push_back(path_energy(out.frames, energy_params).terms.total);
        return out;
    }

    Eigen::VectorXd x = pack_interior_frames(out.frames);
    const std::vector<PathEnergyFrameCache> cache =
        build_path_energy_frame_cache(out.frames, energy_params);

    double radius = std::max(1e-10, tr_params.initial_radius);
    ModelEval cur = eval_model(out.frames, energy_params, cache);
    out.accepted_energy.push_back(cur.energy);

    for (int it = 0; it < tr_params.max_iters; ++it) {
        out.outer_iterations = it + 1;
        const double gnorm = cur.grad.norm();
        if (gnorm < tr_params.grad_tol) {
            out.converged = true;
            break;
        }

        const Eigen::VectorXd s =
            steihaug_cg(out.frames, energy_params, cache, cur, x, cur.grad, tr_params, radius);
        if (s.norm() == 0.0) {
            radius *= 0.5;
            if (radius < 1e-12) break;
            continue;
        }

        std::vector<MeshData> trial = out.frames;
        const Eigen::VectorXd x_trial = x + s;
        unpack_interior_frames(x_trial, trial);
        const double e_trial = path_energy(trial, energy_params, &cache).terms.total;

        const Eigen::VectorXd Hs =
            hvp(out.frames, energy_params, cache, cur, x, s, tr_params.hvp_eps);
        const double pred = -(cur.grad.dot(s) + 0.5 * s.dot(Hs));
        const double ared = cur.energy - e_trial;
        const double rho = (pred > 0.0) ? (ared / pred) : -1.0;

        if (rho > tr_params.accept_eta && std::isfinite(e_trial)) {
            out.frames = std::move(trial);
            x = x_trial;
            cur = eval_model(out.frames, energy_params, cache);
            out.accepted_steps += 1;
            out.accepted_energy.push_back(cur.energy);
            if (rho > 0.75 && std::abs(s.norm() - radius) < 1e-8) {
                radius = std::min(tr_params.max_radius, 2.0 * radius);
            } else if (rho < 0.25) {
                radius *= 0.5;
            }
        } else {
            radius *= 0.5;
        }
        if (radius < 1e-12) break;
    }
    return out;
}

} // namespace rsh
