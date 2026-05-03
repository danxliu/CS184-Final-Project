// Diagnostic validation for the matrix-free HsOperator in
// src/core/HsPreconditioner.cpp. This test intentionally includes the
// implementation file so it can call the internal HsOperator without promoting
// that type to public API.
//
// RSu Sec. 3.2.1 Eq. 12 defines the high-order term B using differences of
// per-face hat-function gradients:
//
//   sum_{S != T} <Df u(S) - Df u(T), Df v(S) - Df v(T)>
//                * area(S) area(T) / |X(S)-X(T)|^(2 sigma + 2).
//
// The paper then says sigma = s - 1 for this high-order term, so the
// distance exponent is 2(s - 1) + 2 = 2s. The current HsOperator does not
// compute Df u at all; it averages vertex values to faces and applies
// pow(r2, s + 1), i.e. a distance falloff |X(S)-X(T)|^(-2(s+1)).
//
// The checks below separate these two facts:
//   1. A field with equal face averages but different face gradients gives
//      zero under the current operator and nonzero under RSu Eq. 12.
//   2. A face-constant field measures the current operator's distance
//      exponent as 2(s+1), not the coordination note's proposed s+1.

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../src/core/HsPreconditioner.cpp"

namespace {

int failures = 0;

void check(bool cond, const std::string &msg) {
    if (cond) {
        std::cout << "  [ok] " << msg << "\n";
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failures;
    }
}

rsh::MeshData make_parallel_triangles(double d) {
    rsh::MeshData mesh;
    mesh.V.resize(6, 3);
    mesh.V << 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, d,
              1.0, 0.0, d,
              0.0, 1.0, d;
    mesh.F.resize(2, 3);
    mesh.F << 0, 1, 2,
              3, 4, 5;
    // Keep the operator's regularization eps = 1e-2 * L0^2 out of this
    // exponent measurement. The faces are well-separated, so no singularity
    // regularization is needed.
    mesh.L0 = 0.0;
    return mesh;
}

Eigen::VectorXd apply_current_hs_operator(const rsh::MeshData &mesh,
                                          const Eigen::VectorXd &u,
                                          double s) {
    rsh::HsPreconditionerParams params;
    params.s = s;
    params.sigma = 1.0;
    params.mass_weight = 0.0;

    const rsh::HsOperators hs = rsh::build_hs_operators(mesh);
    const rsh::FaceGeom geom = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, geom);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, 0.0);

    const rsh::HsOperator op(mesh, params, hs, geom, bvh, bp);
    return op * u;
}

double slope_log_log(const std::vector<double> &x,
                     const std::vector<double> &y) {
    const int n = static_cast<int>(x.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        const double lx = std::log(x[i]);
        const double ly = std::log(y[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "=== HsOperator kernel exponent validation ===\n";

    const double s = 5.0 / 3.0;
    const double reviewer_target_exponent = s + 1.0;
    const double current_distance_exponent = 2.0 * (s + 1.0);
    const double rsu_eq12_exponent = 2.0 * s;

    std::cout << "s = " << s << "\n";
    std::cout << "coordination target exponent over distance = "
              << reviewer_target_exponent << "\n";
    std::cout << "RSu Eq. 12 / Sec. 5.1.2 high-order B exponent = "
              << rsu_eq12_exponent << "\n";
    std::cout << "current code exponent over distance = "
              << current_distance_exponent << "\n";

    {
        std::cout << "-- Eq. 12 high-order structure probe --\n";
        const double d = 4.0;
        const rsh::MeshData mesh = make_parallel_triangles(d);

        // Face 0 has u = x, hence Df u = (1,0,0). Face 1 is constant 1/3,
        // hence Df u = 0. Both face averages are 1/3. Therefore the current
        // face-average operator gives zero, while the RSu high-order B
        // quadratic form is:
        //
        //   2 ordered pairs * area0*area1 * |(1,0,0)-0|^2 / d^(2s)
        // = 0.5 / d^(2s).
        Eigen::VectorXd u(6);
        u << 0.0, 1.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;

        const Eigen::VectorXd y = apply_current_hs_operator(mesh, u, s);
        const double actual = u.dot(y);
        const double paper_eq12 = 0.5 / std::pow(d, rsu_eq12_exponent);

        std::cout << "    d = " << d << "\n";
        std::cout << "    current u^T HsOperator u = " << actual << "\n";
        std::cout << "    RSu Eq.12 high-order B reference = "
                  << paper_eq12 << "\n";
        check(std::abs(actual) < 1e-14,
              "current operator ignores the nonzero face-gradient difference");
        check(paper_eq12 > 1e-4,
              "hand reference for RSu Eq. 12 is nonzero on this field");
    }

    {
        std::cout << "-- face-average distance sweep --\n";
        std::cout << "    d, actual_xTAx, ratio_to_current_formula, "
                     "ratio_to_coordination_target\n";

        std::vector<double> distances = {1.0, 2.0, 4.0, 8.0, 16.0};
        std::vector<double> actual_values;
        std::vector<double> ratios_to_target;
        actual_values.reserve(distances.size());
        ratios_to_target.reserve(distances.size());

        for (double d : distances) {
            const rsh::MeshData mesh = make_parallel_triangles(d);

            // Face-constant data: face averages are 1 and 0, but Df u is zero
            // on both faces. For the current exact near-field loop:
            //
            //   z0 = area0*area1 / d^(2(s+1)); z1 = -z0;
            //   y_i += z_face/3; u^T y = z0 = 0.25 / d^(2(s+1)).
            Eigen::VectorXd u(6);
            u << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

            const Eigen::VectorXd y = apply_current_hs_operator(mesh, u, s);
            const double actual = u.dot(y);
            const double current_formula =
                0.25 / std::pow(d, current_distance_exponent);
            const double coordination_target =
                0.25 / std::pow(d, reviewer_target_exponent);
            const double ratio_current = actual / current_formula;
            const double ratio_target = actual / coordination_target;

            actual_values.push_back(actual);
            ratios_to_target.push_back(ratio_target);

            std::cout << "    " << d << ", " << actual << ", "
                      << ratio_current << ", " << ratio_target << "\n";
            check(std::abs(ratio_current - 1.0) < 1e-12,
                  "actual matvec matches current pow(r2, s+1) formula");
        }

        const double actual_slope = slope_log_log(distances, actual_values);
        const double target_ratio_slope =
            slope_log_log(distances, ratios_to_target);

        std::cout << "    slope log(actual) vs log(d) = "
                  << actual_slope << "\n";
        std::cout << "    slope log(actual/coordination_target) vs log(d) = "
                  << target_ratio_slope << "\n";

        check(std::abs(actual_slope + current_distance_exponent) < 1e-12,
              "distance sweep identifies current exponent 2*(s+1)");
        check(std::abs(target_ratio_slope + reviewer_target_exponent) < 1e-12,
              "ratio to coordination target has slope -(s+1), so target does not match current code");
    }

    std::cout << "\nVERDICT: current HsOperator is not RSu Eq. 12 high-order B "
              << "and its face-average kernel falls as r^(-2*(s+1)).\n";
    std::cout << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures))
              << "\n";
    return failures == 0 ? 0 : 1;
}
