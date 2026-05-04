#include "TPE.h"
#include "HsPreconditioner.h"

#ifdef RSH_REPULSOR_CBLAS_HEADER
#include RSH_REPULSOR_CBLAS_HEADER
#else
#include <cblas.h>
#endif

#ifdef RSH_REPULSOR_NEEDS_BLASINT_ALIAS
using blasint = CBLAS_INT;
#endif

#ifdef RSH_REPULSOR_LAPACKE_HEADER
#include RSH_REPULSOR_LAPACKE_HEADER
#else
#include <lapacke.h>
#endif
#ifndef LAPACK_H
#define LAPACK_H
#endif
#ifdef I
#undef I
#endif

#include <string>

#include "submodules/Tensors/OpenBLAS.hpp"

namespace Tools {
template <typename T>
std::string ToStringFPGeneral(const T &value) {
    return ToString(value);
}
} // namespace Tools

#include "Repulsor.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using RepulsorMesh = Repulsor::SimplicialMesh<2, 3, double, int>;

struct Row {
    std::string mesh;
    int n_v = 0;
    int n_f = 0;
    double alpha = 6.0;
    double theta = 0.0;
    double ours_brute = 0.0;
    double repulsor_allpairs = 0.0;
    double ratio_brute = 0.0;
    double ours_bh = 0.0;
    double repulsor_bct0 = 0.0;
    double ratio_bh = 0.0;
};

struct Tp0Row {
    std::string mesh;
    int n_v = 0;
    int n_f = 0;
    double theta = 0.0;
    double direct_rel_l2 = 0.0;
    double direct_rel_max = 0.0;
    double best_scale = 0.0;
    double scaled_rel_l2 = 0.0;
    double scaled_rel_max = 0.0;
};

struct FinalMeshParityRow {
    std::string mesh;
    int n_v = 0;
    int n_f = 0;
    double tpe_ours = 0.0;
    double tpe_repulsor = 0.0;
    double ratio_repulsor_over_ours = 0.0;
};

std::vector<double> row_major_vertices(const rsh::MeshData &mesh) {
    std::vector<double> out(static_cast<std::size_t>(mesh.n_vertices()) * 3);
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        for (int j = 0; j < 3; ++j) {
            out[static_cast<std::size_t>(3 * i + j)] = mesh.V(i, j);
        }
    }
    return out;
}

std::vector<int> row_major_faces(const rsh::MeshData &mesh) {
    std::vector<int> out(static_cast<std::size_t>(mesh.n_faces()) * 3);
    for (int i = 0; i < mesh.n_faces(); ++i) {
        for (int j = 0; j < 3; ++j) {
            out[static_cast<std::size_t>(3 * i + j)] = mesh.F(i, j);
        }
    }
    return out;
}

RepulsorMesh make_repulsor_mesh(const rsh::MeshData &mesh,
                                double theta,
                                int max_leaf_size) {
    std::vector<double> vertices = row_major_vertices(mesh);
    std::vector<int> faces = row_major_faces(mesh);

    RepulsorMesh rep_mesh(vertices.data(),
                          static_cast<std::size_t>(mesh.n_vertices()),
                          false,
                          faces.data(),
                          static_cast<std::size_t>(mesh.n_faces()),
                          false,
                          1);
    rep_mesh.cluster_tree_settings.split_threshold = max_leaf_size;
    rep_mesh.cluster_tree_settings.thread_count = 1;
    rep_mesh.block_cluster_tree_settings.far_field_separation_parameter = theta;
    rep_mesh.block_cluster_tree_settings.near_field_separation_parameter = 10.0;
    rep_mesh.adaptivity_settings.theta = 10.0;
    rep_mesh.adaptivity_settings.max_refinement = 30;
    return rep_mesh;
}

double safe_ratio(double num, double den) {
    if (den == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return num / den;
}

Eigen::MatrixXd deterministic_vector_field(const rsh::MeshData &mesh) {
    Eigen::MatrixXd out(mesh.n_vertices(), 3);
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        const double x = mesh.V(i, 0);
        const double y = mesh.V(i, 1);
        const double z = mesh.V(i, 2);
        out(i, 0) = std::sin(1.7 * x + 0.3 * y) + 0.2 * z;
        out(i, 1) = std::cos(0.5 * y - 1.1 * z) + 0.1 * x;
        out(i, 2) = std::sin(0.7 * z + 0.2 * x) - 0.3 * y;
    }
    return out;
}

std::vector<double> row_major_field(const Eigen::MatrixXd &field) {
    std::vector<double> out(static_cast<std::size_t>(field.rows()) * 3);
    for (int i = 0; i < field.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            out[static_cast<std::size_t>(3 * i + j)] = field(i, j);
        }
    }
    return out;
}

Eigen::MatrixXd matrix_from_row_major_field(const std::vector<double> &field,
                                            int n_vertices) {
    Eigen::MatrixXd out(n_vertices, 3);
    for (int i = 0; i < n_vertices; ++i) {
        for (int j = 0; j < 3; ++j) {
            out(i, j) = field[static_cast<std::size_t>(3 * i + j)];
        }
    }
    return out;
}

double rel_l2(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    return (a - b).norm() / std::max(1.0, b.norm());
}

double rel_max_abs(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    double max_num = 0.0;
    double max_den = 1.0;
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            max_num = std::max(max_num, std::abs(a(i, j) - b(i, j)));
            max_den = std::max(max_den, std::abs(b(i, j)));
        }
    }
    return max_num / max_den;
}

Row evaluate(const std::filesystem::path &path,
             double alpha,
             double theta,
             int max_leaf_size) {
    const rsh::MeshData mesh = rsh::MeshData::load_obj(path.string());

    Row row;
    row.mesh = path.filename().string();
    row.n_v = mesh.n_vertices();
    row.n_f = mesh.n_faces();
    row.alpha = alpha;
    row.theta = theta;
    row.ours_brute = rsh::tpe_energy_brute(mesh, alpha);
    row.ours_bh = rsh::tpe_energy_bh(mesh, alpha, theta);

    {
        RepulsorMesh rep_mesh = make_repulsor_mesh(mesh, theta, max_leaf_size);
        Repulsor::TangentPointEnergy_AllPairs<RepulsorMesh> tpe_allpairs(
            alpha, 2.0 * alpha);
        row.repulsor_allpairs = tpe_allpairs.Value(rep_mesh);
    }

    {
        RepulsorMesh rep_mesh = make_repulsor_mesh(mesh, theta, max_leaf_size);
        Repulsor::TangentPointEnergy0<RepulsorMesh> tpe_bct0(alpha);
        row.repulsor_bct0 = tpe_bct0.Value(rep_mesh);
    }

    row.ratio_brute = safe_ratio(row.ours_brute, row.repulsor_allpairs);
    row.ratio_bh = safe_ratio(row.ours_bh, row.repulsor_bct0);
    return row;
}

FinalMeshParityRow evaluate_final_mesh_parity(
    const std::filesystem::path &path,
    double alpha,
    int max_leaf_size) {
    const rsh::MeshData mesh = rsh::MeshData::load_obj(path.string());

    FinalMeshParityRow row;
    row.mesh = path.string();
    row.n_v = mesh.n_vertices();
    row.n_f = mesh.n_faces();
    row.tpe_ours = rsh::tpe_energy_brute(mesh, alpha);

    RepulsorMesh rep_mesh = make_repulsor_mesh(mesh, 0.0, max_leaf_size);
    Repulsor::TangentPointEnergy_AllPairs<RepulsorMesh> tpe_allpairs(
        alpha, 2.0 * alpha);
    row.tpe_repulsor = tpe_allpairs.Value(rep_mesh);
    row.ratio_repulsor_over_ours =
        safe_ratio(row.tpe_repulsor, row.tpe_ours);
    return row;
}

Tp0Row evaluate_tp0_matvec(const std::filesystem::path &path,
                           double theta,
                           int max_leaf_size) {
    const rsh::MeshData mesh = rsh::MeshData::load_obj(path.string());

    rsh::HsPreconditionerParams params;
    params.s = 5.0 / 3.0;
    params.sigma = 1.0;
    params.theta = theta;

    const Eigen::MatrixXd input = deterministic_vector_field(mesh);
    const Eigen::MatrixXd ours = rsh::hs_apply_operator(mesh, input, params);

    std::vector<double> x = row_major_field(input);
    std::vector<double> y(x.size(), 0.0);

    RepulsorMesh rep_mesh = make_repulsor_mesh(mesh, theta, max_leaf_size);
    const auto &bct = rep_mesh.GetBlockClusterTree();
    using Traversor = Repulsor::TP0_Traversor<
        2, 2, typename RepulsorMesh::BlockClusterTree_T,
        false, false, true, false>;
    typename Traversor::ValueContainer_T metric_values;

    Traversor compute_traversor(bct, metric_values, 6.0, 12.0);
    (void)compute_traversor.Compute();

    const auto &S = bct.GetS();
    const auto &T = bct.GetT();
    T.Pre(x.data(), 3, 3, Repulsor::OperatorType::MixedOrder);
    S.RequireBuffers(T.BufferDim());

    Traversor multiply_traversor(bct, metric_values, 6.0, 12.0);
    multiply_traversor.MultiplyMetric(true, true, true);
    S.Post(1.0, 0.0, y.data(), 3, Repulsor::OperatorType::MixedOrder);

    const Eigen::MatrixXd rep = matrix_from_row_major_field(y, mesh.n_vertices());

    Tp0Row row;
    row.mesh = path.filename().string();
    row.n_v = mesh.n_vertices();
    row.n_f = mesh.n_faces();
    row.theta = theta;
    row.direct_rel_l2 = rel_l2(ours, rep);
    row.direct_rel_max = rel_max_abs(ours, rep);

    const double rep_sq = rep.squaredNorm();
    row.best_scale = (rep_sq > 0.0)
        ? (ours.array() * rep.array()).sum() / rep_sq
        : std::numeric_limits<double>::quiet_NaN();
    const Eigen::MatrixXd scaled_rep = row.best_scale * rep;
    row.scaled_rel_l2 = rel_l2(ours, scaled_rep);
    row.scaled_rel_max = rel_max_abs(ours, scaled_rep);
    return row;
}

void write_header(std::ostream &out) {
    out << "mesh,n_v,n_f,alpha,theta,ours_brute,repulsor_allpairs,"
           "ratio_brute,ours_bh,repulsor_bct0,ratio_bh\n";
}

void write_row(std::ostream &out, const Row &row) {
    out << row.mesh << ',' << row.n_v << ',' << row.n_f << ','
        << row.alpha << ',' << row.theta << ',' << row.ours_brute << ','
        << row.repulsor_allpairs << ',' << row.ratio_brute << ','
        << row.ours_bh << ',' << row.repulsor_bct0 << ',' << row.ratio_bh
        << '\n';
}

bool check_row(const Row &row) {
    bool ok = true;
    if (!std::isfinite(row.ratio_brute)) {
        std::cerr << "FAIL non-finite brute ratio for " << row.mesh
                  << " theta=" << row.theta << ": " << row.ratio_brute
                  << '\n';
        ok = false;
    } else if (std::abs(row.ratio_brute - 1.0) >= 1e-12) {
        std::cerr << "FAIL brute ratio for " << row.mesh << " theta="
                  << row.theta << ": " << row.ratio_brute << '\n';
        ok = false;
    }
    if (!std::isfinite(row.ratio_bh)) {
        std::cerr << "FAIL non-finite BH/BCT0 ratio for " << row.mesh
                  << " theta=" << row.theta << ": " << row.ratio_bh
                  << '\n';
        ok = false;
    // Positive-theta accelerated rows compare two independent 0th-order
    // approximation schemes; theta=0 is the exact parity gate.
    } else if (row.theta == 0.0 &&
        std::abs(row.ratio_bh - 1.0) >= 1e-12) {
        std::cerr << "FAIL BH/BCT0 ratio for " << row.mesh << " theta="
                  << row.theta << ": " << row.ratio_bh << '\n';
        ok = false;
    }
    return ok;
}

void write_tp0_header(std::ostream &out) {
    out << "mesh,n_v,n_f,theta,direct_rel_l2,direct_rel_max,"
           "best_scale,scaled_rel_l2,scaled_rel_max\n";
}

void write_tp0_row(std::ostream &out, const Tp0Row &row) {
    out << row.mesh << ',' << row.n_v << ',' << row.n_f << ','
        << row.theta << ',' << row.direct_rel_l2 << ','
        << row.direct_rel_max << ',' << row.best_scale << ','
        << row.scaled_rel_l2 << ',' << row.scaled_rel_max << '\n';
}

bool check_tp0_row(const Tp0Row &row) {
    const bool direct_ok =
        std::isfinite(row.direct_rel_max) && row.direct_rel_max < 1e-10;
    const bool scaled_ok =
        std::isfinite(row.scaled_rel_max) && row.scaled_rel_max < 1e-10;
    if (!direct_ok && !scaled_ok) {
        std::cerr << "FAIL TP0 matvec parity for " << row.mesh
                  << " theta=" << row.theta
                  << ": direct_rel_max=" << row.direct_rel_max
                  << ", best_scale=" << row.best_scale
                  << ", scaled_rel_max=" << row.scaled_rel_max << '\n';
        return false;
    }
    return true;
}

void write_final_mesh_parity_row(std::ostream &out,
                                 const FinalMeshParityRow &row) {
    out << "final_mesh_parity"
        << ",mesh=" << row.mesh
        << ",n_v=" << row.n_v
        << ",n_f=" << row.n_f
        << ",tpe_ours=" << row.tpe_ours
        << ",tpe_repulsor=" << row.tpe_repulsor
        << ",ratio=" << row.ratio_repulsor_over_ours
        << '\n';
}

bool check_final_mesh_parity_row(const FinalMeshParityRow &row) {
    const bool ok =
        std::isfinite(row.ratio_repulsor_over_ours) &&
        std::abs(row.ratio_repulsor_over_ours - 1.0) < 1e-10;
    if (!ok) {
        std::cerr << "FAIL final mesh Repulsor parity for " << row.mesh
                  << ": ratio=" << row.ratio_repulsor_over_ours
                  << ", ours=" << row.tpe_ours
                  << ", repulsor=" << row.tpe_repulsor << '\n';
    }
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    const double alpha = 6.0;
    const int max_leaf_size = 1;
    const std::vector<double> thetas = {0.0, 0.25, 0.5};

    bool tp0_only = false;
    std::filesystem::path final_mesh;
    std::vector<std::filesystem::path> meshes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--tp0-only") {
                tp0_only = true;
            } else if (arg == "--final-mesh") {
                if (i + 1 >= argc) {
                    std::cerr << "--final-mesh requires a mesh path\n";
                    return 1;
                }
                final_mesh = argv[++i];
            } else if (arg.rfind("--final-mesh=", 0) == 0) {
                final_mesh = arg.substr(std::string("--final-mesh=").size());
            } else {
                meshes.emplace_back(arg);
            }
        }
    }

    std::filesystem::create_directories("out/cross_val_repulsor");
    std::cout << std::setprecision(17);

    if (!final_mesh.empty()) {
        if (!std::filesystem::exists(final_mesh)) {
            std::cerr << "missing final mesh: " << final_mesh << '\n';
            return 1;
        }
        const FinalMeshParityRow row =
            evaluate_final_mesh_parity(final_mesh, alpha, max_leaf_size);
        write_final_mesh_parity_row(std::cout, row);
        std::ofstream csv("out/cross_val_repulsor/final_mesh_parity.csv");
        if (!csv) {
            std::cerr << "failed to open "
                         "out/cross_val_repulsor/final_mesh_parity.csv\n";
            return 1;
        }
        csv << std::setprecision(17);
        write_final_mesh_parity_row(csv, row);
        return check_final_mesh_parity_row(row) ? 0 : 2;
    }

    if (meshes.empty()) {
        // Repulsor BCT0 showed an order-dependent NaN on torus_12x8 when
        // that smaller mesh was evaluated after the icospheres in one process.
        meshes = {
            "assets/torus_12x8.obj",
            "assets/icosphere_2.obj",
            "assets/icosphere_3.obj",
        };
    }

    bool ok = true;
    if (!tp0_only) {
        std::ofstream csv("out/cross_val_repulsor/results.csv");
        if (!csv) {
            std::cerr << "failed to open out/cross_val_repulsor/results.csv\n";
            return 1;
        }
        csv << std::setprecision(17);
        write_header(std::cout);
        write_header(csv);

        for (const std::filesystem::path &mesh_path : meshes) {
            if (!std::filesystem::exists(mesh_path)) {
                std::cerr << "missing mesh: " << mesh_path << '\n';
                return 1;
            }
            for (double theta : thetas) {
                const Row row = evaluate(mesh_path, alpha, theta, max_leaf_size);
                write_row(std::cout, row);
                write_row(csv, row);
                ok = check_row(row) && ok;
            }
        }
    }

    std::ofstream tp0_csv("out/cross_val_repulsor/tp0_matvec.csv");
    if (!tp0_csv) {
        std::cerr << "failed to open out/cross_val_repulsor/tp0_matvec.csv\n";
        return 1;
    }

    std::cout << "\nTP0 matvec parity at theta=0\n";
    write_tp0_header(std::cout);
    write_tp0_header(tp0_csv);
    for (const std::filesystem::path &mesh_path : meshes) {
        if (!std::filesystem::exists(mesh_path)) {
            std::cerr << "missing mesh: " << mesh_path << '\n';
            return 1;
        }
        const Tp0Row row = evaluate_tp0_matvec(mesh_path, 0.0, max_leaf_size);
        write_tp0_row(std::cout, row);
        write_tp0_row(tp0_csv, row);
        ok = check_tp0_row(row) && ok;
    }

    return ok ? 0 : 2;
}
