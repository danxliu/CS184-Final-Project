#include "TPE.h"

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

} // namespace

int main(int argc, char **argv) {
    const double alpha = 6.0;
    const int max_leaf_size = 1;
    const std::vector<double> thetas = {0.0, 0.25, 0.5};

    std::vector<std::filesystem::path> meshes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            meshes.emplace_back(argv[i]);
        }
    } else {
        // Repulsor BCT0 showed an order-dependent NaN on torus_12x8 when
        // that smaller mesh was evaluated after the icospheres in one process.
        meshes = {
            "assets/torus_12x8.obj",
            "assets/icosphere_2.obj",
            "assets/icosphere_3.obj",
        };
    }

    std::filesystem::create_directories("out/cross_val_repulsor");
    std::ofstream csv("out/cross_val_repulsor/results.csv");
    if (!csv) {
        std::cerr << "failed to open out/cross_val_repulsor/results.csv\n";
        return 1;
    }

    std::cout << std::setprecision(17);
    csv << std::setprecision(17);
    write_header(std::cout);
    write_header(csv);

    bool ok = true;
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

    return ok ? 0 : 2;
}
