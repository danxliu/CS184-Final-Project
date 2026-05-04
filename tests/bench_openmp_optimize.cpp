#include "OptimizeTPE.h"
#include "TestMeshes.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int parse_int_arg(int argc,
                  char **argv,
                  const std::string &name,
                  int default_value) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return std::stoi(argv[++i]);
        }
        const std::string prefix = name + "=";
        if (arg.rfind(prefix, 0) == 0) {
            return std::stoi(arg.substr(prefix.size()));
        }
    }
    return default_value;
}

double parse_double_arg(int argc,
                        char **argv,
                        const std::string &name,
                        double default_value) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return std::stod(argv[++i]);
        }
        const std::string prefix = name + "=";
        if (arg.rfind(prefix, 0) == 0) {
            return std::stod(arg.substr(prefix.size()));
        }
    }
    return default_value;
}

rsh::MeshData make_perturbed_icosphere(int subdiv) {
    rsh::MeshData mesh = rsh::make_icosphere(subdiv);
    mesh.normalize();
    const double amp = 0.35 * mesh.L0;
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        Eigen::Vector3d p = mesh.V.row(i).transpose();
        const double s = std::sin(1.37 * p.x() - 2.11 * p.y() + 0.71 * p.z());
        mesh.V.row(i) += (amp * s * p.normalized()).transpose();
    }
    mesh.V.col(0) *= 1.45;
    mesh.V.col(1) *= 0.85;
    mesh.V.col(2) *= 0.80;
    mesh.normalize();
    return mesh;
}

} // namespace

int main(int argc, char **argv) {
    const int subdiv = parse_int_arg(argc, argv, "--subdiv", 4);
    const int max_iters = parse_int_arg(argc, argv, "--iters", 200);
    const double initial_tau =
        parse_double_arg(argc, argv, "--initial-tau", 0.1);

    rsh::MeshData mesh = make_perturbed_icosphere(subdiv);
    rsh::OptimizeTPEParams params;
    params.max_iters = max_iters;
    params.initial_tau = initial_tau;
    params.armijo_max_backtracks = 80;
    params.out_dir = "";
    params.dump_every_iter = true;

#ifdef _OPENMP
    const int omp_threads = omp_get_max_threads();
#else
    const int omp_threads = 1;
#endif

    const auto t0 = std::chrono::steady_clock::now();
    const rsh::OptimizeTPEResult result = rsh::optimize_tpe(mesh, params);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(t1 - t0).count();

    std::cout << "bench_openmp_optimize"
              << ",subdiv=" << subdiv
              << ",iters=" << max_iters
              << ",initial_tau=" << initial_tau
              << ",omp_threads=" << omp_threads
              << ",elapsed_seconds=" << elapsed
              << ",iterations_completed=" << result.iterations_completed
              << ",remeshes=" << result.remeshes_completed
              << ",final_energy=" << result.final_energy
              << ",final_grad_norm=" << result.final_grad_norm
              << ",stop_reason=" << result.stop_reason
              << "\n";
    return 0;
}
