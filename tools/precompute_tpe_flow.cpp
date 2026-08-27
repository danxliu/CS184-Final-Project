#include "BCT.h"
#include "BVH.h"
#include "Constraints.h"
#include "FaceGeom.h"
#include "HsPreconditioner.h"
#include "MeshData.h"
#include "TPE.h"

#include <Eigen/Dense>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class FlowMode {
    Raw,
    Hs,
};

struct Options {
    std::filesystem::path input;
    FlowMode flow_mode = FlowMode::Hs;
    double tpe_alpha = 6.0;
    double tpe_theta = 0.5;
    double hs_theta = 0.25;
    bool overwrite = false;
};

bool is_frame_obj(const std::filesystem::directory_entry &entry) {
    if (!entry.is_regular_file()) return false;
    const std::string name = entry.path().filename().string();
    return name.rfind("frame_", 0) == 0 && entry.path().extension() == ".obj";
}

std::vector<std::filesystem::path> mesh_paths(const std::filesystem::path &input) {
    if (!std::filesystem::is_directory(input)) {
        return {input};
    }

    std::vector<std::filesystem::path> out;
    for (const auto &entry : std::filesystem::directory_iterator(input)) {
        if (is_frame_obj(entry)) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::filesystem::path flow_path_for_mesh(const std::filesystem::path &mesh_path) {
    const std::string stem = mesh_path.stem().string();
    if (stem.rfind("frame_", 0) == 0) {
        return mesh_path.parent_path() /
               ("flow_" + stem.substr(std::string("frame_").size()) + ".vec");
    }
    return mesh_path.parent_path() / (stem + ".flow.vec");
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

FlowMode parse_flow_mode(const std::string &value) {
    if (value == "raw" || value == "gradient" || value == "l2") {
        return FlowMode::Raw;
    }
    if (value == "hs" || value == "preconditioned" || value == "h") {
        return FlowMode::Hs;
    }
    throw std::runtime_error("unknown flow mode: " + value);
}

const char *flow_mode_name(FlowMode mode) {
    switch (mode) {
    case FlowMode::Raw:
        return "raw";
    case FlowMode::Hs:
        return "hs";
    }
    return "unknown";
}

Eigen::MatrixXd compute_descent_flow(const rsh::MeshData &mesh,
                                     const Options &opts,
                                     double &energy,
                                     double &grad_norm,
                                     int &gmres_iters) {
    const rsh::FaceGeom g = rsh::compute_face_geom(mesh);
    const rsh::BVH bvh = rsh::build_bvh(mesh, g);
    const rsh::BlockPairs bp = rsh::build_bct_self(bvh, opts.tpe_theta);
    energy = rsh::tpe_energy_bh(g, bvh, bp, opts.tpe_alpha);

    Eigen::MatrixXd gradient =
        rsh::tpe_gradient_bh(mesh, g, bvh, bp, opts.tpe_alpha);
    remove_scale_mode(mesh.V, gradient);
    grad_norm = gradient.norm();

    Eigen::MatrixXd direction;
    gmres_iters = 0;
    if (opts.flow_mode == FlowMode::Hs) {
        rsh::HsPreconditionerParams hs_params;
        hs_params.s = 5.0 / 3.0;
        hs_params.sigma = 1.0;
        hs_params.mass_weight = 0.0;
        hs_params.theta = opts.hs_theta;

        rsh::HsConstraints constraints;
        constraints.pin_barycenter = true;

        const rsh::HsDirectionResult hs =
            rsh::hs_preconditioned_direction(
                mesh, gradient, hs_params, constraints);
        direction = hs.direction;
        gmres_iters = hs.max_gmres_iterations;
    } else {
        direction = gradient;
        rsh::project_barycenter(direction);
    }

    remove_scale_mode(mesh.V, direction);
    rsh::project_barycenter(direction);
    return -direction;
}

void write_flow_file(const std::filesystem::path &path,
                     const Eigen::MatrixXd &flow,
                     const Options &opts,
                     const std::filesystem::path &source_path,
                     double energy,
                     double grad_norm,
                     int gmres_iters) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not write " + path.string());
    }

    out << "# rsh_flow_v1\n";
    out << "# source " << source_path.filename().string() << "\n";
    out << "# mode " << flow_mode_name(opts.flow_mode) << "\n";
    out << "# energy " << std::setprecision(17) << energy << "\n";
    out << "# grad_norm " << std::setprecision(17) << grad_norm << "\n";
    if (opts.flow_mode == FlowMode::Hs) {
        out << "# gmres_iters " << gmres_iters << "\n";
    }
    out << flow.rows() << "\n";
    out << std::setprecision(17);
    for (int i = 0; i < flow.rows(); ++i) {
        out << flow(i, 0) << " " << flow(i, 1) << " " << flow(i, 2) << "\n";
    }
}

std::string usage(const char *argv0) {
    return std::string("Usage: ") + argv0 +
           " <mesh.obj | directory-with-frame_XXXX.obj> "
           "[--flow hs|raw] [--overwrite] [--theta N] [--hs-theta N]";
}

Options parse_args(int argc, char **argv) {
    Options opts;
    bool got_input = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--flow" && i + 1 < argc) {
            opts.flow_mode = parse_flow_mode(argv[++i]);
        } else if (arg == "--overwrite") {
            opts.overwrite = true;
        } else if (arg == "--theta" && i + 1 < argc) {
            opts.tpe_theta = std::stod(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            opts.tpe_alpha = std::stod(argv[++i]);
        } else if (arg == "--hs-theta" && i + 1 < argc) {
            opts.hs_theta = std::stod(argv[++i]);
        } else if (!got_input) {
            opts.input = arg;
            got_input = true;
        } else {
            throw std::runtime_error(usage(argv[0]));
        }
    }
    if (!got_input) {
        throw std::runtime_error(usage(argv[0]));
    }
    if (!std::filesystem::exists(opts.input)) {
        throw std::runtime_error("input path does not exist: " +
                                 opts.input.string());
    }
    return opts;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options opts = parse_args(argc, argv);
        const std::vector<std::filesystem::path> paths = mesh_paths(opts.input);
        if (paths.empty()) {
            throw std::runtime_error("no frame_XXXX.obj files found in " +
                                     opts.input.string());
        }

        int written = 0;
        int skipped = 0;
        for (size_t i = 0; i < paths.size(); ++i) {
            const std::filesystem::path &mesh_path = paths[i];
            const std::filesystem::path out_path = flow_path_for_mesh(mesh_path);
            if (std::filesystem::exists(out_path) && !opts.overwrite) {
                ++skipped;
                std::cout << "[" << (i + 1) << "/" << paths.size()
                          << "] skip " << out_path.filename().string()
                          << " (exists)\n";
                continue;
            }

            std::cout << "[" << (i + 1) << "/" << paths.size()
                      << "] compute " << mesh_path.filename().string()
                      << " -> " << out_path.filename().string() << "\n";
            const rsh::MeshData mesh = rsh::MeshData::load_obj(mesh_path.string());
            double energy = 0.0;
            double grad_norm = 0.0;
            int gmres_iters = 0;
            const Eigen::MatrixXd flow =
                compute_descent_flow(mesh, opts, energy, grad_norm, gmres_iters);
            write_flow_file(out_path,
                            flow,
                            opts,
                            mesh_path,
                            energy,
                            grad_norm,
                            gmres_iters);
            ++written;
        }

        std::cout << "precompute_tpe_flow done: written=" << written
                  << ", skipped=" << skipped << "\n";
    } catch (const std::exception &e) {
        std::cerr << "precompute_tpe_flow: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
