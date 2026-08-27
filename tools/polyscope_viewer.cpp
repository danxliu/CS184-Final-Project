#include "BCT.h"
#include "BVH.h"
#include "Constraints.h"
#include "FaceGeom.h"
#include "HsPreconditioner.h"
#include "MeshData.h"
#include "TPE.h"

#include "imgui.h"
#include "polyscope/options.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/surface_vector_quantity.h"
#include "polyscope/view.h"

#include <Eigen/Dense>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kFlowQuantityName = "TPE descent flow";

enum class FlowMode {
    Off,
    Raw,
    Hs,
};

enum class ShadeMode {
    Smooth,
    Flat,
    TriFlat,
};

struct ViewerOptions {
    FlowMode flow_mode = FlowMode::Hs;
    ShadeMode shade_mode = ShadeMode::Smooth;
    double tpe_alpha = 6.0;
    double tpe_theta = 0.5;
    float flow_length_scale = 0.025f;
    float flow_radius = 0.0018f;
    bool compute_missing_flow = false;
};

struct MeshStyle {
    polyscope::MeshShadeStyle shade_style = polyscope::MeshShadeStyle::Smooth;
    polyscope::BackFacePolicy back_face_policy =
        polyscope::BackFacePolicy::Different;
    glm::vec3 surface_color = glm::vec3(0.64f, 0.70f, 0.78f);
    glm::vec3 back_face_color = glm::vec3(0.34f, 0.39f, 0.45f);
    glm::vec3 edge_color = glm::vec3(0.12f, 0.14f, 0.17f);
    double edge_width = 0.08;
    std::string material = "clay";
};

struct FrameData {
    std::filesystem::path path;
    rsh::MeshData mesh;
};

struct SceneBounds {
    Eigen::Vector3d min =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d max =
        Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    bool valid = false;
};

struct FlowCacheEntry {
    bool attempted = false;
    bool ok = false;
    std::vector<glm::vec3> vectors;
    std::string status = "TPE flow not computed";
};

std::vector<glm::vec3> vertex_positions(const rsh::MeshData &mesh) {
    std::vector<glm::vec3> out;
    out.reserve(static_cast<size_t>(mesh.n_vertices()));
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        out.push_back(glm::vec3(static_cast<float>(mesh.V(i, 0)),
                                static_cast<float>(mesh.V(i, 1)),
                                static_cast<float>(mesh.V(i, 2))));
    }
    return out;
}

std::vector<glm::vec3> vertex_vectors(const Eigen::MatrixXd &field) {
    std::vector<glm::vec3> out;
    out.reserve(static_cast<size_t>(field.rows()));
    for (int i = 0; i < field.rows(); ++i) {
        out.push_back(glm::vec3(static_cast<float>(field(i, 0)),
                                static_cast<float>(field(i, 1)),
                                static_cast<float>(field(i, 2))));
    }
    return out;
}

std::vector<std::array<size_t, 3>> face_indices(const rsh::MeshData &mesh) {
    std::vector<std::array<size_t, 3>> out;
    out.reserve(static_cast<size_t>(mesh.n_faces()));
    for (int f = 0; f < mesh.n_faces(); ++f) {
        out.push_back({static_cast<size_t>(mesh.F(f, 0)),
                       static_cast<size_t>(mesh.F(f, 1)),
                       static_cast<size_t>(mesh.F(f, 2))});
    }
    return out;
}

bool same_topology(const rsh::MeshData &a, const rsh::MeshData &b) {
    if (a.n_vertices() != b.n_vertices() || a.n_faces() != b.n_faces()) {
        return false;
    }
    if (a.F.rows() != b.F.rows() || a.F.cols() != b.F.cols()) {
        return false;
    }
    return (a.F.array() == b.F.array()).all();
}

void expand_bounds(SceneBounds &bounds, const rsh::MeshData &mesh) {
    if (mesh.n_vertices() <= 0) return;
    bounds.min = bounds.min.cwiseMin(mesh.V.colwise().minCoeff().transpose());
    bounds.max = bounds.max.cwiseMax(mesh.V.colwise().maxCoeff().transpose());
    bounds.valid = true;
}

SceneBounds bounds_for_frames(const std::vector<FrameData> &frames) {
    SceneBounds bounds;
    for (const FrameData &frame : frames) {
        expand_bounds(bounds, frame.mesh);
    }
    return bounds;
}

glm::vec3 to_glm(const Eigen::Vector3d &v) {
    return glm::vec3(static_cast<float>(v.x()),
                     static_cast<float>(v.y()),
                     static_cast<float>(v.z()));
}

polyscope::MeshShadeStyle shade_style(ShadeMode mode) {
    switch (mode) {
    case ShadeMode::Smooth:
        return polyscope::MeshShadeStyle::Smooth;
    case ShadeMode::Flat:
        return polyscope::MeshShadeStyle::Flat;
    case ShadeMode::TriFlat:
        return polyscope::MeshShadeStyle::TriFlat;
    }
    return polyscope::MeshShadeStyle::Smooth;
}

void style_primary_mesh(polyscope::SurfaceMesh *ps_mesh,
                        const MeshStyle &style) {
    ps_mesh->setShadeStyle(style.shade_style);
    ps_mesh->setSurfaceColor(style.surface_color);
    ps_mesh->setBackFacePolicy(style.back_face_policy);
    ps_mesh->setBackFaceColor(style.back_face_color);
    ps_mesh->setEdgeColor(style.edge_color);
    ps_mesh->setEdgeWidth(style.edge_width);
    ps_mesh->setMaterial(style.material);
}

MeshStyle default_mesh_style(const ViewerOptions &options) {
    MeshStyle style;
    style.shade_style = shade_style(options.shade_mode);
    return style;
}

MeshStyle capture_mesh_style(const std::string &mesh_name,
                             const MeshStyle &fallback) {
    if (!polyscope::hasSurfaceMesh(mesh_name)) {
        return fallback;
    }
    auto *ps_mesh = polyscope::getSurfaceMesh(mesh_name);
    MeshStyle style;
    style.shade_style = ps_mesh->getShadeStyle();
    style.back_face_policy = ps_mesh->getBackFacePolicy();
    style.surface_color = ps_mesh->getSurfaceColor();
    style.back_face_color = ps_mesh->getBackFaceColor();
    style.edge_color = ps_mesh->getEdgeColor();
    style.edge_width = ps_mesh->getEdgeWidth();
    style.material = ps_mesh->getMaterial();
    return style;
}

void style_obstacle_mesh(polyscope::SurfaceMesh *ps_mesh) {
    ps_mesh->setShadeStyle(polyscope::MeshShadeStyle::Smooth);
    ps_mesh->setSurfaceColor(glm::vec3(0.86f, 0.42f, 0.34f));
    ps_mesh->setBackFacePolicy(polyscope::BackFacePolicy::Different);
    ps_mesh->setBackFaceColor(glm::vec3(0.45f, 0.20f, 0.17f));
    ps_mesh->setEdgeWidth(0.0);
    ps_mesh->setMaterial("clay");
    ps_mesh->setTransparency(0.58f);
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

const char *flow_mode_label(FlowMode mode) {
    switch (mode) {
    case FlowMode::Off:
        return "off";
    case FlowMode::Raw:
        return "raw -grad Phi";
    case FlowMode::Hs:
        return "H^s preconditioned";
    }
    return "unknown";
}

std::filesystem::path flow_path_for_mesh(const std::filesystem::path &mesh_path) {
    const std::string stem = mesh_path.stem().string();
    if (stem.rfind("frame_", 0) == 0) {
        return mesh_path.parent_path() /
               ("flow_" + stem.substr(std::string("frame_").size()) + ".vec");
    }
    return mesh_path.parent_path() / (stem + ".flow.vec");
}

std::string next_data_line(std::istream &in) {
    std::string line;
    while (std::getline(in, line)) {
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        return line;
    }
    return "";
}

std::vector<glm::vec3> load_flow_vectors(const std::filesystem::path &path,
                                         int expected_vertices) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("could not open " + path.string());
    }

    const std::string count_line = next_data_line(in);
    if (count_line.empty()) {
        throw std::runtime_error("empty flow file " + path.string());
    }
    int count = -1;
    {
        std::istringstream iss(count_line);
        iss >> count;
    }
    if (count != expected_vertices) {
        throw std::runtime_error(
            "flow vertex count mismatch in " + path.string() +
            ": expected " + std::to_string(expected_vertices) +
            ", got " + std::to_string(count));
    }

    std::vector<glm::vec3> vectors;
    vectors.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const std::string line = next_data_line(in);
        if (line.empty()) {
            throw std::runtime_error("truncated flow file " + path.string());
        }
        std::istringstream iss(line);
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        if (!(iss >> x >> y >> z)) {
            throw std::runtime_error("invalid vector row in " + path.string());
        }
        vectors.push_back(glm::vec3(static_cast<float>(x),
                                    static_cast<float>(y),
                                    static_cast<float>(z)));
    }
    return vectors;
}

void ensure_flow_cache(const rsh::MeshData &mesh,
                       const std::filesystem::path &mesh_path,
                       const ViewerOptions &options,
                       FlowCacheEntry &cache) {
    if (cache.attempted || options.flow_mode == FlowMode::Off) {
        return;
    }

    cache.attempted = true;
    cache.ok = false;
    const std::filesystem::path sidecar = flow_path_for_mesh(mesh_path);
    try {
        if (std::filesystem::exists(sidecar)) {
            cache.vectors = load_flow_vectors(sidecar, mesh.n_vertices());
            cache.ok = true;
            cache.status = "loaded " + sidecar.filename().string();
            return;
        }
        if (!options.compute_missing_flow) {
            cache.status =
                "missing " + sidecar.filename().string() +
                " (run precompute_tpe_flow, or pass --compute-flow)";
            return;
        }

        const rsh::FaceGeom g = rsh::compute_face_geom(mesh);
        const rsh::BVH bvh = rsh::build_bvh(mesh, g);
        const rsh::BlockPairs bp = rsh::build_bct_self(bvh, options.tpe_theta);
        const double energy =
            rsh::tpe_energy_bh(g, bvh, bp, options.tpe_alpha);

        Eigen::MatrixXd gradient =
            rsh::tpe_gradient_bh(mesh, g, bvh, bp, options.tpe_alpha);
        remove_scale_mode(mesh.V, gradient);

        Eigen::MatrixXd direction;
        int gmres_iters = 0;
        double gmres_error = 0.0;
        bool used_identity_fallback = false;
        if (options.flow_mode == FlowMode::Hs) {
            rsh::HsPreconditionerParams hs_params;
            hs_params.s = 5.0 / 3.0;
            hs_params.sigma = 1.0;
            hs_params.mass_weight = 0.0;
            hs_params.theta = 0.25;

            rsh::HsConstraints constraints;
            constraints.pin_barycenter = true;

            const rsh::HsDirectionResult hs =
                rsh::hs_preconditioned_direction(
                    mesh, gradient, hs_params, constraints);
            direction = hs.direction;
            gmres_iters = hs.max_gmres_iterations;
            gmres_error = hs.max_gmres_error;
            used_identity_fallback = hs.used_identity_fallback;
        } else {
            direction = gradient;
            rsh::project_barycenter(direction);
        }

        remove_scale_mode(mesh.V, direction);
        rsh::project_barycenter(direction);
        const Eigen::MatrixXd descent = -direction;
        cache.vectors = vertex_vectors(descent);
        cache.ok = true;

        std::ostringstream oss;
        oss << flow_mode_label(options.flow_mode)
            << " flow: E=" << std::setprecision(6) << energy
            << ", |grad|=" << gradient.norm();
        if (options.flow_mode == FlowMode::Hs) {
            oss << ", GMRES iters=" << gmres_iters
                << ", err=" << gmres_error;
            if (used_identity_fallback) {
                oss << ", identity fallback";
            }
        }
        cache.status = oss.str();
    } catch (const std::exception &e) {
        cache.vectors.clear();
        cache.status = std::string("TPE flow failed: ") + e.what();
    }
}

void apply_flow_quantity(const std::string &mesh_name,
                         const FlowCacheEntry *cache,
                         const ViewerOptions &options,
                         bool enabled) {
    if (!polyscope::hasSurfaceMesh(mesh_name)) return;
    auto *ps_mesh = polyscope::getSurfaceMesh(mesh_name);
    if (!enabled || cache == nullptr || !cache->ok) {
        if (auto *q = ps_mesh->getQuantity(kFlowQuantityName)) {
            q->setEnabled(false);
        }
        polyscope::requestRedraw();
        return;
    }

    auto *q = dynamic_cast<polyscope::SurfaceVertexVectorQuantity *>(
        ps_mesh->getQuantity(kFlowQuantityName));
    if (q != nullptr) {
        q->updateData(cache->vectors);
    } else {
        q = ps_mesh->addVertexVectorQuantity(
            kFlowQuantityName, cache->vectors, polyscope::VectorType::STANDARD);
        q->setVectorColor(glm::vec3(0.97f, 0.64f, 0.18f));
        q->setVectorLengthScale(options.flow_length_scale, true);
        q->setVectorRadius(options.flow_radius, true);
        q->setMaterial("wax");
    }
    q->setEnabled(true);
    polyscope::requestRedraw();
}

FlowMode parse_flow_mode(const std::string &value) {
    if (value == "off" || value == "none" || value == "0") {
        return FlowMode::Off;
    }
    if (value == "raw" || value == "gradient" || value == "l2") {
        return FlowMode::Raw;
    }
    if (value == "hs" || value == "preconditioned" || value == "h") {
        return FlowMode::Hs;
    }
    throw std::runtime_error("unknown flow mode: " + value);
}

ShadeMode parse_shade_mode(const std::string &value) {
    if (value == "smooth") return ShadeMode::Smooth;
    if (value == "flat") return ShadeMode::Flat;
    if (value == "tri-flat" || value == "triflat" || value == "tri") {
        return ShadeMode::TriFlat;
    }
    throw std::runtime_error("unknown shade mode: " + value);
}

// First-time registration; sets material/color/etc. once.
void register_mesh(const rsh::MeshData &mesh,
                   const std::string &name,
                   const std::string &label,
                   const MeshStyle &style) {
    if (polyscope::hasSurfaceMesh(name)) {
        polyscope::removeSurfaceMesh(name);
    }
    auto *ps_mesh = polyscope::registerSurfaceMesh(
        name, vertex_positions(mesh), face_indices(mesh));
    style_primary_mesh(ps_mesh, style);
    polyscope::updateStructureExtents();
    polyscope::requestRedraw();
    std::cout << "loaded " << label << "  vertices=" << mesh.n_vertices()
              << " faces=" << mesh.n_faces() << "\n";
}

// Update vertex positions in-place. Topology must match the original
// registration. Avoids triggering extent recomputation, which would
// re-fit the camera and mask translational motion across frames.
void update_mesh_positions(const rsh::MeshData &mesh,
                           const std::string &name) {
    auto *ps_mesh = polyscope::getSurfaceMesh(name);
    ps_mesh->updateVertexPositions(vertex_positions(mesh));
    polyscope::requestRedraw();
}

void set_mesh_enabled(const std::string &name, bool enabled) {
    if (!polyscope::hasSurfaceMesh(name)) return;
    polyscope::getSurfaceMesh(name)->setEnabled(enabled);
    polyscope::requestRedraw();
}

// Static obstacle mesh dumped alongside the frames (e.g. capsule tube
// from demo_phase3_ball_tube). Registered once with a contrasting color.
void register_obstacle(const std::filesystem::path &obstacle_path) {
    if (!std::filesystem::exists(obstacle_path)) return;
    rsh::MeshData mesh = rsh::MeshData::load_obj(obstacle_path.string());
    auto *ps_mesh = polyscope::registerSurfaceMesh(
        "obstacle", vertex_positions(mesh), face_indices(mesh));
    style_obstacle_mesh(ps_mesh);
    std::cout << "loaded obstacle " << obstacle_path.filename().string()
              << "  vertices=" << mesh.n_vertices()
              << " faces=" << mesh.n_faces() << "\n";
}

bool is_frame_obj(const std::filesystem::directory_entry &entry) {
    if (!entry.is_regular_file()) return false;
    const std::string name = entry.path().filename().string();
    return name.rfind("frame_", 0) == 0 && entry.path().extension() == ".obj";
}

std::vector<std::filesystem::path> frame_paths(const std::filesystem::path &dir) {
    std::vector<std::filesystem::path> out;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (is_frame_obj(entry)) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<FrameData> load_frames(const std::vector<std::filesystem::path> &paths) {
    std::vector<FrameData> frames;
    frames.reserve(paths.size());
    for (const auto &path : paths) {
        frames.push_back(FrameData{path, rsh::MeshData::load_obj(path.string())});
    }
    return frames;
}

std::string usage(const char *argv0) {
    return std::string("Usage: ") + argv0 +
           " <mesh.obj | directory-with-frame_XXXX.obj> [--fps N] "
           "[--zoom-out N] [--flow hs|raw|off] [--flow-scale N] "
           "[--flow-radius N] [--shade smooth|flat|tri-flat] "
           "[--compute-flow]";
}

// Place the camera from an explicit scene bounding box instead of relying on
// Polyscope's relative zoom state, which is sensitive to when structures are
// registered and when the first render loop tick runs.
void apply_initial_view(const SceneBounds &bounds, float zoom_amount) {
    if (!bounds.valid) {
        polyscope::view::resetCameraToHomeView();
        return;
    }

    const Eigen::Vector3d center = 0.5 * (bounds.min + bounds.max);
    const double diag = (bounds.max - bounds.min).norm();
    const float scene_scale =
        static_cast<float>(std::max(diag, 1e-3));
    const float distance =
        scene_scale * std::max(1.0f, zoom_amount);
    const glm::vec3 target = to_glm(center);
    const glm::vec3 view_dir =
        glm::normalize(glm::vec3(0.62f, -1.15f, 0.48f));
    const glm::vec3 camera = target + distance * view_dir;

    polyscope::view::setViewCenterRaw(target);
    polyscope::view::lookAt(camera, target, glm::vec3(0.0f, 0.0f, 1.0f), false);
}

} // namespace

int main(int argc, char **argv) {
    std::filesystem::path input;
    float initial_fps = 12.0f;
    float zoom_out_amount = 4.0f;
    ViewerOptions viewer_options;
    bool got_input = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fps" && i + 1 < argc) {
            initial_fps = std::stof(argv[++i]);
        } else if (arg == "--zoom-out" && i + 1 < argc) {
            zoom_out_amount = std::stof(argv[++i]);
        } else if (arg == "--flow" && i + 1 < argc) {
            viewer_options.flow_mode = parse_flow_mode(argv[++i]);
        } else if (arg == "--no-flow") {
            viewer_options.flow_mode = FlowMode::Off;
        } else if (arg == "--flow-scale" && i + 1 < argc) {
            viewer_options.flow_length_scale = std::stof(argv[++i]);
        } else if (arg == "--flow-radius" && i + 1 < argc) {
            viewer_options.flow_radius = std::stof(argv[++i]);
        } else if (arg == "--shade" && i + 1 < argc) {
            viewer_options.shade_mode = parse_shade_mode(argv[++i]);
        } else if (arg == "--compute-flow") {
            viewer_options.compute_missing_flow = true;
        } else if (!got_input) {
            input = arg;
            got_input = true;
        } else {
            std::cerr << usage(argv[0]) << "\n";
            return 1;
        }
    }
    if (!got_input) {
        std::cerr << usage(argv[0]) << "\n";
        return 1;
    }

    try {
        if (!std::filesystem::exists(input)) {
            throw std::runtime_error("input path does not exist: " +
                                     input.string());
        }

        polyscope::options::programName =
            "Repulsive Shells - " + input.string();
        // Preserve world-space registration between frame meshes and static
        // obstacles. Per-structure centering/scaling makes a ball outside a
        // tube display on top of the tube.
        polyscope::options::autocenterStructures = false;
        polyscope::options::autoscaleStructures = false;
        polyscope::options::groundPlaneMode =
            polyscope::GroundPlaneMode::ShadowOnly;
        polyscope::options::shadowDarkness = 0.18f;
        polyscope::options::shadowBlurIters = 3;
        polyscope::options::ssaaFactor = 2;
        polyscope::init();
        polyscope::view::bgColor =
            std::array<float, 4>{0.045f, 0.050f, 0.057f, 1.0f};

        const std::string mesh_name = "mesh";
        // Frame-mode state lives at function scope so polyscope::show()'s
        // userCallback (which captures by reference) outlives its first fire.
        std::vector<FrameData> frames;
        std::vector<FlowCacheEntry> frame_flow_cache;
        FlowCacheEntry single_flow_cache;
        SceneBounds initial_bounds;
        MeshStyle active_mesh_style = default_mesh_style(viewer_options);
        rsh::MeshData registered_mesh;
        std::string registered_mesh_name = mesh_name;
        int active_frame = 0;
        bool playing = false;
        float fps = initial_fps;
        bool zoom_pending = true;
        bool flow_visible = viewer_options.flow_mode != FlowMode::Off;
        std::string flow_status =
            viewer_options.flow_mode == FlowMode::Off
                ? "TPE flow disabled"
                : "TPE flow not computed";
        auto last_tick = std::chrono::steady_clock::now();

        if (std::filesystem::is_directory(input)) {
            const std::vector<std::filesystem::path> paths = frame_paths(input);
            if (paths.empty()) {
                throw std::runtime_error("directory has no frame_XXXX.obj files: " +
                                         input.string());
            }
            frames = load_frames(paths);
            initial_bounds = bounds_for_frames(frames);
            const std::filesystem::path obstacle_path = input / "obstacle.obj";
            if (std::filesystem::exists(obstacle_path)) {
                expand_bounds(initial_bounds,
                              rsh::MeshData::load_obj(obstacle_path.string()));
            }
            frame_flow_cache.resize(frames.size());
            register_obstacle(obstacle_path);

            // First registration sets up material + camera fit on the
            // initial frame. Frames with unchanged topology update positions
            // in place so translational motion remains visible. Remeshed
            // frames must be re-registered because Polyscope validates both
            // topology and vertex-array sizes.
            register_mesh(frames[0].mesh, mesh_name,
                          frames[0].path.filename().string(),
                          active_mesh_style);
            registered_mesh = frames[0].mesh;
            registered_mesh_name = mesh_name;

            auto show_frame = [&](int frame_idx) {
                active_frame = std::clamp(frame_idx,
                                          0,
                                          static_cast<int>(frames.size()) - 1);
                const FrameData &frame =
                    frames[static_cast<size_t>(active_frame)];
                if (same_topology(registered_mesh, frame.mesh)) {
                    update_mesh_positions(frame.mesh, registered_mesh_name);
                } else {
                    active_mesh_style =
                        capture_mesh_style(registered_mesh_name,
                                           active_mesh_style);
                    set_mesh_enabled(registered_mesh_name, false);
                    registered_mesh_name =
                        mesh_name + "_frame_" + std::to_string(active_frame);
                    if (polyscope::hasSurfaceMesh(registered_mesh_name)) {
                        set_mesh_enabled(registered_mesh_name, true);
                        style_primary_mesh(
                            polyscope::getSurfaceMesh(registered_mesh_name),
                            active_mesh_style);
                        update_mesh_positions(frame.mesh, registered_mesh_name);
                    } else {
                        register_mesh(frame.mesh,
                                      registered_mesh_name,
                                      frame.path.filename().string(),
                                      active_mesh_style);
                    }
                    registered_mesh = frame.mesh;
                }

                FlowCacheEntry *flow_cache =
                    &frame_flow_cache[static_cast<size_t>(active_frame)];
                if (viewer_options.flow_mode != FlowMode::Off &&
                    flow_visible) {
                    ensure_flow_cache(frame.mesh,
                                      frame.path,
                                      viewer_options,
                                      *flow_cache);
                    flow_status = flow_cache->status;
                    apply_flow_quantity(registered_mesh_name,
                                        flow_cache,
                                        viewer_options,
                                        true);
                } else {
                    flow_status = "TPE flow disabled";
                    apply_flow_quantity(registered_mesh_name,
                                        nullptr,
                                        viewer_options,
                                        false);
                }
            };

            show_frame(0);
            polyscope::options::alwaysRedraw = true;
            polyscope::state::userCallback = [&, show_frame, input]() mutable {
                if (zoom_pending) {
                    apply_initial_view(initial_bounds, zoom_out_amount);
                    zoom_pending = false;
                }
                int requested_frame = active_frame;
                ImGui::Text("source: %s", input.string().c_str());
                ImGui::Text("frame: %s",
                            frames[static_cast<size_t>(active_frame)]
                                .path.filename()
                                .string()
                                .c_str());
                if (viewer_options.flow_mode != FlowMode::Off) {
                    const bool old_flow_visible = flow_visible;
                    ImGui::Checkbox("TPE flow", &flow_visible);
                    ImGui::TextWrapped("%s", flow_status.c_str());
                    if (flow_visible != old_flow_visible) {
                        show_frame(active_frame);
                    }
                }
                if (ImGui::SliderInt("frame index",
                                     &requested_frame,
                                     0,
                                     static_cast<int>(frames.size()) - 1) &&
                    requested_frame != active_frame) {
                    show_frame(requested_frame);
                }
                if (ImGui::Button("previous")) {
                    show_frame(std::max(0, active_frame - 1));
                }
                ImGui::SameLine();
                if (ImGui::Button("next")) {
                    show_frame(std::min(static_cast<int>(frames.size()) - 1,
                                        active_frame + 1));
                }
                ImGui::Checkbox("play", &playing);
                ImGui::SliderFloat("fps", &fps, 1.0f, 60.0f, "%.1f");

                if (playing && frames.size() > 1) {
                    const auto now = std::chrono::steady_clock::now();
                    const double dt =
                        std::chrono::duration<double>(now - last_tick).count();
                    if (dt >= 1.0 / static_cast<double>(std::max(fps, 1.0f))) {
                        show_frame((active_frame + 1) %
                                   static_cast<int>(frames.size()));
                        last_tick = now;
                    }
                } else {
                    last_tick = std::chrono::steady_clock::now();
                }
            };
        } else {
            rsh::MeshData mesh = rsh::MeshData::load_obj(input.string());
            expand_bounds(initial_bounds, mesh);
            register_mesh(mesh, mesh_name, input.filename().string(),
                          active_mesh_style);
            if (viewer_options.flow_mode != FlowMode::Off && flow_visible) {
                ensure_flow_cache(mesh, input, viewer_options, single_flow_cache);
                flow_status = single_flow_cache.status;
                apply_flow_quantity(mesh_name,
                                    &single_flow_cache,
                                    viewer_options,
                                    true);
            }
            polyscope::state::userCallback = [&]() {
                if (zoom_pending) {
                    apply_initial_view(initial_bounds, zoom_out_amount);
                    zoom_pending = false;
                }
                if (viewer_options.flow_mode != FlowMode::Off) {
                    const bool old_flow_visible = flow_visible;
                    ImGui::Checkbox("TPE flow", &flow_visible);
                    ImGui::TextWrapped("%s", flow_status.c_str());
                    if (flow_visible != old_flow_visible) {
                        apply_flow_quantity(mesh_name,
                                            &single_flow_cache,
                                            viewer_options,
                                            flow_visible);
                    }
                }
            };
        }

        polyscope::show();
        polyscope::state::userCallback = nullptr;
        polyscope::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "polyscope_viewer: " << e.what() << "\n";
        if (polyscope::isInitialized()) {
            polyscope::shutdown();
        }
        return 1;
    }
    return 0;
}
