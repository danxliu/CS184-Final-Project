#include "MeshData.h"

#include "imgui.h"
#include "polyscope/options.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <Eigen/Dense>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FrameData {
    std::filesystem::path path;
    rsh::MeshData mesh;
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

void register_mesh(const rsh::MeshData &mesh,
                   const std::string &name,
                   const std::string &label) {
    if (polyscope::hasSurfaceMesh(name)) {
        polyscope::removeSurfaceMesh(name);
    }
    auto *ps_mesh = polyscope::registerSurfaceMesh(
        name, vertex_positions(mesh), face_indices(mesh));
    ps_mesh->setSmoothShade(true);
    ps_mesh->setEdgeWidth(0.25);
    ps_mesh->setSurfaceColor(glm::vec3(0.72f, 0.74f, 0.78f));
    ps_mesh->setMaterial("wax");
    polyscope::updateStructureExtents();
    polyscope::requestRedraw();
    std::cout << "loaded " << label << "  vertices=" << mesh.n_vertices()
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
           " <mesh.obj | directory-with-frame_XXXX.obj>";
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << usage(argv[0]) << "\n";
        return 1;
    }

    try {
        const std::filesystem::path input(argv[1]);
        if (!std::filesystem::exists(input)) {
            throw std::runtime_error("input path does not exist: " +
                                     input.string());
        }

        polyscope::options::programName =
            "Repulsive Shells - " + input.string();
        polyscope::options::autocenterStructures = true;
        polyscope::options::autoscaleStructures = true;
        polyscope::options::groundPlaneMode =
            polyscope::GroundPlaneMode::ShadowOnly;
        polyscope::init();

        const std::string mesh_name = "mesh";
        if (std::filesystem::is_directory(input)) {
            const std::vector<std::filesystem::path> paths = frame_paths(input);
            if (paths.empty()) {
                throw std::runtime_error("directory has no frame_XXXX.obj files: " +
                                         input.string());
            }
            std::vector<FrameData> frames = load_frames(paths);
            int active_frame = 0;
            bool playing = false;
            float fps = 12.0f;
            auto last_tick = std::chrono::steady_clock::now();

            auto show_frame = [&](int frame_idx) {
                active_frame = std::clamp(frame_idx,
                                          0,
                                          static_cast<int>(frames.size()) - 1);
                register_mesh(frames[static_cast<size_t>(active_frame)].mesh,
                              mesh_name,
                              frames[static_cast<size_t>(active_frame)]
                                  .path.filename()
                                  .string());
            };

            show_frame(0);
            polyscope::options::alwaysRedraw = true;
            polyscope::state::userCallback = [&]() {
                int requested_frame = active_frame;
                ImGui::Text("source: %s", input.string().c_str());
                ImGui::Text("frame: %s",
                            frames[static_cast<size_t>(active_frame)]
                                .path.filename()
                                .string()
                                .c_str());
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
            register_mesh(mesh, mesh_name, input.filename().string());
        }

        polyscope::show();
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
