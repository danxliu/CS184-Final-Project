#include "TestMeshes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace rsh {

namespace {

void icosahedron_base(std::vector<Eigen::Vector3d> &verts,
                      std::vector<Eigen::Vector3i> &faces) {
    const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
    const double s = 1.0 / std::sqrt(1.0 + phi * phi);
    const double a = s;
    const double b = phi * s;

    verts.clear();
    verts.push_back(Eigen::Vector3d(-a,  b,  0));
    verts.push_back(Eigen::Vector3d( a,  b,  0));
    verts.push_back(Eigen::Vector3d(-a, -b,  0));
    verts.push_back(Eigen::Vector3d( a, -b,  0));
    verts.push_back(Eigen::Vector3d( 0, -a,  b));
    verts.push_back(Eigen::Vector3d( 0,  a,  b));
    verts.push_back(Eigen::Vector3d( 0, -a, -b));
    verts.push_back(Eigen::Vector3d( 0,  a, -b));
    verts.push_back(Eigen::Vector3d( b,  0, -a));
    verts.push_back(Eigen::Vector3d( b,  0,  a));
    verts.push_back(Eigen::Vector3d(-b,  0, -a));
    verts.push_back(Eigen::Vector3d(-b,  0,  a));

    const int f[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    };
    faces.clear();
    faces.reserve(20);
    for (int i = 0; i < 20; ++i) {
        faces.push_back(Eigen::Vector3i(f[i][0], f[i][1], f[i][2]));
    }
}

int get_midpoint(int a, int b,
                 std::unordered_map<std::uint64_t, int> &cache,
                 std::vector<Eigen::Vector3d> &verts) {
    const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) |
                              static_cast<std::uint64_t>(std::max(a, b));
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const Eigen::Vector3d m = (verts[a] + verts[b]).normalized();
    const int idx = static_cast<int>(verts.size());
    verts.push_back(m);
    cache.emplace(key, idx);
    return idx;
}

int wrap_index(int i, int n) {
    int out = i % n;
    if (out < 0) out += n;
    return out;
}

int torus_grid_index(int i, int j, int nu, int nv) {
    return wrap_index(i, nu) * nv + wrap_index(j, nv);
}

std::array<int, 6> torus_vertex_star_boundary(int i,
                                              int j,
                                              int nu,
                                              int nv) {
    // Boundary of the triangulated one-ring after removing vertex (i,j) and
    // its incident faces. The diagonal edges follow make_torus()'s quad split.
    return {
        torus_grid_index(i,     j - 1, nu, nv),
        torus_grid_index(i + 1, j,     nu, nv),
        torus_grid_index(i + 1, j + 1, nu, nv),
        torus_grid_index(i,     j + 1, nu, nv),
        torus_grid_index(i - 1, j,     nu, nv),
        torus_grid_index(i - 1, j - 1, nu, nv),
    };
}

bool face_uses_removed_vertex(const Eigen::Vector3i &face,
                              const std::vector<char> &removed) {
    return removed[static_cast<size_t>(face[0])] ||
           removed[static_cast<size_t>(face[1])] ||
           removed[static_cast<size_t>(face[2])];
}

std::array<int, 6> remapped_cap_loop(int torus,
                                     const std::array<int, 6> &local_loop,
                                     int base_nv,
                                     const std::vector<int> &remap) {
    std::array<int, 6> out;
    const int offset = torus * base_nv;
    for (int k = 0; k < 6; ++k) {
        const int mapped = remap[static_cast<size_t>(offset + local_loop[k])];
        if (mapped < 0) {
            throw std::runtime_error(
                "make_n_torus: cap boundary touched a removed vertex");
        }
        out[k] = mapped;
    }
    return out;
}

std::array<int, 6> best_aligned_loop(const std::array<int, 6> &a,
                                     const std::array<int, 6> &b,
                                     const Eigen::MatrixXd &V) {
    std::array<int, 6> best = b;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int reversed = 0; reversed < 2; ++reversed) {
        for (int shift = 0; shift < 6; ++shift) {
            std::array<int, 6> candidate;
            double cost = 0.0;
            for (int k = 0; k < 6; ++k) {
                const int idx = reversed
                                    ? wrap_index(shift - k, 6)
                                    : wrap_index(shift + k, 6);
                candidate[k] = b[static_cast<size_t>(idx)];
                cost += (V.row(a[k]) - V.row(candidate[k])).squaredNorm();
            }
            if (cost < best_cost) {
                best_cost = cost;
                best = candidate;
            }
        }
    }
    return best;
}

} // namespace

MeshData make_icosphere(int subdivisions) {
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> faces;
    icosahedron_base(verts, faces);

    for (int s = 0; s < subdivisions; ++s) {
        std::unordered_map<std::uint64_t, int> cache;
        std::vector<Eigen::Vector3i> next;
        next.reserve(faces.size() * 4);
        for (const auto &t : faces) {
            const int a = get_midpoint(t[0], t[1], cache, verts);
            const int b = get_midpoint(t[1], t[2], cache, verts);
            const int c = get_midpoint(t[2], t[0], cache, verts);
            next.push_back(Eigen::Vector3i(t[0], a, c));
            next.push_back(Eigen::Vector3i(t[1], b, a));
            next.push_back(Eigen::Vector3i(t[2], c, b));
            next.push_back(Eigen::Vector3i(a, b, c));
        }
        faces = std::move(next);
    }

    MeshData mesh;
    mesh.V.resize(static_cast<int>(verts.size()), 3);
    mesh.N.resize(static_cast<int>(verts.size()), 3);
    for (size_t i = 0; i < verts.size(); ++i) {
        mesh.V.row(i) = verts[i].transpose();
        mesh.N.row(i) = verts[i].normalized().transpose();
    }
    mesh.F.resize(static_cast<int>(faces.size()), 3);
    for (size_t i = 0; i < faces.size(); ++i) mesh.F.row(i) = faces[i].transpose();
    return mesh;
}

MeshData make_torus(double R, double r, int nu, int nv) {
    MeshData mesh;
    mesh.V.resize(nu * nv, 3);
    mesh.N.resize(nu * nv, 3);
    for (int i = 0; i < nu; ++i) {
        const double u = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(nu);
        const double cu = std::cos(u), su = std::sin(u);
        for (int j = 0; j < nv; ++j) {
            const double v = 2.0 * M_PI * static_cast<double>(j) / static_cast<double>(nv);
            const double cv = std::cos(v), sv = std::sin(v);
            const int idx = i * nv + j;
            mesh.V(idx, 0) = (R + r * cv) * cu;
            mesh.V(idx, 1) = (R + r * cv) * su;
            mesh.V(idx, 2) = r * sv;
            mesh.N(idx, 0) = cv * cu;
            mesh.N(idx, 1) = cv * su;
            mesh.N(idx, 2) = sv;
        }
    }

    mesh.F.resize(2 * nu * nv, 3);
    for (int i = 0; i < nu; ++i) {
        const int i1 = (i + 1) % nu;
        for (int j = 0; j < nv; ++j) {
            const int j1 = (j + 1) % nv;
            const int a = i * nv + j;
            const int b = i1 * nv + j;
            const int c = i1 * nv + j1;
            const int d = i * nv + j1;
            const int fidx = 2 * (i * nv + j);
            mesh.F.row(fidx + 0) << a, b, c;
            mesh.F.row(fidx + 1) << a, c, d;
        }
    }
    return mesh;
}

MeshData make_n_torus(int genus,
                      double R,
                      double r,
                      int nu_per_torus,
                      int nv) {
    if (genus < 1) {
        throw std::runtime_error("make_n_torus: genus must be at least 1");
    }
    if (R <= 0.0 || r <= 0.0 || R <= r) {
        throw std::runtime_error("make_n_torus: require R > r > 0");
    }
    if (nu_per_torus < 6 || nv < 4) {
        throw std::runtime_error(
            "make_n_torus: require nu_per_torus >= 6 and nv >= 4");
    }

    if (genus == 1) {
        MeshData torus = make_torus(R, r, nu_per_torus, nv);
        torus.L0 = torus.compute_L0();
        return torus;
    }

    const MeshData base = make_torus(R, r, nu_per_torus, nv);
    const int base_nv = base.n_vertices();
    const int base_nf = base.n_faces();
    const int total_base_vertices = genus * base_nv;
    const int right_i = 0;
    const int left_i = nu_per_torus / 2;
    const int cap_j = 0;
    const int right_cap_vertex =
        torus_grid_index(right_i, cap_j, nu_per_torus, nv);
    const int left_cap_vertex =
        torus_grid_index(left_i, cap_j, nu_per_torus, nv);
    if (right_cap_vertex == left_cap_vertex) {
        throw std::runtime_error("make_n_torus: cap vertices overlap");
    }

    const std::array<int, 6> right_loop_local =
        torus_vertex_star_boundary(right_i, cap_j, nu_per_torus, nv);
    const std::array<int, 6> left_loop_local =
        torus_vertex_star_boundary(left_i, cap_j, nu_per_torus, nv);

    std::vector<std::vector<char>> removed(
        static_cast<size_t>(genus),
        std::vector<char>(static_cast<size_t>(base_nv), 0));
    for (int torus = 0; torus < genus; ++torus) {
        if (torus + 1 < genus) {
            removed[static_cast<size_t>(torus)]
                   [static_cast<size_t>(right_cap_vertex)] = 1;
        }
        if (torus > 0) {
            removed[static_cast<size_t>(torus)]
                   [static_cast<size_t>(left_cap_vertex)] = 1;
        }
    }

    const double bridge_gap = std::max(2.0 * r,
                                       8.0 * r /
                                           static_cast<double>(nu_per_torus));
    const double spacing = 2.0 * (R + r) + bridge_gap;

    MeshData mesh;
    mesh.N.resize(0, 3);
    std::vector<Eigen::RowVector3d> vertices;
    vertices.reserve(static_cast<size_t>(total_base_vertices));
    std::vector<int> remap(static_cast<size_t>(total_base_vertices), -1);
    for (int torus = 0; torus < genus; ++torus) {
        const Eigen::RowVector3d shift(torus * spacing, 0.0, 0.0);
        const int offset = torus * base_nv;
        const auto &removed_torus = removed[static_cast<size_t>(torus)];
        for (int v = 0; v < base_nv; ++v) {
            if (removed_torus[static_cast<size_t>(v)]) continue;
            remap[static_cast<size_t>(offset + v)] =
                static_cast<int>(vertices.size());
            vertices.push_back(base.V.row(v) + shift);
        }
    }

    mesh.V.resize(static_cast<int>(vertices.size()), 3);
    for (int v = 0; v < static_cast<int>(vertices.size()); ++v) {
        mesh.V.row(v) = vertices[static_cast<size_t>(v)];
    }

    std::vector<Eigen::Vector3i> faces;
    faces.reserve(static_cast<size_t>(genus * (base_nf - 12) +
                                      12 * (genus - 1)));
    for (int torus = 0; torus < genus; ++torus) {
        const int offset = torus * base_nv;
        const auto &removed_torus = removed[static_cast<size_t>(torus)];
        for (int f = 0; f < base_nf; ++f) {
            const Eigen::Vector3i local_face = base.F.row(f).transpose();
            if (face_uses_removed_vertex(local_face, removed_torus)) continue;
            const int a = remap[static_cast<size_t>(offset + local_face[0])];
            const int b = remap[static_cast<size_t>(offset + local_face[1])];
            const int c = remap[static_cast<size_t>(offset + local_face[2])];
            if (a < 0 || b < 0 || c < 0) {
                throw std::runtime_error(
                    "make_n_torus: invalid remap for retained face");
            }
            faces.emplace_back(a, b, c);
        }
    }

    for (int bridge = 0; bridge + 1 < genus; ++bridge) {
        const std::array<int, 6> a =
            remapped_cap_loop(bridge, right_loop_local, base_nv, remap);
        const std::array<int, 6> b_raw =
            remapped_cap_loop(bridge + 1, left_loop_local, base_nv, remap);
        const std::array<int, 6> b = best_aligned_loop(a, b_raw, mesh.V);
        for (int i = 0; i < 6; ++i) {
            const int j = (i + 1) % 6;
            faces.emplace_back(a[i], a[j], b[j]);
            faces.emplace_back(a[i], b[j], b[i]);
        }
    }

    mesh.F.resize(static_cast<int>(faces.size()), 3);
    for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
        mesh.F.row(f) = faces[static_cast<size_t>(f)].transpose();
    }
    mesh.L0 = mesh.compute_L0();
    return mesh;
}

} // namespace rsh
