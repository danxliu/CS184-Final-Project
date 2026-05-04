#include "TestMeshes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
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

int closest_face_to_point(const MeshData &mesh, const Eigen::Vector3d &target) {
    int best_face = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int f = 0; f < mesh.n_faces(); ++f) {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int k = 0; k < 3; ++k) {
            centroid += mesh.V.row(mesh.F(f, k)).transpose();
        }
        centroid /= 3.0;
        const double d2 = (centroid - target).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best_face = f;
        }
    }
    return best_face;
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
    const int right_face =
        closest_face_to_point(base, Eigen::Vector3d(R + r, 0.0, 0.0));
    const int left_face =
        closest_face_to_point(base, Eigen::Vector3d(-(R + r), 0.0, 0.0));
    if (right_face < 0 || left_face < 0 || right_face == left_face) {
        throw std::runtime_error("make_n_torus: failed to choose bridge caps");
    }

    struct Bridge {
        int left_torus = 0;
        int right_torus = 0;
        Eigen::Vector3i left_loop = Eigen::Vector3i::Zero();
        Eigen::Vector3i right_loop = Eigen::Vector3i::Zero();
    };

    std::vector<std::set<int>> omitted_faces(static_cast<size_t>(genus));
    std::vector<Bridge> bridges;
    bridges.reserve(static_cast<size_t>(genus - 1));
    for (int k = 0; k + 1 < genus; ++k) {
        omitted_faces[static_cast<size_t>(k)].insert(right_face);
        omitted_faces[static_cast<size_t>(k + 1)].insert(left_face);
        Bridge bridge;
        bridge.left_torus = k;
        bridge.right_torus = k + 1;
        bridge.left_loop = base.F.row(right_face).transpose();
        bridge.right_loop = base.F.row(left_face).transpose();
        bridges.push_back(bridge);
    }

    const double bridge_gap = 4.0 * r / static_cast<double>(nu_per_torus);
    const double spacing = 2.0 * (R + r) + bridge_gap;

    MeshData mesh;
    mesh.V.resize(genus * base_nv, 3);
    mesh.N.resize(0, 3);
    for (int torus = 0; torus < genus; ++torus) {
        const Eigen::RowVector3d shift(torus * spacing, 0.0, 0.0);
        const int offset = torus * base_nv;
        for (int v = 0; v < base_nv; ++v) {
            mesh.V.row(offset + v) = base.V.row(v) + shift;
        }
    }

    std::vector<Eigen::Vector3i> faces;
    faces.reserve(static_cast<size_t>(genus * base_nf + 6 * (genus - 1)));
    for (int torus = 0; torus < genus; ++torus) {
        const int offset = torus * base_nv;
        const auto &omit = omitted_faces[static_cast<size_t>(torus)];
        for (int f = 0; f < base_nf; ++f) {
            if (omit.find(f) != omit.end()) continue;
            faces.push_back(base.F.row(f).transpose() +
                            Eigen::Vector3i::Constant(offset));
        }
    }

    for (const Bridge &bridge : bridges) {
        std::array<int, 3> a = {
            bridge.left_torus * base_nv + bridge.left_loop[0],
            bridge.left_torus * base_nv + bridge.left_loop[1],
            bridge.left_torus * base_nv + bridge.left_loop[2],
        };
        const std::array<int, 3> b_removed = {
            bridge.right_torus * base_nv + bridge.right_loop[0],
            bridge.right_torus * base_nv + bridge.right_loop[1],
            bridge.right_torus * base_nv + bridge.right_loop[2],
        };
        const std::array<int, 3> b = {b_removed[0], b_removed[2],
                                      b_removed[1]};
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3;
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
