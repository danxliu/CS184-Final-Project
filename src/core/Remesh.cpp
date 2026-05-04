#include "Remesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <vector>

namespace rsh {

namespace {

struct Edge {
    int a = -1;
    int b = -1;
    double length = 0.0;
};

struct MeshState {
    std::vector<Eigen::Vector3d> V;
    std::vector<Eigen::Vector3i> F;
    std::vector<bool> is_new_vertex;
};

constexpr int kMaxSimplexValence = 9;

std::pair<int, int> ordered_edge(int a, int b) {
    if (a > b) std::swap(a, b);
    return {a, b};
}

Eigen::Vector3d face_normal_raw(const std::vector<Eigen::Vector3d> &V,
                                const Eigen::Vector3i &f) {
    return (V[static_cast<size_t>(f[1])] - V[static_cast<size_t>(f[0])])
        .cross(V[static_cast<size_t>(f[2])] - V[static_cast<size_t>(f[0])]);
}

double face_area2(const std::vector<Eigen::Vector3d> &V,
                  const Eigen::Vector3i &f) {
    return face_normal_raw(V, f).norm();
}

Eigen::Vector3d face_normal_raw_mesh(const MeshData &mesh, int f) {
    const Eigen::Vector3d p0 = mesh.V.row(mesh.F(f, 0)).transpose();
    const Eigen::Vector3d p1 = mesh.V.row(mesh.F(f, 1)).transpose();
    const Eigen::Vector3d p2 = mesh.V.row(mesh.F(f, 2)).transpose();
    return (p1 - p0).cross(p2 - p0);
}

void recompute_vertex_normals(MeshData &mesh) {
    mesh.N = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
    for (int f = 0; f < mesh.n_faces(); ++f) {
        const Eigen::Vector3d n = face_normal_raw_mesh(mesh, f);
        mesh.N.row(mesh.F(f, 0)) += n.transpose();
        mesh.N.row(mesh.F(f, 1)) += n.transpose();
        mesh.N.row(mesh.F(f, 2)) += n.transpose();
    }
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        const double nn = mesh.N.row(i).norm();
        if (nn > 0.0) mesh.N.row(i) /= nn;
    }
}

MeshData finalize_meshdata(MeshData mesh, double target_l0, bool had_normals) {
    mesh.L0 = target_l0;
    if (had_normals) {
        recompute_vertex_normals(mesh);
    } else {
        mesh.N.resize(0, 0);
    }
    return mesh;
}

std::vector<Edge> collect_edges(const MeshState &state) {
    std::map<std::pair<int, int>, int> seen;
    for (const Eigen::Vector3i &f : state.F) {
        for (int k = 0; k < 3; ++k) {
            seen.emplace(ordered_edge(f[k], f[(k + 1) % 3]), 1);
        }
    }

    std::vector<Edge> out;
    out.reserve(seen.size());
    for (const auto &item : seen) {
        const int a = item.first.first;
        const int b = item.first.second;
        out.push_back({a, b, (state.V[static_cast<size_t>(a)] -
                              state.V[static_cast<size_t>(b)]).norm()});
    }
    return out;
}

std::vector<int> vertex_valences(const MeshState &state) {
    std::vector<std::set<int>> nbr(state.V.size());
    for (const Eigen::Vector3i &f : state.F) {
        for (int k = 0; k < 3; ++k) {
            const int a = f[k];
            const int b = f[(k + 1) % 3];
            nbr[static_cast<size_t>(a)].insert(b);
            nbr[static_cast<size_t>(b)].insert(a);
        }
    }

    std::vector<int> val(state.V.size(), 0);
    for (size_t i = 0; i < nbr.size(); ++i) {
        val[i] = static_cast<int>(nbr[i].size());
    }
    return val;
}

std::vector<int> vertex_simplex_valences(const MeshState &state) {
    std::vector<int> val(state.V.size(), 0);
    for (const Eigen::Vector3i &f : state.F) {
        ++val[static_cast<size_t>(f[0])];
        ++val[static_cast<size_t>(f[1])];
        ++val[static_cast<size_t>(f[2])];
    }
    return val;
}

bool contains_vertex(const Eigen::Vector3i &f, int v) {
    return f[0] == v || f[1] == v || f[2] == v;
}

std::set<int> vertex_neighbors(const MeshState &state, int v) {
    std::set<int> out;
    for (const Eigen::Vector3i &f : state.F) {
        if (!contains_vertex(f, v)) continue;
        for (int k = 0; k < 3; ++k) {
            if (f[k] != v) out.insert(f[k]);
        }
    }
    return out;
}

std::vector<int> edge_parent_faces(const MeshState &state, int a, int b) {
    std::vector<int> out;
    for (int i = 0; i < static_cast<int>(state.F.size()); ++i) {
        const Eigen::Vector3i &f = state.F[static_cast<size_t>(i)];
        if (contains_vertex(f, a) && contains_vertex(f, b)) {
            out.push_back(i);
        }
    }
    return out;
}

std::set<int> edge_opposite_vertices(const MeshState &state,
                                     const std::vector<int> &edge_faces,
                                     int a,
                                     int b) {
    std::set<int> out;
    for (int face_id : edge_faces) {
        const Eigen::Vector3i &f = state.F[static_cast<size_t>(face_id)];
        for (int k = 0; k < 3; ++k) {
            if (f[k] != a && f[k] != b) out.insert(f[k]);
        }
    }
    return out;
}

int openmesh_face_valence(const OMesh &om, OMesh::VertexHandle vh) {
    int valence = 0;
    for (auto vf_it = om.cvf_iter(vh); vf_it.is_valid(); ++vf_it) {
        ++valence;
    }
    return valence;
}

std::vector<OMesh::VertexHandle> openmesh_opposite_vertices(const OMesh &om,
                                                            OMesh::EdgeHandle eh) {
    std::vector<OMesh::VertexHandle> out;
    for (int side = 0; side < 2; ++side) {
        const auto heh = om.halfedge_handle(eh, side);
        if (!heh.is_valid()) continue;
        if (!om.face_handle(heh).is_valid()) continue;
        const auto opp = om.opposite_vh(heh);
        if (opp.is_valid()) out.push_back(opp);
    }
    return out;
}

bool split_topology_ok(const OMesh &om,
                       OMesh::EdgeHandle eh,
                       const std::vector<char> &blocked) {
    if (!eh.is_valid()) return false;

    const std::vector<OMesh::VertexHandle> opposite =
        openmesh_opposite_vertices(om, eh);
    for (const auto opp : opposite) {
        if (!opp.is_valid()) return false;
        const int idx = opp.idx();
        if (idx < 0 || idx >= static_cast<int>(blocked.size())) {
            return false;
        }
        if (blocked[static_cast<size_t>(idx)]) {
            return false;
        }
        if (openmesh_face_valence(om, opp) + 1 > kMaxSimplexValence) {
            return false;
        }
    }
    return true;
}

int split_face_delta(const OMesh &om, OMesh::EdgeHandle eh) {
    int delta = 0;
    for (int side = 0; side < 2; ++side) {
        const auto heh = om.halfedge_handle(eh, side);
        if (heh.is_valid() && om.face_handle(heh).is_valid()) {
            ++delta;
        }
    }
    return delta;
}

bool manifold_faces_ok(const MeshState &state) {
    std::map<std::pair<int, int>, int> edge_count;
    std::set<std::array<int, 3>> face_keys;
    for (const Eigen::Vector3i &f : state.F) {
        if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) {
            return false;
        }
        if (face_area2(state.V, f) <= 1e-14) {
            return false;
        }
        std::array<int, 3> key = {f[0], f[1], f[2]};
        std::sort(key.begin(), key.end());
        if (!face_keys.insert(key).second) {
            return false;
        }
        for (int k = 0; k < 3; ++k) {
            const auto e = ordered_edge(f[k], f[(k + 1) % 3]);
            const int n = ++edge_count[e];
            if (n > 2) return false;
        }
    }
    return true;
}

MeshState compact_state(const MeshState &state, int preserve_vertex) {
    std::vector<char> used(state.V.size(), 0);
    for (const Eigen::Vector3i &f : state.F) {
        used[static_cast<size_t>(f[0])] = 1;
        used[static_cast<size_t>(f[1])] = 1;
        used[static_cast<size_t>(f[2])] = 1;
    }
    if (preserve_vertex >= 0 && preserve_vertex < static_cast<int>(used.size())) {
        used[static_cast<size_t>(preserve_vertex)] = 1;
    }

    std::vector<int> remap(state.V.size(), -1);
    MeshState out;
    for (int i = 0; i < static_cast<int>(state.V.size()); ++i) {
        if (!used[static_cast<size_t>(i)]) continue;
        remap[static_cast<size_t>(i)] = static_cast<int>(out.V.size());
        out.V.push_back(state.V[static_cast<size_t>(i)]);
        out.is_new_vertex.push_back(state.is_new_vertex[static_cast<size_t>(i)]);
    }
    out.F.reserve(state.F.size());
    for (const Eigen::Vector3i &f : state.F) {
        out.F.push_back(Eigen::Vector3i(
            remap[static_cast<size_t>(f[0])],
            remap[static_cast<size_t>(f[1])],
            remap[static_cast<size_t>(f[2])]));
    }
    return out;
}

bool collapse_topology_ok(const MeshState &state, int a, int b) {
    const std::vector<int> simplex_val = vertex_simplex_valences(state);
    if (simplex_val[static_cast<size_t>(a)] <= 3 ||
        simplex_val[static_cast<size_t>(b)] <= 3) {
        return false;
    }

    const std::vector<int> edge_faces = edge_parent_faces(state, a, b);
    if (edge_faces.size() != 2) {
        return false;
    }

    const int expected_valence =
        simplex_val[static_cast<size_t>(a)] +
        simplex_val[static_cast<size_t>(b)] -
        static_cast<int>(edge_faces.size()) - 2;
    if (expected_valence > kMaxSimplexValence) {
        return false;
    }

    const std::set<int> opposite =
        edge_opposite_vertices(state, edge_faces, a, b);
    for (int v : opposite) {
        if (simplex_val[static_cast<size_t>(v)] <= 3) {
            return false;
        }
    }

    std::set<int> common;
    const std::set<int> nbr_a = vertex_neighbors(state, a);
    const std::set<int> nbr_b = vertex_neighbors(state, b);
    std::set_intersection(nbr_a.begin(), nbr_a.end(),
                          nbr_b.begin(), nbr_b.end(),
                          std::inserter(common, common.begin()));
    return common == opposite;
}

bool collapse_candidate(const MeshState &state,
                        int a,
                        int b,
                        MeshState &out) {
    if (a < 0 || b < 0 || a >= static_cast<int>(state.V.size()) ||
        b >= static_cast<int>(state.V.size()) || a == b) {
        return false;
    }
    if (!collapse_topology_ok(state, a, b)) {
        return false;
    }

    std::vector<Eigen::Vector3d> candV = state.V;
    candV[static_cast<size_t>(a)] =
        0.5 * (state.V[static_cast<size_t>(a)] + state.V[static_cast<size_t>(b)]);

    std::vector<Eigen::Vector3i> candF;
    std::vector<int> old_face_index;
    candF.reserve(state.F.size());
    old_face_index.reserve(state.F.size());

    for (int fi = 0; fi < static_cast<int>(state.F.size()); ++fi) {
        Eigen::Vector3i f = state.F[static_cast<size_t>(fi)];
        for (int k = 0; k < 3; ++k) {
            if (f[k] == b) f[k] = a;
        }
        if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) {
            continue;
        }
        candF.push_back(f);
        old_face_index.push_back(fi);
    }

    // Foldover check on the surviving faces touched by the collapse.
    for (int ci = 0; ci < static_cast<int>(candF.size()); ++ci) {
        const int oi = old_face_index[static_cast<size_t>(ci)];
        const Eigen::Vector3i oldf = state.F[static_cast<size_t>(oi)];
        if (!contains_vertex(oldf, a) && !contains_vertex(oldf, b)) {
            continue;
        }
        const Eigen::Vector3d n0 = face_normal_raw(state.V, oldf);
        const Eigen::Vector3d n1 =
            face_normal_raw(candV, candF[static_cast<size_t>(ci)]);
        if (n0.norm() <= 1e-14 || n1.norm() <= 1e-14) {
            return false;
        }
        if (n0.dot(n1) <= 0.0) {
            return false;
        }
    }

    MeshState candidate;
    candidate.V = std::move(candV);
    candidate.F = std::move(candF);
    candidate.is_new_vertex = state.is_new_vertex;
    if (!manifold_faces_ok(candidate)) {
        return false;
    }

    std::set<int> affected;
    affected.insert(a);
    for (const Eigen::Vector3i &f : state.F) {
        if (!contains_vertex(f, a) && !contains_vertex(f, b)) continue;
        for (int k = 0; k < 3; ++k) {
            affected.insert(f[k] == b ? a : f[k]);
        }
    }

    const std::vector<int> val = vertex_valences(candidate);
    for (int v : affected) {
        if (v < 0 || v >= static_cast<int>(val.size()) || v == b) continue;
        const int vv = val[static_cast<size_t>(v)];
        // Complete removal of a tiny isolated component is allowed by the
        // collapse primitive tests; production closed surfaces keep valence > 0.
        if (vv == 0) continue;
        if (vv < 4 || vv > kMaxSimplexValence) {
            return false;
        }
    }

    out = compact_state(candidate, a);
    return true;
}

MeshState from_meshdata(const MeshData &mesh, int original_vertex_count) {
    MeshState out;
    out.V.reserve(static_cast<size_t>(mesh.n_vertices()));
    out.is_new_vertex.reserve(static_cast<size_t>(mesh.n_vertices()));
    for (int i = 0; i < mesh.n_vertices(); ++i) {
        out.V.push_back(mesh.V.row(i).transpose());
        out.is_new_vertex.push_back(i >= original_vertex_count);
    }
    out.F.reserve(static_cast<size_t>(mesh.n_faces()));
    for (int f = 0; f < mesh.n_faces(); ++f) {
        out.F.push_back(mesh.F.row(f).transpose());
    }
    return out;
}

MeshData to_meshdata(const MeshState &state, double target_l0, bool compute_normals) {
    MeshData out;
    out.V.resize(static_cast<int>(state.V.size()), 3);
    for (int i = 0; i < static_cast<int>(state.V.size()); ++i) {
        out.V.row(i) = state.V[static_cast<size_t>(i)].transpose();
    }
    out.F.resize(static_cast<int>(state.F.size()), 3);
    for (int f = 0; f < static_cast<int>(state.F.size()); ++f) {
        out.F.row(f) = state.F[static_cast<size_t>(f)].transpose();
    }
    out.L0 = target_l0;

    if (compute_normals) {
        out.N = Eigen::MatrixXd::Zero(out.n_vertices(), 3);
        for (int f = 0; f < out.n_faces(); ++f) {
            const Eigen::Vector3d n = face_normal_raw(state.V, state.F[static_cast<size_t>(f)]);
            out.N.row(out.F(f, 0)) += n.transpose();
            out.N.row(out.F(f, 1)) += n.transpose();
            out.N.row(out.F(f, 2)) += n.transpose();
        }
        for (int i = 0; i < out.n_vertices(); ++i) {
            const double nn = out.N.row(i).norm();
            if (nn > 0.0) out.N.row(i) /= nn;
        }
    }
    return out;
}

void split_long_edges(OMesh &om, double split_threshold, int max_faces) {
    constexpr int kMaxSplitPasses = 8;
    for (int pass = 0; pass < kMaxSplitPasses; ++pass) {
        if (max_faces > 0 &&
            static_cast<int>(om.n_faces()) >= max_faces) {
            return;
        }

        struct SplitEdge {
            int a;
            int b;
            double length;
        };
        std::vector<SplitEdge> edges;
        for (auto e_it = om.edges_begin(); e_it != om.edges_end(); ++e_it) {
            const auto heh = om.halfedge_handle(*e_it, 0);
            const auto vh0 = om.from_vertex_handle(heh);
            const auto vh1 = om.to_vertex_handle(heh);
            const auto p0 = om.point(vh0);
            const auto p1 = om.point(vh1);
            const Eigen::Vector3d d(p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]);
            if (d.norm() > split_threshold) {
                edges.push_back({vh0.idx(), vh1.idx(), d.norm()});
            }
        }
        if (edges.empty()) return;
        std::sort(edges.begin(), edges.end(),
                  [](const SplitEdge &x, const SplitEdge &y) {
                      return x.length > y.length;
                  });

        bool split_any = false;
        std::vector<char> blocked(static_cast<size_t>(om.n_vertices()), 0);
        for (const SplitEdge &e : edges) {
            if (e.a >= static_cast<int>(om.n_vertices()) ||
                e.b >= static_cast<int>(om.n_vertices())) {
                continue;
            }
            if (blocked[static_cast<size_t>(e.a)] ||
                blocked[static_cast<size_t>(e.b)]) {
                continue;
            }
            const auto va = OMesh::VertexHandle(e.a);
            const auto vb = OMesh::VertexHandle(e.b);
            auto heh = om.find_halfedge(va, vb);
            if (!heh.is_valid()) heh = om.find_halfedge(vb, va);
            if (!heh.is_valid()) continue;
            const auto eh = om.edge_handle(heh);
            if (!split_topology_ok(om, eh, blocked)) continue;
            if (max_faces > 0) {
                const int delta = split_face_delta(om, eh);
                if (delta <= 0 ||
                    static_cast<int>(om.n_faces()) + delta > max_faces) {
                    continue;
                }
            }

            const auto pa = om.point(va);
            const auto pb = om.point(vb);
            const Eigen::Vector3d d(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
            if (d.norm() <= split_threshold) continue;
            const OMesh::Point mid(
                0.5 * (pa[0] + pb[0]),
                0.5 * (pa[1] + pb[1]),
                0.5 * (pa[2] + pb[2]));
            const int old_vertex_count = static_cast<int>(om.n_vertices());
            om.split(eh, mid);
            blocked.resize(static_cast<size_t>(om.n_vertices()), 0);
            blocked[static_cast<size_t>(e.a)] = 1;
            blocked[static_cast<size_t>(e.b)] = 1;
            for (int v = old_vertex_count; v < static_cast<int>(om.n_vertices()); ++v) {
                blocked[static_cast<size_t>(v)] = 1;
            }
            split_any = true;
        }
        if (!split_any) return;
    }
}

void collapse_short_edges(MeshState &state, double collapse_threshold) {
    constexpr int kMaxCollapsePasses = 8;
    for (int pass = 0; pass < kMaxCollapsePasses; ++pass) {
        std::vector<Edge> edges = collect_edges(state);
        std::sort(edges.begin(), edges.end(), [](const Edge &x, const Edge &y) {
            return x.length < y.length;
        });

        bool changed = false;
        for (const Edge &e : edges) {
            if (e.length >= collapse_threshold) break;
            if (state.is_new_vertex[static_cast<size_t>(e.a)] ||
                state.is_new_vertex[static_cast<size_t>(e.b)]) {
                continue;
            }
            MeshState candidate;
            if (collapse_candidate(state, e.a, e.b, candidate)) {
                state = std::move(candidate);
                changed = true;
                break;
            }
        }
        if (!changed) return;
    }
}

double clamp_unit(double x) {
    return std::max(-1.0, std::min(1.0, x));
}

double angle_at(const Eigen::Vector3d &a,
                const Eigen::Vector3d &b,
                const Eigen::Vector3d &c) {
    const Eigen::Vector3d u = b - a;
    const Eigen::Vector3d v = c - a;
    const double denom = u.norm() * v.norm();
    if (!(denom > 0.0)) return 0.0;
    return std::acos(clamp_unit(u.dot(v) / denom));
}

bool is_non_delaunay_edge(const OMesh &om, OMesh::EdgeHandle eh) {
    if (!om.is_flip_ok(eh)) return false;

    const auto h0 = om.halfedge_handle(eh, 0);
    const auto h1 = om.halfedge_handle(eh, 1);
    const auto f0 = om.face_handle(h0);
    const auto f1 = om.face_handle(h1);
    if (!f0.is_valid() || !f1.is_valid()) return false;

    const auto va = om.from_vertex_handle(h0);
    const auto vb = om.to_vertex_handle(h0);
    const auto vc = om.opposite_vh(h0);
    const auto vd = om.opposite_vh(h1);
    if (!va.is_valid() || !vb.is_valid() || !vc.is_valid() || !vd.is_valid()) {
        return false;
    }

    auto point = [&om](OMesh::VertexHandle vh) {
        const auto p = om.point(vh);
        return Eigen::Vector3d(p[0], p[1], p[2]);
    };

    const double a0 = angle_at(point(vc), point(va), point(vb));
    const double a1 = angle_at(point(vd), point(va), point(vb));
    return a0 + a1 > M_PI + 1e-12;
}

std::vector<std::vector<int>> incident_faces(const MeshData &mesh) {
    std::vector<std::vector<int>> inc(static_cast<size_t>(mesh.n_vertices()));
    for (int f = 0; f < mesh.n_faces(); ++f) {
        inc[static_cast<size_t>(mesh.F(f, 0))].push_back(f);
        inc[static_cast<size_t>(mesh.F(f, 1))].push_back(f);
        inc[static_cast<size_t>(mesh.F(f, 2))].push_back(f);
    }
    return inc;
}

std::vector<char> boundary_vertices(const MeshData &mesh) {
    std::map<std::pair<int, int>, int> edge_count;
    for (int f = 0; f < mesh.n_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            edge_count[ordered_edge(mesh.F(f, k), mesh.F(f, (k + 1) % 3))]++;
        }
    }

    std::vector<char> boundary(static_cast<size_t>(mesh.n_vertices()), 0);
    for (const auto &item : edge_count) {
        if (item.second == 1) {
            boundary[static_cast<size_t>(item.first.first)] = 1;
            boundary[static_cast<size_t>(item.first.second)] = 1;
        }
    }
    return boundary;
}

} // namespace

MeshData remesh_split_collapse(const MeshData &mesh, int max_faces) {
    const double target_l0 = mesh.L0 > 0.0 ? mesh.L0 : mesh.compute_L0();
    if (!(target_l0 > 0.0) || mesh.n_faces() == 0) {
        MeshData out = mesh;
        out.L0 = target_l0;
        return out;
    }

    const double split_threshold = (4.0 / 3.0) * target_l0;
    const double collapse_threshold = (4.0 / 5.0) * target_l0;
    const int original_vertex_count = mesh.n_vertices();
    const bool had_normals = mesh.N.rows() == mesh.V.rows();

    MeshState state = from_meshdata(mesh, original_vertex_count);
    collapse_short_edges(state, collapse_threshold);

    MeshData after_collapse = to_meshdata(state, target_l0, had_normals);
    OMesh om = after_collapse.to_openmesh();
    split_long_edges(om, split_threshold, max_faces);

    return finalize_meshdata(MeshData::from_openmesh(om), target_l0, had_normals);
}

MeshData remesh_delaunay_flip(const MeshData &mesh, int max_passes) {
    const double target_l0 = mesh.L0 > 0.0 ? mesh.L0 : mesh.compute_L0();
    const bool had_normals = mesh.N.rows() == mesh.V.rows();
    OMesh om = mesh.to_openmesh();

    const int capped_passes = std::max(0, max_passes);
    for (int pass = 0; pass < capped_passes; ++pass) {
        std::vector<int> edge_ids;
        edge_ids.reserve(om.n_edges());
        for (auto e_it = om.edges_begin(); e_it != om.edges_end(); ++e_it) {
            edge_ids.push_back(e_it->idx());
        }

        bool flipped_any = false;
        for (int edge_id : edge_ids) {
            if (edge_id < 0 || edge_id >= static_cast<int>(om.n_edges())) continue;
            const auto eh = OMesh::EdgeHandle(edge_id);
            if (!is_non_delaunay_edge(om, eh)) continue;
            om.flip(eh);
            flipped_any = true;
        }
        if (!flipped_any) break;
    }

    return finalize_meshdata(MeshData::from_openmesh(om), target_l0, had_normals);
}

MeshData remesh_tangential_smooth(const MeshData &mesh,
                                  double rho,
                                  int n_iters) {
    const double target_l0 = mesh.L0 > 0.0 ? mesh.L0 : mesh.compute_L0();
    const bool had_normals = mesh.N.rows() == mesh.V.rows();
    MeshData out = mesh;
    out.L0 = target_l0;
    if (out.n_faces() == 0 || out.n_vertices() == 0 || n_iters <= 0 ||
        rho == 0.0) {
        return finalize_meshdata(out, target_l0, had_normals);
    }

    const std::vector<std::vector<int>> inc = incident_faces(out);
    const std::vector<char> boundary = boundary_vertices(out);
    for (int iter = 0; iter < n_iters; ++iter) {
        Eigen::MatrixXd next = out.V;
        for (int v = 0; v < out.n_vertices(); ++v) {
            if (boundary[static_cast<size_t>(v)] ||
                inc[static_cast<size_t>(v)].empty()) {
                continue;
            }

            Eigen::Vector3d mean_center = Eigen::Vector3d::Zero();
            Eigen::Matrix3d tangent_projector = Eigen::Matrix3d::Zero();
            double area_sum = 0.0;
            int center_count = 0;
            for (int f : inc[static_cast<size_t>(v)]) {
                const Eigen::Vector3d p0 = out.V.row(out.F(f, 0)).transpose();
                const Eigen::Vector3d p1 = out.V.row(out.F(f, 1)).transpose();
                const Eigen::Vector3d p2 = out.V.row(out.F(f, 2)).transpose();
                const Eigen::Vector3d n_raw = (p1 - p0).cross(p2 - p0);
                const double area = 0.5 * n_raw.norm();
                if (!(area > 0.0)) continue;
                const Eigen::Vector3d n = n_raw.normalized();
                mean_center += (p0 + p1 + p2) / 3.0;
                tangent_projector +=
                    area * (Eigen::Matrix3d::Identity() - n * n.transpose());
                area_sum += area;
                ++center_count;
            }
            if (!(area_sum > 0.0) || center_count <= 0) continue;

            mean_center /= static_cast<double>(center_count);
            tangent_projector /= area_sum;
            const Eigen::Vector3d delta =
                (3.0 * rho) *
                (tangent_projector *
                 (mean_center - out.V.row(v).transpose()));
            next.row(v) += delta.transpose();
        }
        out.V = std::move(next);
    }

    return finalize_meshdata(out, target_l0, had_normals);
}

MeshData remesh_full(const MeshData &mesh, int max_faces) {
    MeshData out = remesh_split_collapse(mesh, max_faces);
    out = remesh_delaunay_flip(out);
    out = remesh_tangential_smooth(out);
    out.L0 = mesh.L0 > 0.0 ? mesh.L0 : mesh.compute_L0();
    return out;
}

} // namespace rsh
