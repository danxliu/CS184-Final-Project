#include "MeshData.h"

#include <iostream>
#include <set>
#include <utility>

namespace rsh {

MeshData MeshData::from_openmesh(const OMesh &mesh) {
    MeshData out;
    out.V.resize(static_cast<int>(mesh.n_vertices()), 3);
    out.F.resize(static_cast<int>(mesh.n_faces()), 3);

    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        const auto p = mesh.point(*v_it);
        const int i = v_it->idx();
        out.V(i, 0) = p[0];
        out.V(i, 1) = p[1];
        out.V(i, 2) = p[2];
    }

    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        const int fi = f_it->idx();
        int k = 0;
        for (auto fv_it = mesh.cfv_iter(*f_it); fv_it.is_valid() && k < 3; ++fv_it, ++k) {
            out.F(fi, k) = fv_it->idx();
        }
    }
    return out;
}

OMesh MeshData::to_openmesh() const {
    OMesh mesh;
    const bool have_normals = N.rows() == V.rows();
    if (have_normals) mesh.request_vertex_normals();

    std::vector<OMesh::VertexHandle> handles;
    handles.reserve(V.rows());
    for (int i = 0; i < V.rows(); ++i) {
        auto h = mesh.add_vertex(OMesh::Point(V(i, 0), V(i, 1), V(i, 2)));
        handles.push_back(h);
        if (have_normals) {
            mesh.set_normal(h, OMesh::Normal(N(i, 0), N(i, 1), N(i, 2)));
        }
    }
    std::vector<OMesh::VertexHandle> tri(3);
    for (int i = 0; i < F.rows(); ++i) {
        tri[0] = handles[F(i, 0)];
        tri[1] = handles[F(i, 1)];
        tri[2] = handles[F(i, 2)];
        mesh.add_face(tri);
    }
    return mesh;
}

MeshData MeshData::load_obj(const std::string &filename) {
    OMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, filename)) {
        std::cerr << "MeshData::load_obj failed on " << filename << std::endl;
        std::exit(-1);
    }
    return from_openmesh(mesh);
}

void MeshData::save_obj(const std::string &filename) const {
    OMesh mesh = to_openmesh();
    OpenMesh::IO::Options opts;
    if (N.rows() == V.rows()) {
        opts += OpenMesh::IO::Options::VertexNormal;
    }
    if (!OpenMesh::IO::write_mesh(mesh, filename, opts)) {
        std::cerr << "MeshData::save_obj failed on " << filename << std::endl;
    }
}

Eigen::Vector3d MeshData::centroid() const {
    return V.colwise().mean().transpose();
}

double MeshData::bbox_diagonal() const {
    const Eigen::Vector3d mn = V.colwise().minCoeff();
    const Eigen::Vector3d mx = V.colwise().maxCoeff();
    return (mx - mn).norm();
}

double MeshData::compute_L0() const {
    std::set<std::pair<int, int>> edges;
    for (int f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            int a = F(f, k);
            int b = F(f, (k + 1) % 3);
            if (a > b) std::swap(a, b);
            edges.emplace(a, b);
        }
    }
    double sum = 0.0;
    for (const auto &e : edges) {
        sum += (V.row(e.first) - V.row(e.second)).norm();
    }
    return edges.empty() ? 0.0 : sum / static_cast<double>(edges.size());
}

void MeshData::normalize() {
    const Eigen::RowVector3d c = V.colwise().mean();
    V.rowwise() -= c;
    const double diag = bbox_diagonal();
    if (diag > 0.0) V /= diag;
    L0 = compute_L0();
}

} // namespace rsh
