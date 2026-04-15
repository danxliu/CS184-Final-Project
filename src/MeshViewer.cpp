#include "MeshViewer.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <nanogui/opengl.h>
#include <nanogui/renderpass.h>
#include <nanogui/screen.h>
#include <nanogui/shader.h>
#include <nanogui/vector.h>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace nanogui;

std::string MeshViewer::read_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open shader file: " << filename << std::endl;
    std::cerr << "Make sure you are running the program from the project root "
                 "directory."
              << std::endl;
    exit(-1);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void MeshViewer::get_mesh_data(Mesh &mesh) {
  // Store mesh data into m_vertices, m_normals, and m_indices
  if (mesh.n_vertices() == 0 || mesh.n_faces() == 0) {
    std::cerr << "Loaded mesh is empty" << std::endl;
    exit(-1);
  }

  m_vertices.clear();
  m_normals.clear();
  m_indices.clear();

  mesh.request_vertex_normals();
  mesh.update_normals();

  for (Mesh::VertexIter v_it = mesh.vertices_begin();
       v_it != mesh.vertices_end(); v_it++) {
    Mesh::Point p = mesh.point(*v_it);
    m_vertices.push_back(p[0]);
    m_vertices.push_back(p[1]);
    m_vertices.push_back(p[2]);
    Mesh::Normal n = mesh.normal(*v_it);
    m_normals.push_back(n[0]);
    m_normals.push_back(n[1]);
    m_normals.push_back(n[2]);
  }

  for (Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
       f_it++) {
    for (Mesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid();
         fv_it++) {
      m_indices.push_back(fv_it->idx());
    }
  }
}

MeshViewer::MeshViewer(const std::string &filename)
    : Screen(Vector2i(800, 600), "Renderer", true, false, false, true, false) {
  Mesh mesh;
  if (!OpenMesh::IO::read_mesh(mesh, filename)) {
    std::cerr << "Could not load mesh: " << filename << std::endl;
    exit(-1);
  }

  get_mesh_data(mesh);
  m_index_count = m_indices.size();
  m_render_pass = new RenderPass({ this });
  m_render_pass->set_clear_color(0, Color(0.2f, 0.25f, 0.3f, 1.0f));
  m_render_pass->set_clear_depth(1.0f);

  m_shader =
      new Shader(m_render_pass, "mesh_viewer", read_file("src/shader.vert"),
                 read_file("src/shader.frag"));
  m_shader->set_buffer("position", VariableType::Float32,
                       {m_vertices.size() / 3, 3}, m_vertices.data());

  m_shader->set_buffer("normal", VariableType::Float32,
                       {m_normals.size() / 3, 3}, m_normals.data());

  m_shader->set_buffer("indices", VariableType::UInt32,
                       {m_indices.size()}, m_indices.data());
}

void MeshViewer::draw_contents() {
  float fov = 45.0f * (M_PI / 180.0f);
  float aspect = (float)m_size.x() / (float)m_size.y();
  Matrix4f proj = Matrix4f::perspective(fov, 0.1f, 100.0f, aspect);
  Matrix4f view = Matrix4f::look_at(Vector3f(0.0f, 0.0f, 5.0f), Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f));
  Matrix4f model = Matrix4f::scale(Vector3f(0.1f));
  Matrix4f mvp = proj * view * model;

  m_shader->set_uniform("mvp", mvp);
  m_shader->set_uniform("model", model);

  m_render_pass->resize(framebuffer_size());
  m_render_pass->begin();
  m_render_pass->set_cull_mode(RenderPass::CullMode::Disabled);
  m_render_pass->set_depth_test(RenderPass::DepthTest::Less, true);
  m_shader->begin();
  m_shader->draw_array(Shader::PrimitiveType::Triangle, 0, m_index_count, true);
  m_shader->end();

  m_render_pass->end();
}
