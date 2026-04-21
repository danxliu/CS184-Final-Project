#include "MeshViewer.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nanogui/button.h>
#include <nanogui/label.h>
#include <nanogui/layout.h>
#include <nanogui/opengl.h>
#include <nanogui/renderpass.h>
#include <nanogui/screen.h>
#include <nanogui/shader.h>
#include <nanogui/slider.h>
#include <nanogui/vector.h>
#include <nanogui/widget.h>
#include <nanogui/window.h>
#include <sstream>
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

void MeshViewer::read_mesh(const std::string &filename) {
  Mesh mesh;
  mesh.request_vertex_normals();
  OpenMesh::IO::Options read_options;
  read_options += OpenMesh::IO::Options::VertexNormal;
  if (!OpenMesh::IO::read_mesh(mesh, filename, read_options)) {
    std::cerr << "Could not load mesh: " << filename << std::endl;
    exit(-1);
  }

  if (mesh.n_vertices() == 0 || mesh.n_faces() == 0) {
    std::cerr << "Loaded mesh is empty" << std::endl;
    exit(-1);
  }

  m_vertices.clear();
  m_normals.clear();
  m_indices.clear();

  bool has_imported_normals =
      read_options.check(OpenMesh::IO::Options::VertexNormal);
  if (!has_imported_normals) {
    // Area-weighted smooth vertex normals. OpenMesh's default update_normals
    // does an unweighted face-normal average, which biases toward tiny
    // triangles (Spot's ears/tail have much smaller faces than the body) and
    // looks wrong. We weight each face's contribution by 2*area via the raw
    // cross-product (cross = 2*area * unit_normal), accumulate per vertex,
    // and normalize.
    mesh.request_face_normals();
    std::vector<Mesh::Normal> vn(mesh.n_vertices(), Mesh::Normal(0, 0, 0));
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
      auto fv = mesh.cfv_iter(*f_it);
      const auto p0 = mesh.point(*fv); ++fv;
      const auto p1 = mesh.point(*fv); ++fv;
      const auto p2 = mesh.point(*fv);
      const auto weighted = (p1 - p0) % (p2 - p0);  // 2*area*unit_normal
      for (auto fv2 = mesh.cfv_iter(*f_it); fv2.is_valid(); ++fv2) {
        vn[fv2->idx()] += weighted;
      }
    }
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
      auto n = vn[v_it->idx()];
      const float len = n.norm();
      if (len > 0.f) n /= len;
      mesh.set_normal(*v_it, n);
    }
  }

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
  namespace fs = std::filesystem;
  const bool is_sequence = fs::is_directory(filename);
  if (is_sequence) {
    if (!load_frame_sequence(filename)) {
      std::cerr << "No frame_*.obj files found in " << filename << std::endl;
      std::exit(-1);
    }
    read_mesh(m_frames[0]);
  } else {
    read_mesh(filename);
  }
  m_index_count = m_indices.size();
  m_render_pass = new RenderPass({this}, this);
  m_render_pass->set_clear_color(0, m_background_color);
  m_render_pass->set_clear_depth(1.0f);

  m_shader =
      new Shader(m_render_pass, "mesh_viewer",
                 read_file("src/viewer/shader.vert"),
                 read_file("src/viewer/shader.frag"));
  upload_mesh_buffers();

  if (is_sequence) {
    build_playback_ui();
    perform_layout();
  }
}

void MeshViewer::upload_mesh_buffers() {
  m_shader->set_buffer("position", VariableType::Float32,
                       {m_vertices.size() / 3, 3}, m_vertices.data());
  m_shader->set_buffer("normal", VariableType::Float32,
                       {m_normals.size() / 3, 3}, m_normals.data());
  m_shader->set_buffer("indices", VariableType::UInt32, {m_indices.size()},
                       m_indices.data());
  m_index_count = m_indices.size();
}

bool MeshViewer::load_frame_sequence(const std::string &dir) {
  namespace fs = std::filesystem;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    if (name.rfind("frame_", 0) == 0 &&
        entry.path().extension() == ".obj") {
      m_frames.push_back(entry.path().string());
    }
  }
  std::sort(m_frames.begin(), m_frames.end());
  return !m_frames.empty();
}

void MeshViewer::load_frame(int idx) {
  if (m_frames.empty()) return;
  idx = std::clamp(idx, 0, static_cast<int>(m_frames.size()) - 1);
  if (idx != m_frame_idx || m_vertices.empty()) {
    m_frame_idx = idx;
    read_mesh(m_frames[idx]);
    upload_mesh_buffers();
  }
  if (m_frame_label) {
    std::ostringstream oss;
    oss << "frame " << (idx + 1) << " / " << m_frames.size();
    m_frame_label->set_caption(oss.str());
  }
  if (m_frame_slider) {
    const float t = m_frames.size() > 1
                        ? static_cast<float>(idx) /
                              static_cast<float>(m_frames.size() - 1)
                        : 0.0f;
    m_frame_slider->set_value(t);
  }
  redraw();
}

void MeshViewer::build_playback_ui() {
  Window *panel = new Window(this, "Phase 1.8 playback");
  panel->set_position(Vector2i(15, 15));
  panel->set_fixed_width(260);
  panel->set_layout(new GroupLayout(8, 6, 10, 8));

  m_frame_label = new Label(panel, "", "sans-bold");
  m_frame_slider = new Slider(panel);
  m_frame_slider->set_range({0.0f, 1.0f});
  m_frame_slider->set_callback([this](float t) {
    const int n = static_cast<int>(m_frames.size());
    if (n <= 0) return;
    const int idx =
        std::clamp(static_cast<int>(std::round(t * (n - 1))), 0, n - 1);
    load_frame(idx);
  });

  Widget *row = new Widget(panel);
  row->set_layout(
      new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 4));
  Button *first = new Button(row, "|<");
  first->set_callback([this]() { load_frame(0); });
  Button *prev = new Button(row, "<");
  prev->set_callback([this]() { load_frame(m_frame_idx - 1); });
  m_play_button = new Button(row, "Play");
  m_play_button->set_callback([this]() {
    m_playing = !m_playing;
    m_play_button->set_caption(m_playing ? "Pause" : "Play");
    if (m_playing) m_last_advance_time = glfwGetTime();
    redraw();
  });
  Button *next = new Button(row, ">");
  next->set_callback([this]() { load_frame(m_frame_idx + 1); });
  Button *last = new Button(row, ">|");
  last->set_callback(
      [this]() { load_frame(static_cast<int>(m_frames.size()) - 1); });

  Widget *fps_row = new Widget(panel);
  fps_row->set_layout(
      new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 6));
  new Label(fps_row, "fps");
  Slider *fps_slider = new Slider(fps_row);
  fps_slider->set_range({2.0f, 60.0f});
  fps_slider->set_value(m_fps);
  fps_slider->set_fixed_width(130);
  Label *fps_value = new Label(fps_row, "");
  auto set_fps_label = [fps_value](float v) {
    std::ostringstream oss;
    oss << static_cast<int>(std::round(v));
    fps_value->set_caption(oss.str());
  };
  set_fps_label(m_fps);
  fps_slider->set_callback([this, set_fps_label](float v) {
    m_fps = v;
    set_fps_label(v);
  });

  load_frame(0);
}

Vector3f MeshViewer::camera_forward() const {
  return normalize(m_camera_target - m_camera_eye);
}

Vector3f MeshViewer::camera_right() const {
  return normalize(cross(camera_forward(), m_world_up));
}

bool MeshViewer::handle_keyboard_motion(float delta_time) {
  Vector3f forward = camera_forward();
  Vector3f right = camera_right();
  Vector3f delta(0.0f, 0.0f, 0.0f);

  if (m_keys[GLFW_KEY_W]) {
    delta += forward;
  }
  if (m_keys[GLFW_KEY_S]) {
    delta -= forward;
  }
  if (m_keys[GLFW_KEY_D]) {
    delta += right;
  }
  if (m_keys[GLFW_KEY_A]) {
    delta -= right;
  }

  if (squared_norm(delta) > 0.0f) {
    delta = normalize(delta) * (m_move_speed * delta_time);
    m_camera_eye += delta;
    m_camera_target += delta;
    return true;
  }
  return false;
}

void MeshViewer::orbit_camera(const Vector2f &rel) {
  Vector3f offset = m_camera_eye - m_camera_target;
  float radius = norm(offset);
  if (radius <= std::numeric_limits<float>::epsilon()) {
    return;
  }

  float yaw = std::atan2(offset.x(), offset.z());
  float horizontal =
      std::sqrt(offset.x() * offset.x() + offset.z() * offset.z());
  float pitch = std::atan2(offset.y(), horizontal);

  yaw -= rel.x() * m_orbit_sensitivity;
  pitch += rel.y() * m_orbit_sensitivity;
  const float max_pitch = 1.55334f; // ~89 degrees
  pitch = clip(pitch, -max_pitch, max_pitch);

  Vector3f new_offset(radius * std::sin(yaw) * std::cos(pitch),
                      radius * std::sin(pitch),
                      radius * std::cos(yaw) * std::cos(pitch));
  m_camera_eye = m_camera_target + new_offset;
}

void MeshViewer::pan_camera(const Vector2f &rel) {
  float distance = norm(m_camera_target - m_camera_eye);
  Vector3f right = camera_right();
  Vector3f up = normalize(cross(right, camera_forward()));
  Vector3f delta =
      (-rel.x() * right + rel.y() * up) * m_pan_sensitivity * distance * 0.002f;

  m_camera_eye += delta;
  m_camera_target += delta;
}

void MeshViewer::zoom_camera(float scroll_delta) {
  Vector3f offset = m_camera_eye - m_camera_target;
  float current_distance = norm(offset);
  if (current_distance <= std::numeric_limits<float>::epsilon()) {
    return;
  }

  float zoom_scale = 1.0f - scroll_delta * m_zoom_sensitivity;
  if (zoom_scale <= 0.05f) {
    zoom_scale = 0.05f;
  }
  float new_distance = clip(current_distance * zoom_scale,
                            m_min_camera_distance, m_max_camera_distance);
  m_camera_eye = m_camera_target + normalize(offset) * new_distance;
}

bool MeshViewer::keyboard_event(int key, int scancode, int action,
                                int modifiers) {
  if (Screen::keyboard_event(key, scancode, action, modifiers)) {
    return true;
  }

  if (key >= 0 && key < static_cast<int>(m_keys.size())) {
    if (action == GLFW_PRESS) {
      m_keys[static_cast<size_t>(key)] = true;
    } else if (action == GLFW_RELEASE) {
      m_keys[static_cast<size_t>(key)] = false;
    }
  }

  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    set_visible(false);
    return true;
  }

  if (!m_frames.empty() && action == GLFW_PRESS) {
    if (key == GLFW_KEY_SPACE) {
      m_playing = !m_playing;
      if (m_play_button)
        m_play_button->set_caption(m_playing ? "Pause" : "Play");
      if (m_playing) m_last_advance_time = glfwGetTime();
      redraw();
      return true;
    }
    if (key == GLFW_KEY_RIGHT) {
      load_frame(m_frame_idx + 1);
      return true;
    }
    if (key == GLFW_KEY_LEFT) {
      load_frame(m_frame_idx - 1);
      return true;
    }
    if (key == GLFW_KEY_HOME) {
      load_frame(0);
      return true;
    }
    if (key == GLFW_KEY_END) {
      load_frame(static_cast<int>(m_frames.size()) - 1);
      return true;
    }
  }

  return false;
}

bool MeshViewer::mouse_button_event(const Vector2i &p, int button, bool down,
                                    int modifiers) {
  if (Screen::mouse_button_event(p, button, down, modifiers)) {
    return true;
  }

  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    m_left_mouse_down = down;
    if (down) {
      m_camera_target = Vector3f(0.0f, 0.0f, 0.0f);
    }
    return true;
  }
  if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    m_right_mouse_down = down;
    return true;
  }

  return false;
}

bool MeshViewer::mouse_motion_event_f(const Vector2f &p, const Vector2f &rel,
                                      int button, int modifiers) {
  if (Screen::mouse_motion_event_f(p, rel, button, modifiers)) {
    return true;
  }

  if (m_left_mouse_down) {
    orbit_camera(rel);
    return true;
  }
  if (m_right_mouse_down) {
    pan_camera(rel);
    return true;
  }

  return false;
}

bool MeshViewer::scroll_event(const Vector2i &p, const Vector2f &rel) {
  if (Screen::scroll_event(p, rel)) {
    return true;
  }
  zoom_camera(rel.y());
  return true;
}

void MeshViewer::draw_contents() {
  double now = glfwGetTime();
  if (m_last_frame_time < 0.0) {
    m_last_frame_time = now;
  }
  float delta_time = static_cast<float>(now - m_last_frame_time);
  m_last_frame_time = now;
  delta_time = clip(delta_time, 0.0f, 0.1f);

  bool moved = handle_keyboard_motion(delta_time);
  if (moved && run_mode() == RunMode::Lazy) {
    redraw();
  }

  if (m_playing && !m_frames.empty() && m_fps > 0.0f) {
    const double interval = 1.0 / static_cast<double>(m_fps);
    if (m_last_advance_time < 0.0) m_last_advance_time = now;
    if (now - m_last_advance_time >= interval) {
      m_last_advance_time = now;
      int next = m_frame_idx + 1;
      if (next >= static_cast<int>(m_frames.size())) next = 0;
      load_frame(next);
    } else {
      redraw();
    }
  }

  float fov = m_fov * (M_PI / 180.0f);
  float aspect = (float)m_size.x() / (float)m_size.y();
  Matrix4f proj = Matrix4f::perspective(fov, 0.1f, 100.0f, aspect);
  Matrix4f view = Matrix4f::look_at(m_camera_eye, m_camera_target, m_world_up);
  Matrix4f model = Matrix4f::scale(Vector3f(0.1f));
  Matrix4f mvp = proj * view * model;

  m_shader->set_uniform("mvp", mvp);
  m_shader->set_uniform("model", model);

  // Passed into shaders
  m_shader->set_uniform("lightPos", m_light_pos);
  m_shader->set_uniform("viewPos", m_camera_eye);
  Vector3f obj_rgb(m_object_color.r(), m_object_color.g(), m_object_color.b());
  m_shader->set_uniform("objectColor", obj_rgb);

  m_render_pass->resize(framebuffer_size());
  m_render_pass->begin();
  m_render_pass->set_cull_mode(RenderPass::CullMode::Back);
  m_render_pass->set_depth_test(RenderPass::DepthTest::Less, true);
  m_shader->begin();
  m_shader->draw_array(Shader::PrimitiveType::Triangle, 0, m_index_count, true);
  m_shader->end();

  m_render_pass->end();
}
