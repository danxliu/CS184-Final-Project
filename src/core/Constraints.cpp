#include "Constraints.h"

#include <stdexcept>
#include <string>

namespace rsh {

namespace {

void validate_vertex_field(const Eigen::MatrixXd &d, const char *name) {
    if (d.cols() != 3) {
        throw std::runtime_error(std::string(name) + ": field must have three columns");
    }
}

} // namespace

void project_barycenter(Eigen::MatrixXd &d) {
    validate_vertex_field(d, "project_barycenter");
    if (d.rows() == 0) return;

    const Eigen::RowVector3d mean = d.colwise().mean();
    d.rowwise() -= mean;
}

void apply_pin_mask(Eigen::MatrixXd &d, const std::vector<bool> &pin) {
    validate_vertex_field(d, "apply_pin_mask");
    if (static_cast<int>(pin.size()) != d.rows()) {
        throw std::runtime_error("apply_pin_mask: pin mask size must match row count");
    }

    for (int i = 0; i < d.rows(); ++i) {
        if (pin[static_cast<size_t>(i)]) {
            d.row(i).setZero();
        }
    }
}

} // namespace rsh
