#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../vendor/blackrubby.hpp"

namespace py = pybind11;
using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class PyBlackRubbyRegressor {
public:
    PyBlackRubbyRegressor(int expansion_dim = 1024,
                          double alpha = 0.0,
                          double bandwidth = 0.0,
                          std::uint64_t random_state = 42,
                          bool use_symbolic = true,
                          int sym_population = 128,
                          int sym_generations = 100,
                          int sym_rows = 1024,
                          int sym_top_k = 6,
                          double sym_time_limit = 0.0,
                          int select_rows = 2048,
                          int select_dim = 128,
                          int batch_size = 2048,
                          int solve_rows = -1,
                          double max_frequency = 54.5981500331)
        : expansion_dim_(expansion_dim), alpha_(alpha), bandwidth_(bandwidth),
          random_state_(random_state), use_symbolic_(use_symbolic),
          sym_population_(sym_population), sym_generations_(sym_generations),
          sym_rows_(sym_rows), sym_top_k_(sym_top_k), sym_time_limit_(sym_time_limit),
          select_rows_(select_rows), select_dim_(select_dim), batch_size_(batch_size), solve_rows_(solve_rows),
          max_frequency_(max_frequency) {
        reset_model();
    }

    PyBlackRubbyRegressor& fit(const py::array& x, const py::array& y) {
        auto [inputs, rows, cols] = as_matrix(x, "X");
        auto targets = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y);
        if (!targets || targets.ndim() > 2) throw std::invalid_argument("y must be a numeric vector.");
        if (targets.ndim() == 2 && targets.shape(1) != 1) {
            throw std::invalid_argument("multi-output y is not supported; pass one target column.");
        }
        if (targets.size() != rows) throw std::invalid_argument("y must contain one target per X row.");
        const double* target_data = targets.data();
        Eigen::Map<const Eigen::VectorXd> mapped_y(target_data, rows);
        if (!mapped_y.allFinite()) throw std::invalid_argument("y must contain only finite values.");

        reset_model();
        const auto start = std::chrono::steady_clock::now();
        {
            py::gil_scoped_release release;
            model_->fit(inputs, mapped_y);
        }
        fit_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        n_features_in_ = cols;
        fitted_ = true;
        return *this;
    }

    py::array_t<double> predict(const py::array& x) const {
        if (!fitted_) throw std::logic_error("fit must be called before predict.");
        auto [inputs, rows, cols] = as_matrix(x, "X");
        if (cols != n_features_in_) throw std::invalid_argument("X has a different number of features than fit data.");
        Eigen::VectorXd predictions;
        {
            py::gil_scoped_release release;
            predictions = model_->predict(inputs);
        }
        py::array_t<double> output(rows);
        std::copy(predictions.data(), predictions.data() + rows, output.mutable_data());
        return output;
    }

    double score(const py::array& x, const py::array& y) const {
        auto targets = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y);
        if (!targets || targets.ndim() > 2 || (targets.ndim() == 2 && targets.shape(1) != 1)) {
            throw std::invalid_argument("y must be a numeric vector or one-column matrix.");
        }
        py::array_t<double> predictions = predict(x);
        if (targets.size() != predictions.size()) throw std::invalid_argument("y must contain one target per X row.");
        Eigen::Map<const Eigen::VectorXd> truth(targets.data(), targets.size());
        Eigen::Map<const Eigen::VectorXd> predicted(predictions.data(), predictions.size());
        if (!truth.allFinite()) throw std::invalid_argument("y must contain only finite values.");
        const double total = (truth.array() - truth.mean()).square().sum();
        if (total <= std::numeric_limits<double>::epsilon()) {
            return (predicted - truth).squaredNorm() <= std::numeric_limits<double>::epsilon() ? 1.0 : 0.0;
        }
        return 1.0 - (predicted - truth).squaredNorm() / total;
    }

    py::dict get_params() const {
        py::dict params;
        params["expansion_dim"] = expansion_dim_;
        params["alpha"] = alpha_;
        params["bandwidth"] = bandwidth_;
        params["random_state"] = random_state_;
        params["use_symbolic"] = use_symbolic_;
        params["sym_population"] = sym_population_;
        params["sym_generations"] = sym_generations_;
        params["sym_rows"] = sym_rows_;
        params["sym_top_k"] = sym_top_k_;
        params["sym_time_limit"] = sym_time_limit_;
        params["select_rows"] = select_rows_;
        params["select_dim"] = select_dim_;
        params["batch_size"] = batch_size_;
        params["solve_rows"] = solve_rows_;
        params["max_frequency"] = max_frequency_;
        return params;
    }

    PyBlackRubbyRegressor& set_params(const py::kwargs& kwargs) {
        py::dict params = get_params();
        for (const auto& item : kwargs) {
            const std::string key = py::cast<std::string>(item.first);
            if (!params.contains(py::str(key))) throw std::invalid_argument("unknown parameter: " + key);
            params[py::str(key)] = item.second;
        }
        expansion_dim_ = py::cast<int>(params["expansion_dim"]);
        alpha_ = py::cast<double>(params["alpha"]);
        bandwidth_ = py::cast<double>(params["bandwidth"]);
        random_state_ = py::cast<std::uint64_t>(params["random_state"]);
        use_symbolic_ = py::cast<bool>(params["use_symbolic"]);
        sym_population_ = py::cast<int>(params["sym_population"]);
        sym_generations_ = py::cast<int>(params["sym_generations"]);
        sym_rows_ = py::cast<int>(params["sym_rows"]);
        sym_top_k_ = py::cast<int>(params["sym_top_k"]);
        sym_time_limit_ = py::cast<double>(params["sym_time_limit"]);
        select_rows_ = py::cast<int>(params["select_rows"]);
        select_dim_ = py::cast<int>(params["select_dim"]);
        batch_size_ = py::cast<int>(params["batch_size"]);
        solve_rows_ = py::cast<int>(params["solve_rows"]);
        max_frequency_ = py::cast<double>(params["max_frequency"]);
        reset_model();
        return *this;
    }

    bool is_fitted() const { return fitted_; }
    int n_features_in() const { return fitted_ ? n_features_in_ : 0; }
    double fit_seconds() const { return fit_seconds_; }
    double symbolic_seconds() const { return fitted_ ? model_->symbolic_seconds() : 0.0; }
    double chosen_bandwidth() const { return fitted_ ? model_->chosen_bandwidth() : 0.0; }
    double chosen_alpha() const { return fitted_ ? model_->chosen_alpha() : 0.0; }
    bool uses_symbolic() const { return fitted_ && model_->uses_symbolic(); }
    int symbolic_columns() const { return fitted_ ? model_->symbolic_columns() : 0; }
    std::string champion_formula() const { return fitted_ ? model_->champion_formula() : std::string(); }

private:
    int expansion_dim_;
    double alpha_;
    double bandwidth_;
    std::uint64_t random_state_;
    bool use_symbolic_;
    int sym_population_;
    int sym_generations_;
    int sym_rows_;
    int sym_top_k_;
    double sym_time_limit_;
    int select_rows_;
    int select_dim_;
    int batch_size_;
    int solve_rows_;
    double max_frequency_;
    int n_features_in_ = 0;
    double fit_seconds_ = 0.0;
    bool fitted_ = false;
    std::unique_ptr<blackrubby::BlackRubby> model_;

    static std::tuple<Eigen::MatrixXd, Eigen::Index, Eigen::Index> as_matrix(const py::array& value,
                                                                              const char* name) {
        auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
        if (!array || array.ndim() < 1 || array.ndim() > 2 || array.size() == 0) {
            throw std::invalid_argument(std::string(name) + " must be a non-empty 1D or 2D numeric array.");
        }
        const Eigen::Index rows = array.shape(0);
        const Eigen::Index cols = array.ndim() == 1 ? 1 : array.shape(1);
        if (rows == 0 || cols == 0) throw std::invalid_argument(std::string(name) + " must have no empty dimensions.");
        Eigen::Map<const RowMajorMatrix> mapped(array.data(), rows, cols);
        if (!mapped.allFinite()) throw std::invalid_argument(std::string(name) + " must contain only finite values.");
        return {Eigen::MatrixXd(mapped), rows, cols};
    }

    void reset_model() {
        blackrubby::Options options;
        options.expansion_dim = expansion_dim_;
        options.alpha = alpha_;
        options.bandwidth = bandwidth_;
        options.seed = random_state_;
        options.use_symbolic = use_symbolic_;
        options.sym_population = sym_population_;
        options.sym_generations = sym_generations_;
        options.sym_rows = sym_rows_;
        options.sym_top_k = sym_top_k_;
        options.sym_time_limit = sym_time_limit_;
        options.select_rows = select_rows_;
        options.select_dim = select_dim_;
        options.batch_size = batch_size_;
        options.solve_rows = solve_rows_;
        options.max_frequency = max_frequency_;
        model_ = std::make_unique<blackrubby::BlackRubby>(options);
        fitted_ = false;
        n_features_in_ = 0;
        fit_seconds_ = 0.0;
    }
};

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native BlackRubby symbolic-plus-kernel regression core.";
    py::class_<PyBlackRubbyRegressor>(module, "BlackRubbyRegressor")
        .def(py::init<int, double, double, std::uint64_t, bool, int, int, int, int, double, int, int, int, int, double>(),
             py::arg("expansion_dim") = 1024,
             py::arg("alpha") = 0.0,
             py::arg("bandwidth") = 0.0,
             py::arg("random_state") = 42,
             py::arg("use_symbolic") = true,
             py::arg("sym_population") = 128,
             py::arg("sym_generations") = 100,
             py::arg("sym_rows") = 1024,
             py::arg("sym_top_k") = 6,
             py::arg("sym_time_limit") = 0.0,
             py::arg("select_rows") = 2048,
             py::arg("select_dim") = 128,
             py::arg("batch_size") = 2048,
             py::arg("solve_rows") = -1,
             py::arg("max_frequency") = 54.5981500331)
        .def("fit", &PyBlackRubbyRegressor::fit, py::return_value_policy::reference_internal,
             py::arg("X"), py::arg("y"), "Fit one numeric target from a 1D/2D feature array.")
        .def("predict", &PyBlackRubbyRegressor::predict, py::arg("X"), "Predict targets for rows of numeric features.")
        .def("score", &PyBlackRubbyRegressor::score, py::arg("X"), py::arg("y"), "Return the coefficient of determination R^2.")
        .def("get_params", &PyBlackRubbyRegressor::get_params)
        .def("set_params", &PyBlackRubbyRegressor::set_params, py::return_value_policy::reference_internal)
        .def_property_readonly("is_fitted_", &PyBlackRubbyRegressor::is_fitted)
        .def_property_readonly("n_features_in_", &PyBlackRubbyRegressor::n_features_in)
        .def_property_readonly("fit_seconds_", &PyBlackRubbyRegressor::fit_seconds)
        .def_property_readonly("symbolic_seconds_", &PyBlackRubbyRegressor::symbolic_seconds)
        .def_property_readonly("chosen_bandwidth_", &PyBlackRubbyRegressor::chosen_bandwidth)
        .def_property_readonly("chosen_alpha_", &PyBlackRubbyRegressor::chosen_alpha)
        .def_property_readonly("uses_symbolic_", &PyBlackRubbyRegressor::uses_symbolic)
        .def_property_readonly("symbolic_columns_", &PyBlackRubbyRegressor::symbolic_columns)
        .def_property_readonly("champion_formula_", &PyBlackRubbyRegressor::champion_formula);
}
