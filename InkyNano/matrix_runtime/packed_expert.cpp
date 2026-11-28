#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t hidden = 2304;
constexpr int64_t intermediate = 1152;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> reference_matvec(const std::vector<char> & bytes, size_t start,
                                    ggml_tensor * weight, const std::vector<float> & input) {
    const auto * traits = ggml_get_type_traits(weight->type);
    if (!traits->to_float) {
        throw std::runtime_error("reference dequantization is unavailable");
    }
    std::vector<float> row(input.size()), output(weight->ne[1]);
    const size_t stride = ggml_row_size(weight->type, weight->ne[0]);
    for (size_t i = 0; i < output.size(); ++i) {
        traits->to_float(bytes.data() + start + i * stride, row.data(), weight->ne[0]);
        double sum = 0.0;
        for (size_t j = 0; j < row.size(); ++j) {
            sum += static_cast<double>(row[j]) * input[j];
        }
        output[i] = static_cast<float>(sum);
    }
    return output;
}

ggml_tensor * cap(ggml_context * ctx, ggml_tensor * x, float limit) {
    return ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, x, 1.0f / limit)), limit);
}

int run(const std::string & backend_name, const std::string & profile, const char * path) {
    if ((backend_name != "cpu" && backend_name != "cuda")
        || (profile != "q4" && profile != "q6")) {
        throw std::runtime_error("expected cpu|cuda and q4|q6");
    }
    ggml_backend_load_all();
    const auto device = backend_name == "cpu"
        ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
        : ggml_backend_dev_by_name("CUDA0");
    if (!device) {
        throw std::runtime_error("requested backend is unavailable in this build");
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}), ggml_free);
    if (!backend || !ctx) {
        throw std::runtime_error("could not initialize the GGML backend or context");
    }

    const bool q6 = profile == "q6";
    const auto gate_type = q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
    const auto down_type = q6 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q5_0;
    std::array<ggml_tensor *, 3> weights{
        ggml_new_tensor_2d(ctx.get(), gate_type, hidden, intermediate),
        ggml_new_tensor_2d(ctx.get(), gate_type, hidden, intermediate),
        ggml_new_tensor_2d(ctx.get(), down_type, intermediate, hidden),
    };
    size_t weight_bytes = 0;
    std::array<size_t, 3> offsets{};
    for (size_t i = 0; i < weights.size(); ++i) {
        offsets[i] = weight_bytes;
        weight_bytes += ggml_nbytes(weights[i]);
    }
    std::vector<char> bytes(weight_bytes);
    std::ifstream input_file(path, std::ios::binary);
    if (!input_file.read(bytes.data(), bytes.size()) || input_file.peek() != EOF) {
        throw std::runtime_error("expert span does not match the exact selected layout");
    }

    std::vector<float> input(hidden), actual(hidden);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(std::sin(i * 0.051) + 0.17 * std::cos(i * 0.032));
    }
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, 1);
    auto * gate = ggml_mul_mat(ctx.get(), weights[0], x);
    auto * up = ggml_mul_mat(ctx.get(), weights[1], x);
    auto * activated = ggml_mul(ctx.get(), ggml_mul(ctx.get(), cap(ctx.get(), gate, 4.0f),
                                        ggml_sigmoid(ctx.get(), gate)), cap(ctx.get(), up, 25.0f));
    auto * output = ggml_mul_mat(ctx.get(), weights[2], activated);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    const auto start_upload = Clock::now();
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("could not allocate packed graph buffers");
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        ggml_backend_tensor_set(weights[i], bytes.data() + offsets[i], 0, ggml_nbytes(weights[i]));
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_synchronize(backend.get());
    const double upload_seconds = seconds(start_upload);
    const auto start_compute = Clock::now();
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("packed expert graph execution failed");
    }
    ggml_backend_synchronize(backend.get());
    const double compute_seconds = seconds(start_compute);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    auto ref_gate = reference_matvec(bytes, offsets[0], weights[0], input);
    const auto ref_up = reference_matvec(bytes, offsets[1], weights[1], input);
    for (size_t i = 0; i < ref_gate.size(); ++i) {
        const float g = ref_gate[i], u = ref_up[i];
        ref_gate[i] = 4.0f * std::tanh(g / 4.0f) / (1.0f + std::exp(-g))
                    * 25.0f * std::tanh(u / 25.0f);
    }
    const auto expected = reference_matvec(bytes, offsets[2], weights[2], ref_gate);
    double norm = 0.0, error = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("expert produced a nonfinite output");
        }
        const double delta = static_cast<double>(actual[i]) - expected[i];
        error += delta * delta;
        norm += static_cast<double>(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    if (norm == 0) {
        throw std::runtime_error("degenerate reference output");
    }
    std::cout << nlohmann::json{
        {"backend", ggml_backend_name(backend.get())}, {"profile", profile},
        {"activation", "deterministic synthetic input"},
        {"reference", "rowwise float dequantization of the same packed expert"},
        {"packed_weight_bytes", weight_bytes},
        {"backend_buffer_bytes", ggml_backend_buffer_get_size(buffer.get())},
        {"buffer_and_upload_seconds", upload_seconds}, {"compute_seconds", compute_seconds},
        {"relative_l2", std::sqrt(error / norm)}, {"max_abs_error", max_abs},
    }.dump() << '\n';
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 4) {
            throw std::runtime_error("usage: inky-packed-expert cpu|cuda q4|q6 EXPERT_SPAN");
        }
        return run(argv[1], argv[2], argv[3]);
    } catch (const std::exception & error) {
        std::cerr << "inky-packed-expert: " << error.what() << '\n';
        return 1;
    }
}
