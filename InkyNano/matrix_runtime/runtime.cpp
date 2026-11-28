#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

constexpr int n_layer = 24;
constexpr int n_expert = 128;
constexpr int n_slot = 48;
constexpr int n_active = 8;

double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

uint64_t peak_rss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS state{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &state, sizeof(state))) {
        throw std::runtime_error("cannot read peak process working set");
    }
    return state.PeakWorkingSetSize;
#else
    struct rusage state{};
    if (getrusage(RUSAGE_SELF, &state)) {
        throw std::runtime_error("cannot read peak process resident set");
    }
#ifdef __APPLE__
    return state.ru_maxrss;
#else
    return static_cast<uint64_t>(state.ru_maxrss) * 1024;
#endif
#endif
}

struct routing {
    struct by_phase {
        std::array<std::array<double, n_expert>, n_layer> mass{};
        std::array<int, n_layer> visits{};
        int tokens = 0;
    };
    std::array<by_phase, 2> result;
    int phase = 0;
    bool invalid = false;
    bool ordered = false;
    int token_position = -1;
    json events = json::array();
    std::array<std::vector<float>, n_slot> raw_scores, rank_scores;
    std::array<int, n_slot> event_index{};

    void begin_token(int position) {
        token_position = position;
        event_index.fill(-1);
        for (auto & v : raw_scores) v.clear();
        for (auto & v : rank_scores) v.clear();
    }
    void finish_token() const {
        for (int slot = 0; slot < n_slot; ++slot) {
            if (event_index[slot] < 0) throw std::runtime_error("missing ordered router visit");
            const auto & e = events.at(event_index[slot]);
            if (!e.contains("scaled_weights")) throw std::runtime_error("missing native scaled mixture weights");
        }
    }
    static std::vector<float> read_scores(ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] < n_active || tensor->ne[0] > n_expert ||
            tensor->ne[1] != 1 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("unexpected single-token router score shape");
        }
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        std::vector<float> out(tensor->ne[0]);
        for (size_t i = 0; i < out.size(); ++i) {
            std::memcpy(&out[i], bytes.data() + i * tensor->nb[0], sizeof(float));
            if (!std::isfinite(out[i])) throw std::runtime_error("nonfinite router score");
        }
        return out;
    }

    template<typename T> static void read(ggml_tensor * tensor, std::vector<T> & out) {
        const bool two_dimensional = tensor->ne[0] == n_active && tensor->ne[2] == 1;
        const bool three_dimensional = tensor->ne[0] == 1 && tensor->ne[1] == n_active;
        if ((!two_dimensional && !three_dimensional) || tensor->ne[3] != 1) {
            throw std::runtime_error(std::string("unexpected routing tensor shape for ") + ggml_get_name(tensor)
                + " [" + std::to_string(tensor->ne[0]) + "," + std::to_string(tensor->ne[1])
                + "," + std::to_string(tensor->ne[2]) + "]");
        }
        const int expert_axis = two_dimensional ? 0 : 1;
        const int token_axis = expert_axis + 1;
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        out.resize(n_active * tensor->ne[token_axis]);
        for (int64_t i = 0; i < tensor->ne[token_axis]; ++i) {
            for (int j = 0; j < n_active; ++j) {
                std::memcpy(out.data() + i * n_active + j,
                            bytes.data() + i * tensor->nb[token_axis] + j * tensor->nb[expert_axis],
                            sizeof(T));
            }
        }
    }

    bool capture(ggml_tensor * tensor, bool ask) {
        const std::string name = ggml_get_name(tensor);
        if (ordered) {
            // The pinned build_moe_ffn exposes the unbiased sigmoid scores,
            // the bias-adjusted ranking scores, and final normalized/scaled
            // mixture coefficients as distinct tensors. Read exact names,
            // never derived views. This mode deliberately uses batch=1.
            const std::array<std::string,3> names = {"ffn_moe_probs-", "ffn_moe_probs_biased-", "ffn_moe_weights_scaled-"};
            for (size_t kind=0; kind<names.size(); ++kind) {
                if (name.rfind(names[kind],0) != 0) continue;
                const auto tail = name.substr(names[kind].size());
                if (tail.empty() || tail.find_first_not_of("0123456789") != std::string::npos) continue;
                if (ask) return true;
                const int slot = std::stoi(tail);
                if (slot < 0 || slot >= n_slot) throw std::runtime_error("invalid ordered router slot");
                if (kind < 2) {
                    auto & target = kind == 0 ? raw_scores[slot] : rank_scores[slot];
                    if (!target.empty()) throw std::runtime_error("duplicate router score snapshot");
                    target = read_scores(tensor);
                } else {
                    if (event_index[slot] < 0 || tensor->type != GGML_TYPE_F32)
                        throw std::runtime_error("scaled mixture arrived before selected IDs");
                    std::vector<float> scaled;
                    read(tensor, scaled);
                    if (scaled.size() != n_active) throw std::runtime_error("invalid scaled mixture shape");
                    for (float w : scaled) if (!std::isfinite(w) || w < 0) throw std::runtime_error("invalid scaled mixture");
                    auto & event = events.at(event_index[slot]);
                    if (event.contains("scaled_weights")) throw std::runtime_error("duplicate scaled mixture");
                    event["scaled_weights"] = scaled;
                }
                return true;
            }
        }
        constexpr const char * weight_name = "ffn_moe_weights-";
        if (name.rfind(weight_name, 0) != 0) {
            return false;
        }
        const auto suffix = name.substr(std::strlen(weight_name));
        // GGML names derived views after their source, with a " (view)" suffix.
        if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos) {
            return false;
        }
        if (ask) {
            return true;
        }
        const int slot = std::stoi(suffix);
        if (slot < 0 || slot >= n_slot || tensor->type != GGML_TYPE_F32 || tensor->op != GGML_OP_GET_ROWS ||
            !tensor->src[1] || tensor->src[1]->type != GGML_TYPE_I32) {
            throw std::runtime_error("unexpected routing tensor identity");
        }
        // Read the exact gather indices while the graph still uses them for expert GEMMs.
        std::vector<int32_t> ids;
        std::vector<float> weights;
        read(tensor->src[1], ids);
        read(tensor, weights);
        if (ids.size() != weights.size()) {
            throw std::runtime_error("routing IDs and weights disagree at slot " + std::to_string(slot)
                + " (" + std::to_string(ids.size()) + ", " + std::to_string(weights.size()) + ")");
        }
        const int physical = slot < n_layer ? slot : 6 + (slot - n_layer) % 12;
        if (ordered) {
            if (ids.size() != n_active || event_index[slot] >= 0 || raw_scores[slot].empty() ||
                rank_scores[slot].size() != raw_scores[slot].size()) {
                throw std::runtime_error("incomplete or duplicate single-token routing snapshot");
            }
            const int pass = slot < n_layer ? 0 : 1 + (slot - n_layer) / 12;
            event_index[slot] = static_cast<int>(events.size());
            events.push_back({{"token",token_position},{"phase",phase==0?"prefill":"decode"},
                {"slot",slot},{"layer",physical},{"pass",pass},{"experts",ids},
                {"selected_scores",weights},{"scores",raw_scores[slot]},{"rank_scores",rank_scores[slot]}});
        }
        auto & stage = result.at(phase);
        for (size_t i = 0; i < ids.size(); i += n_active) {
            double total = 0;
            std::array<bool, n_expert> seen{};
            for (int j = 0; j < n_active; ++j) {
                const int expert = ids[i + j];
                const float weight = weights[i + j];
                if (expert < 0 || expert >= n_expert || seen[expert] ||
                    !std::isfinite(weight) || weight < 0) {
                    throw std::runtime_error("invalid native routing label");
                }
                seen[expert] = true;
                total += weight;
            }
            if (total <= 0) {
                throw std::runtime_error("no routing mass at a native visit");
            }
            for (int j = 0; j < n_active; ++j) {
                stage.mass[physical][ids[i + j]] += weights[i + j] / total;
            }
            ++stage.visits[physical];
        }
        return true;
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        auto * self = static_cast<routing *>(data);
        if (self->invalid) {
            return !ask;
        }
        try {
            return self->capture(tensor, ask);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "native routing callback: %s\n", error.what());
            self->invalid = true;
            return true;
        }
    }

    json receipt() const {
        json out;
        for (size_t i = 0; i < result.size(); ++i) {
            const auto & stage = result[i];
            for (int layer = 0; layer < n_layer; ++layer) {
                const int passes = layer >= 6 && layer < 18 ? 3 : 1;
                if (invalid || stage.tokens * passes != stage.visits[layer]) {
                    throw std::runtime_error("incomplete physical layer visit coverage");
                }
            }
            out[i == 0 ? "prefill" : "decode"] = {
                {"mass", stage.mass}, {"visits", stage.visits}, {"tokens", stage.tokens},
            };
        }
        return out;
    }
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt, bool parse_special) {
    int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, parse_special);
    if (count <= 0) {
        throw std::runtime_error("empty tokenized prompt");
    }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, parse_special) != count) {
        throw std::runtime_error("tokenizer count changed");
    }
    return tokens;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::string result(256, '\0');
    int count = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    if (count < 0) {
        result.resize(-count);
        count = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    }
    if (count < 0) {
        throw std::runtime_error("cannot decode token piece");
    }
    result.resize(count);
    return result;
}

size_t planned_cuda_bytes(const char * path, llama_model_params params,
                         llama_context_params context, ggml_backend_dev_t device) {
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, params), llama_model_free);
    if (!model) {
        throw std::runtime_error("CUDA memory plan could not load model metadata");
    }
    const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), context), llama_free);
    if (!ctx) {
        throw std::runtime_error("CUDA memory plan could not reserve a context");
    }
    size_t bytes = 0;
    for (const auto & [buffer, memory] : llama_get_memory_breakdown(ctx.get())) {
        if (!ggml_backend_buft_is_host(buffer) && ggml_backend_buft_get_device(buffer) == device) {
            bytes += memory.total();
        }
    }
    return bytes;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 10 && argc != 11 && argc != 13) {
            throw std::runtime_error("usage: inky-matrix MODEL PROMPT OUT MAX_NEW CTX THREADS trace|trace-ram|trace-events|trace-events-ram|mmap|ram|inspect MAX_RSS_BYTES PARSE_SPECIAL [BATCH_TOKENS [GPU_LAYERS MAX_VRAM_BYTES]]");
        }
        const int max_new = std::stoi(argv[4]);
        const int context = std::stoi(argv[5]);
        const int threads = std::stoi(argv[6]);
        const std::string mode = argv[7];
        const bool ordered_routing = mode == "trace-events" || mode == "trace-events-ram";
        const bool capture_routing = mode == "trace" || mode == "trace-ram" || ordered_routing;
        const uint64_t rss_limit = std::stoull(argv[8]);
        const std::string parse_special = argv[9];
        const int batch_tokens = argc >= 11 ? std::stoi(argv[10]) : 128;
        const int gpu_layers = argc == 13 ? std::stoi(argv[11]) : 0;
        const size_t vram_limit = argc == 13 ? std::stoull(argv[12]) : 0;
        ggml_backend_dev_t cuda_device = nullptr;
        size_t cuda_free_start = 0;
        size_t peak_device_delta = 0;
        const auto check_resources = [&]() {
            const uint64_t rss = peak_rss();
            if (rss > rss_limit) {
                throw std::runtime_error("native runtime crossed the process RAM limit at " + std::to_string(rss) + " bytes");
            }
            if (cuda_device) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(cuda_device, &free, &total);
                if (!total) {
                    throw std::runtime_error("CUDA device stopped reporting memory");
                }
                peak_device_delta = std::max(peak_device_delta, cuda_free_start - std::min(cuda_free_start, free));
                if (peak_device_delta > vram_limit) {
                    throw std::runtime_error("CUDA device free memory fell by more than the VRAM limit");
                }
            }
        };
        if (max_new < 2 || context < 128 || threads < 1 || batch_tokens < 1 || batch_tokens > context ||
            !rss_limit || gpu_layers < 0 || (gpu_layers > 0) != (vram_limit > 0) ||
            (parse_special != "0" && parse_special != "1") ||
            (ordered_routing && (batch_tokens != 1 || context > 1024)) ||
            (!capture_routing && mode != "mmap" && mode != "ram" && mode != "inspect")) {
            throw std::runtime_error("invalid runtime mode or dimensions");
        }
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) {
            throw std::runtime_error("could not open prompt");
        }
        const std::string prompt((std::istreambuf_iterator<char>(input)), {});
        llama_backend_init();
        auto model_parameters = llama_model_default_params();
        model_parameters.n_gpu_layers = gpu_layers;
        model_parameters.load_mode = mode == "ram" || mode == "trace-ram" || mode == "trace-events-ram" || mode == "inspect" ?
            LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;
        model_parameters.no_alloc = mode == "inspect";
        auto context_parameters = llama_context_default_params();
        context_parameters.n_ctx = context;
        context_parameters.n_batch = batch_tokens;
        context_parameters.n_ubatch = batch_tokens;
        context_parameters.n_threads = threads;
        context_parameters.n_threads_batch = threads;
        std::array<ggml_backend_dev_t, 2> devices{};
        size_t planned_vram = 0;
        double cuda_preflight_seconds = 0;
        if (gpu_layers) {
            const auto started = clock_type::now();
            cuda_device = ggml_backend_dev_by_name("CUDA0");
            if (!cuda_device) {
                throw std::runtime_error("requested CUDA0 is unavailable in this build");
            }
            devices[0] = cuda_device;
            model_parameters.devices = devices.data();
            size_t total = 0;
            ggml_backend_dev_memory(cuda_device, &cuda_free_start, &total);
            planned_vram = planned_cuda_bytes(argv[1], model_parameters, context_parameters, cuda_device);
            constexpr size_t cuda_overhead = 256u << 20;
            if (!planned_vram || planned_vram + cuda_overhead > std::min(vram_limit, cuda_free_start)) {
                throw std::runtime_error("CUDA plan exceeds available VRAM or the configured cap: " +
                                         std::to_string(planned_vram) + " bytes plus runtime overhead");
            }
            check_resources();
            cuda_preflight_seconds = seconds(started);
        }
        const auto load_start = clock_type::now();
        const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_parameters), llama_model_free);
        if (!model) {
            throw std::runtime_error("could not load model");
        }
        check_resources();
        const auto * vocab = llama_model_get_vocab(model.get());
        const auto token_start = clock_type::now();
        auto tokens = tokenize(vocab, prompt, parse_special == "1");
        const double tokenize_seconds = seconds(token_start);
        if (mode == "inspect") {
            std::ofstream out(argv[3], std::ios::binary);
            out << json{
                {"mode", "native_tensor_layout_and_tokenizer"},
                {"input_tokens", tokens}, {"embedding_length", llama_model_n_embd(model.get())},
                {"layer_slots", llama_model_n_layer(model.get())}, {"peak_rss_bytes", peak_rss()},
                {"planned_vram_bytes", planned_vram},
            }.dump() << '\n';
            if (!out) {
                throw std::runtime_error("could not write inspect receipt");
            }
            return 0;
        }
        if (tokens.size() + max_new > static_cast<size_t>(context)) {
            throw std::runtime_error("prompt and output budget exceed context");
        }
        routing trace;
        trace.ordered = ordered_routing;
        if (capture_routing) {
            context_parameters.cb_eval = routing::callback;
            context_parameters.cb_eval_user_data = &trace;
        }
        const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), context_parameters), llama_free);
        if (!ctx) {
            throw std::runtime_error("could not initialize context");
        }
        check_resources();
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) {
            throw std::runtime_error("could not allocate input batch");
        }
        llama_pos position = 0;
        const auto evaluate = [&](const llama_token * input_tokens, int n) {
            if (ordered_routing) {
                if (n != 1) throw std::runtime_error("ordered trace requires single-token forward");
                trace.begin_token(position);
            }
            llama_batch_ext_clear(batch.get());
            for (int i = 0; i < n; ++i) {
                const int index = llama_batch_ext_add_token(batch.get(), 0, input_tokens[i]);
                if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &position) ||
                    !llama_batch_ext_set_output_logits(batch.get(), index, i == n - 1)) {
                    throw std::runtime_error("could not assemble batch");
                }
                ++position;
            }
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                throw std::runtime_error("native token evaluation failed");
            }
            if (capture_routing && trace.invalid) {
                throw std::runtime_error("could not capture valid native routing weights and their expert IDs");
            }
            if (ordered_routing) trace.finish_token();
            check_resources();
        };
        const double load_seconds = seconds(load_start) - tokenize_seconds;
        const auto prefill_start = clock_type::now();
        for (size_t i = 0; i < tokens.size(); i += context_parameters.n_batch) {
            const int n = std::min<size_t>(context_parameters.n_batch, tokens.size() - i);
            evaluate(tokens.data() + i, n);
        }
        llama_synchronize(ctx.get());
        const double prefill_seconds = seconds(prefill_start);
        trace.result[0].tokens = tokens.size();
        trace.phase = 1;
        int decoded = 0;
        int produced = 0;
        const auto decode_start = clock_type::now();
        const std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(), llama_sampler_free);
        std::string output;
        for (; produced < max_new; ++produced) {
            auto next = llama_sampler_sample(sampler.get(), ctx.get(), -1);
            if (llama_vocab_is_eog(vocab, next)) {
                break;
            }
            output += piece(vocab, next);
            if (produced + 1 < max_new) {
                evaluate(&next, 1);
                ++decoded;
            }
        }
        llama_synchronize(ctx.get());
        const double decode_seconds = seconds(decode_start);
        if (cuda_device) {
            check_resources();
        }
        trace.result[1].tokens = decoded;
        json result = {
            {"content", output}, {"input_tokens", tokens.size()}, {"decoded_tokens", decoded},
            {"embedding_length", llama_model_n_embd(model.get())},
            {"output_tokens", produced}, {"stop", produced == max_new ? "limit" : "eog"},
            {"tokenize_seconds", tokenize_seconds}, {"load_seconds", load_seconds},
            {"prefill_seconds", prefill_seconds}, {"decode_seconds", decode_seconds},
            {"prefill_tps", tokens.size() / prefill_seconds}, {"decode_tps", decoded / decode_seconds},
            {"peak_rss_bytes", peak_rss()},
        };
        if (cuda_device) {
            result["cuda"] = {
                {"device", ggml_backend_dev_description(cuda_device)}, {"gpu_layers_requested", gpu_layers},
                {"preflight_seconds", cuda_preflight_seconds}, {"planned_vram_bytes", planned_vram},
                {"sampled_peak_device_free_delta_bytes", peak_device_delta}, {"vram_cap_bytes", vram_limit},
            };
        }
        if (capture_routing) {
            result["phases"] = trace.receipt();
            if (ordered_routing) result["routing_events"] = std::move(trace.events);
        }
        std::ofstream out(argv[3], std::ios::binary);
        out << result.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
        if (!out) {
            throw std::runtime_error("could not write runtime receipt");
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "inky-matrix: %s\n", error.what());
        return 1;
    }
}
