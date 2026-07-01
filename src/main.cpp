#include <iostream>
#include <thread>
#include <string>
#include <cstring>
#include <atomic>
#include <vector>

#include "llama.h"
#include "ggml.h"

#include "capture_layer.h"
#include "tui_dashboard.h"

static CaptureLayer g_capture;
static std::atomic<bool> g_running{true};

static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (ask) return true;

    const char* op_name     = ggml_op_name(t->op);
    const char* tensor_name = (t->name && t->name[0]) ? t->name : op_name;

    int64_t dims[4];
    int n_dims = 1;
    for (int i = 0; i < 4; i++) {
        dims[i] = t->ne[i];
        if (t->ne[i] > 1) n_dims = i + 1;
    }

    g_capture.on_tensor(op_name, tensor_name, n_dims, dims, (int)t->type, true);
    return true;
}

void inference_thread(const std::string& model_path) {
    std::cerr << "Loading model: " << model_path << std::endl;

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        g_running = false;
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const char* prompt = "Explain how neural networks work.";

    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(
        vocab, prompt, (int)strlen(prompt),
        tokens.data(), (int)tokens.size(), true, false
    );

    if (n_tokens < 0) {
        std::cerr << "Tokenization failed" << std::endl;
        llama_model_free(model);
        g_running = false;
        return;
    }
    tokens.resize(n_tokens);

    while (g_running) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx             = 512;
        ctx_params.n_batch           = 512;
        ctx_params.cb_eval           = eval_callback;
        ctx_params.cb_eval_user_data = nullptr;

        llama_context* ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            std::cerr << "Failed to create context" << std::endl;
            break;
        }

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Decode failed" << std::endl;
            llama_free(ctx);
            break;
        }

        llama_free(ctx);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    llama_model_free(model);
}

int main(int argc, char* argv[]) {
    std::string model_path =
        "/home/ankit/llm-tui/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    if (argc > 1) model_path = argv[1];

    std::thread inf_thread([&] { inference_thread(model_path); });

    run_dashboard(g_capture);

    g_running = false;
    inf_thread.join();

    return 0;
}
