# LLM TUI Monitor — Local Transformer Instrumentation & Tracing Platform

A lightweight, real-time telemetry and diagnostic tool for local transformer models. Built in C++ using llama.cpp and FTXUI, this tool hooks non-invasively into a model's execution pipeline and presents live internal states through an interactive terminal dashboard.

---

## What This Does

When a transformer model (like TinyLlama) processes text, it performs hundreds of mathematical operations across dozens of layers. Normally these are invisible. This tool taps into every single operation as it happens — without modifying the model's source code — and displays the data live in a terminal dashboard.

Think of it like putting a transparent cover on an engine so you can watch all the gears turning in real time.

---

## Features

### Core Features
- **Non-invasive hooking** — Uses llama.cpp's `ggml_backend_sched_eval_callback` to intercept every tensor operation without modifying any model source code
- **Live Packet Stream** — Shows every tensor operation in real time: op name, tensor name, shape, and per-layer latency
- **Fixed-size Ring Buffer** — Stores the last 256 events in a circular buffer so RAM usage stays constant regardless of how long the model runs
- **Interactive TUI** — Full keyboard-driven terminal dashboard with vim-style navigation

### Dashboard Panels
1. **Model Topology** — Tree view of the model architecture (embed_tokens, layers, norm, lm_head)
2. **Live Packet Stream** — Real-time table of tensor operations with timestamps, op names, tensor names, shapes, and latency
3. **Attention Matrix Visualizer** — Block-character heatmap (░▒▓█) of attention scores with pan support
4. **Runtime Metrics Inspector** — Per-tensor details: shape, dtype, sparsity rate, latency delta, mean/max values
5. **Numerical Anomaly Ledger** — Flags unusual events: high activation values (>6.0), high sparsity (>90%), slow layers (>100ms)

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Tab` | Cycle focus between panels |
| `j` / `k` | Navigate up/down in event list |
| `h` / `l` | Pan attention matrix left/right |
| `q` | Quit |

### Metrics Captured Per Tensor
- Operation name (MUL_MAT, SOFT_MAX, RMS_NORM, etc.)
- Tensor name (which layer it belongs to)
- Shape — up to 4 dimensions
- Data type (f32, f16, q4_0, etc.)
- Timestamp and latency since previous operation
- Mean value, max value, sparsity rate
- Device (CPU/CUDA)

---

## Tech Stack

- **C++17**
- **llama.cpp** — runs the transformer model locally, provides the eval callback hook
- **FTXUI** — C++ terminal UI library for the interactive dashboard
- **CMake** — build system (auto-downloads FTXUI via FetchContent)

---

## Prerequisites

- Ubuntu 20.04 or later (or any Linux with glibc)
- At minimum 4GB RAM (8GB recommended)
- Git, CMake 3.14+, GCC/G++ with C++17 support
- Internet connection (to download llama.cpp and FTXUI during build)

Install dependencies:
```bash
sudo apt update
sudo apt install -y build-essential cmake git curl wget
```

---

## Setup and Build Instructions

### 1. Clone this repository
```bash
git clone https://github.com/Keshri-Ankit/llm-tui-monitor.git
cd llm-tui-monitor
```

### 2. Download and build llama.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

### 3. Download a model
Download the free TinyLlama 1.1B model (approx 668MB):
```bash
mkdir -p models
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 4. Build the project
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
CMake automatically downloads and compiles FTXUI during this step. This may take 3-5 minutes on the first build.

### 5. Run
```bash
./build/llm_monitor models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Or with a custom model path:
```bash
./build/llm_monitor /path/to/your/model.gguf
```

The dashboard will appear. The model takes 10-15 seconds to load, after which the Live Packet Stream will begin filling with real tensor operations.

---

## Project Structure
llm-tui-monitor/
├── CMakeLists.txt          # Build system (downloads FTXUI automatically)
├── README.md
├── include/
│   ├── capture_layer.h     # Ring buffer and capture data structures
│   └── tui_dashboard.h     # Dashboard interface
├── src/
│   ├── main.cpp            # Entry point, llama.cpp integration, eval callback
│   ├── capture_layer.cpp   # Tensor capture, stats, anomaly detection
│   └── tui_dashboard.cpp   # All 5 TUI panels, keyboard handling
└── llama.cpp/              # Downloaded separately (see setup instructions)

---

## How the Hook Works

llama.cpp exposes a callback function pointer called `cb_eval` in its context parameters. When set, llama.cpp calls this function before and after every tensor operation during inference. We register our own function:

```cpp
ctx_params.cb_eval = eval_callback;
```

Our callback receives a pointer to the tensor being computed, from which we read:
- The operation type (`t->op`)
- The tensor name (`t->name`)
- The dimensions (`t->ne[0..3]`)
- The data type (`t->type`)

This is entirely non-invasive — zero modifications to llama.cpp source code.

---

## Assumptions

- The project targets CPU inference only (no CUDA required). GPU support can be added by rebuilding llama.cpp with CUDA flags.
- The model must be in GGUF format (the standard format for llama.cpp compatible models).
- Any GGUF model works — TinyLlama is used as the default because it is small and free. Larger models like LLaMA-3-8B will show richer layer data.
- The ring buffer capacity (256 events) is configurable in `include/capture_layer.h` by changing the `CAPACITY` constant.

---

## Additional Features / Future Improvements

- **Replay mode** — Save ring buffer snapshots to disk and replay them later for offline analysis
- **Layer comparison** — Select two layers and diff their activation statistics side by side
- **Export to JSON/CSV** — Dump captured events for external analysis
- **CUDA device tracking** — Show which operations fall back to CPU when running on GPU
- **Token-level tracing** — Trace which tokens caused spikes in activation or latency
- **Real attention matrix sampling** — Currently shows demo pattern; real softmax values can be extracted by matching tensor names containing `kq`

---

## Team

Built as part of GDSC IIT Roorkee project.
Team size: 2 members (Ankit Kumar and Nagesh Desai)
