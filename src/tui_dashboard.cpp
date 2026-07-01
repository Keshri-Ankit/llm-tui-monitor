#include "tui_dashboard.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace ftxui;

// Helper: format a number nicely
static std::string fmt_ms(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << ms << "ms";
    return ss.str();
}

// Helper: sparsity bar
static Element sparsity_bar(float sparsity) {
    int filled = (int)(sparsity * 13);
    std::string bar = "";
    for (int i = 0; i < 13; i++)
        bar += (i < filled) ? "█" : "░";
    bar += " " + std::to_string((int)(sparsity * 100)) + "%";
    Color c = sparsity > 0.8f ? Color::Red :
              sparsity > 0.5f ? Color::Yellow : Color::Green;
    return text(bar) | color(c);
}

// Helper: attention heatmap character
static std::string heat_char(float val, float min_v, float max_v) {
    if (max_v <= min_v) return "░";
    float norm = (val - min_v) / (max_v - min_v);
    if (norm < 0.25f) return "░";
    if (norm < 0.50f) return "▒";
    if (norm < 0.75f) return "▓";
    return "█";
}

void run_dashboard(CaptureLayer& capture) {
    auto screen = ScreenInteractive::Fullscreen();

    // State
    int selected_event = 0;
    int focused_panel = 0;  // 0=stream, 1=metrics, 2=attn, 3=anomaly
    int attn_pan_x = 0, attn_pan_y = 0;

    // ── Panel 1: Live Event Stream (packet table) ──────────────────────────
    auto packet_stream = [&]() -> Element {
        auto evts = capture.events.get_all();
        
        Elements rows;
        // Header
        rows.push_back(
            hbox({
                text(" ID  ") | bold,
                text("│"),
                text(" TIMESTAMP   ") | bold,
                text("│"),
                text(" OP NAME       ") | bold,
                text("│"),
                text(" TENSOR NAME          ") | bold,
                text("│"),
                text(" SHAPE              ") | bold,
                text("│"),
                text(" LAT   ") | bold,
            }) | color(Color::Cyan)
        );
        rows.push_back(separator());

        int start = std::max(0, (int)evts.size() - 12);
        for (int i = start; i < (int)evts.size(); i++) {
            auto& e = evts[i];
            bool sel = (i == selected_event && focused_panel == 0);

            // Format shape
            std::string shape = "[";
            for (int d = 0; d < e.n_dims; d++) {
                if (d) shape += ",";
                shape += std::to_string(e.dims[d]);
            }
            shape += "]";

            // Timestamp
            std::string ts = std::to_string((int)(e.timestamp_ms / 1000)) + "." +
                             std::to_string((int)e.timestamp_ms % 1000);

            auto row = hbox({
                text(" " + std::to_string(i) + " ") | size(WIDTH, EQUAL, 5),
                text("│"),
                text(" " + ts) | size(WIDTH, EQUAL, 13),
                text("│"),
                text(" " + e.op_name) | size(WIDTH, EQUAL, 15),
                text("│"),
                text(" " + e.tensor_name) | size(WIDTH, EQUAL, 22),
                text("│"),
                text(" " + shape) | size(WIDTH, EQUAL, 20),
                text("│"),
                text(" " + fmt_ms(e.latency_ms)) | size(WIDTH, EQUAL, 8),
            });

            if (sel)
                rows.push_back(row | bgcolor(Color::Blue));
            else if (e.is_attention)
                rows.push_back(row | color(Color::Yellow));
            else
                rows.push_back(row);
        }

        bool focused = (focused_panel == 0);
        std::string title = focused ? "█ 2. LIVE PACKET STREAM (Focus Active)" 
                                    : "2. LIVE PACKET STREAM";
        return window(text(title) | color(focused ? Color::Green : Color::White),
                      vbox(rows)) | flex;
    };

    // ── Panel 2: Runtime Metrics ────────────────────────────────────────────
    auto metrics_panel = [&]() -> Element {
        auto evts = capture.events.get_all();
        bool focused = (focused_panel == 1);

        Elements rows;
        if (evts.empty() || selected_event >= (int)evts.size()) {
            rows.push_back(text("No data yet — waiting for model...") 
                          | color(Color::GrayDark));
        } else {
            int idx = std::min(selected_event, (int)evts.size() - 1);
            auto& e = evts[idx];

            std::string shape = "[";
            for (int d = 0; d < e.n_dims; d++) {
                if (d) shape += ", ";
                shape += std::to_string(e.dims[d]);
            }
            shape += "]";

            rows.push_back(hbox({
                text("Tensor Shape : ") | bold,
                text(shape) | color(Color::Cyan),
                text("   Dtype: ") | bold,
                text(e.dtype) | color(Color::Yellow),
            }));
            rows.push_back(hbox({
                text("Sparsity     : ") | bold,
                sparsity_bar(e.sparsity),
            }));
            rows.push_back(hbox({
                text("Latency      : ") | bold,
                text(fmt_ms(e.latency_ms)) | 
                    color(e.latency_ms > 50 ? Color::Red : Color::Green),
                text(e.latency_ms > 50 ? "  ⚠ Slow" : "  ✓ Normal") |
                    color(e.latency_ms > 50 ? Color::Red : Color::Green),
            }));
            rows.push_back(hbox({
                text("Mean Value   : ") | bold,
                text(std::to_string(e.mean_val)) | color(Color::Cyan),
            }));
            rows.push_back(hbox({
                text("Max Value    : ") | bold,
                text(std::to_string(e.max_val)) |
                    color(e.max_val > 6.0f ? Color::Red : Color::Cyan),
            }));
            rows.push_back(hbox({
                text("Device       : ") | bold,
                text(e.device) | color(Color::Magenta),
            }));
            rows.push_back(hbox({
                text("Op Type      : ") | bold,
                text(e.op_name) | color(Color::Yellow),
            }));
        }

        std::string title = focused ? "█ 4. RUNTIME METRICS (Focus Active)" 
                                    : "4. RUNTIME METRICS";
        return window(text(title) | color(focused ? Color::Green : Color::White),
                      vbox(rows));
    };

    // ── Panel 3: Attention Matrix Heatmap ──────────────────────────────────
    auto attention_panel = [&]() -> Element {
        auto evts = capture.events.get_all();
        bool focused = (focused_panel == 2);

        Elements rows;

        // Find the most recent attention event
        CaptureEvent* attn = nullptr;
        for (int i = (int)evts.size() - 1; i >= 0; i--) {
            if (evts[i].is_attention && !evts[i].attn_data.empty()) {
                attn = &evts[i];
                break;
            }
        }

        if (!attn) {
            // Show a demo pattern when no real attention data yet
            rows.push_back(text("Waiting for attention tensors...") 
                          | color(Color::GrayDark));
            rows.push_back(text("(Demo pattern shown below)") 
                          | color(Color::GrayDark));
            rows.push_back(separator());

            // Draw a fake diagonal attention pattern for demo
            int sz = 8;
            std::vector<std::string> tok_labels = {
                "[I]","[want]","[it]","[to]","[be]","[key]","[drv]","[TUI]"
            };
            // Header row
            std::string hdr = "        ";
            for (int j = 0; j < sz; j++) hdr += tok_labels[j] + " ";
            rows.push_back(text(hdr) | color(Color::Yellow));

            for (int i = 0; i < sz; i++) {
                std::string row_str = tok_labels[i];
                while ((int)row_str.size() < 8) row_str += " ";
                for (int j = 0; j < sz; j++) {
                    // Causal attention: can only attend to previous tokens
                    if (j > i) { row_str += "░░ "; continue; }
                    float val = (i == j) ? 1.0f : 0.3f / (i - j + 1);
                    row_str += heat_char(val, 0.0f, 1.0f);
                    row_str += heat_char(val, 0.0f, 1.0f);
                    row_str += " ";
                }
                rows.push_back(text(row_str));
            }
        } else {
            // Real attention data
            float mn = *std::min_element(attn->attn_data.begin(), attn->attn_data.end());
            float mx = *std::max_element(attn->attn_data.begin(), attn->attn_data.end());

            int view_rows = std::min(attn->attn_rows, 8);
            int view_cols = std::min(attn->attn_cols, 8);

            for (int i = 0; i + attn_pan_y < attn->attn_rows && i < view_rows; i++) {
                std::string row_str = "";
                for (int j = 0; j + attn_pan_x < attn->attn_cols && j < view_cols; j++) {
                    int ri = i + attn_pan_y;
                    int ci = j + attn_pan_x;
                    float v = attn->attn_data[ri * attn->attn_cols + ci];
                    row_str += heat_char(v, mn, mx);
                    row_str += heat_char(v, mn, mx);
                    row_str += " ";
                }
                rows.push_back(text(row_str));
            }
        }

        std::string title = focused ? "█ 3. ATTENTION MATRIX (Focus Active)" 
                                    : "3. ATTENTION MATRIX";
        return window(
            text(title) | color(focused ? Color::Green : Color::White),
            vbox({
                vbox(rows),
                separator(),
                text("[h/j/k/l]: Pan  [+/-]: Contrast") | color(Color::GrayDark),
            })
        ) | flex;
    };

    // ── Panel 4: Anomaly Ledger ─────────────────────────────────────────────
    auto anomaly_panel = [&]() -> Element {
        bool focused = (focused_panel == 3);

        Elements rows;
        std::lock_guard<std::mutex> lock(capture.anomaly_mutex);
        
        if (capture.anomalies.empty()) {
            rows.push_back(text("✓ No anomalies detected") | color(Color::Green));
        } else {
            int start = std::max(0, (int)capture.anomalies.size() - 6);
            for (int i = start; i < (int)capture.anomalies.size(); i++) {
                auto& a = capture.anomalies[i];
                rows.push_back(hbox({
                    text(a.timestamp + " ") | color(Color::GrayLight),
                    text(a.message) | color(a.is_error ? Color::Red : Color::Yellow),
                }));
            }
        }

        std::string title = focused ? "█ 5. ANOMALY LEDGER (Focus Active)" 
                                    : "5. ANOMALY LEDGER";
        return window(text(title) | color(focused ? Color::Green : Color::White),
                      vbox(rows));
    };

    // ── Status bar ──────────────────────────────────────────────────────────
    auto status_bar = [&]() -> Element {
        auto evts = capture.events.get_all();
        std::string evt_count = "Events: " + std::to_string(evts.size());
        return hbox({
            text(" [Tab]: Cycle Focus") | color(Color::Cyan),
            text("  [j/k]: Navigate") | color(Color::Cyan),
            text("  [h/l]: Pan Attention") | color(Color::Cyan),
            text("  [q]: Quit") | color(Color::Red),
            text("  " + evt_count) | color(Color::Green) | flex,
        }) | bgcolor(Color::GrayDark);
    };

    // ── Main renderer ───────────────────────────────────────────────────────
    auto renderer = Renderer([&] {
        return vbox({
            // Top section
            hbox({
                // Left: topology tree (simplified for now)
                window(
                    text("1. MODEL TOPOLOGY") | color(Color::White),
                    vbox({
                        text("▼ tinyllama-1.1b") | color(Color::Yellow),
                        text("  ► embed_tokens"),
                        text("  ▼ layers"),
                        text("    ► layers.0"),
                        text("    ► layers.1"),
                        text("    ► layers.2"),
                        text("    ► layers.3"),
                        text("    ► ..."),
                        text("  ► norm"),
                        text("  ► lm_head"),
                    })
                ) | size(WIDTH, EQUAL, 28),
                // Right: live packet stream
                packet_stream(),
            }) | size(HEIGHT, EQUAL, 16),

            // Middle: attention matrix
            attention_panel() | size(HEIGHT, EQUAL, 14),

            // Bottom row
            hbox({
                metrics_panel() | flex,
                anomaly_panel() | flex,
            }) | size(HEIGHT, EQUAL, 10),

            // Status bar
            status_bar(),
        });
    });

    // ── Keyboard handling ────────────────────────────────────────────────────
    auto component = CatchEvent(renderer, [&](Event event) {
        if (event == Event::Character('q') || event == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        if (event == Event::Tab) {
            focused_panel = (focused_panel + 1) % 4;
            return true;
        }
        if (event == Event::Character('j')) {
            auto evts = capture.events.get_all();
            if (!evts.empty())
                selected_event = std::min(selected_event + 1, (int)evts.size() - 1);
            return true;
        }
        if (event == Event::Character('k')) {
            selected_event = std::max(selected_event - 1, 0);
            return true;
        }
        // Attention pan
        if (event == Event::Character('h')) { attn_pan_x = std::max(0, attn_pan_x - 1); return true; }
        if (event == Event::Character('l')) { attn_pan_x++; return true; }
        if (event == Event::Character('H')) { attn_pan_y = std::max(0, attn_pan_y - 1); return true; }
        if (event == Event::Character('L')) { attn_pan_y++; return true; }
        return false;
    });

    // ── Refresh loop (runs on background thread) ────────────────────────────
    std::thread refresh_thread([&] {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });
    refresh_thread.detach();

    screen.Loop(component);
}
