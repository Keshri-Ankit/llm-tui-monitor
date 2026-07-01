#include "capture_layer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <ctime>

// --- RingBuffer implementation ---

void RingBuffer::push(const CaptureEvent& event) {
    std::lock_guard<std::mutex> lock(mtx);
    buffer[head] = event;
    head = (head + 1) % CAPACITY;
    if (count < CAPACITY) count++;
}

std::vector<CaptureEvent> RingBuffer::get_all() const {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<CaptureEvent> result;
    result.reserve(count);
    
    if (count < CAPACITY) {
        // Buffer not full yet, read from 0 to count
        for (int i = 0; i < count; i++)
            result.push_back(buffer[i]);
    } else {
        // Buffer full, read from head (oldest) to head-1 (newest)
        for (int i = 0; i < CAPACITY; i++)
            result.push_back(buffer[(head + i) % CAPACITY]);
    }
    return result;
}

int RingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return count;
}

// --- CaptureLayer implementation ---

void CaptureLayer::compute_stats(
    const float* data, int n_samples,
    float& out_mean, float& out_max, float& out_sparsity
) {
    if (!data || n_samples <= 0) {
        out_mean = out_max = out_sparsity = 0.0f;
        return;
    }

    float sum = 0.0f;
    float mx = data[0];
    int near_zero = 0;
    const float zero_threshold = 1e-6f;

    for (int i = 0; i < n_samples; i++) {
        float v = data[i];
        // Skip NaN and Inf to avoid corrupting stats
        if (std::isnan(v) || std::isinf(v)) continue;
        sum += v;
        if (v > mx) mx = v;
        if (std::abs(v) < zero_threshold) near_zero++;
    }

    out_mean = sum / n_samples;
    out_max = mx;
    out_sparsity = (float)near_zero / n_samples;
}

bool CaptureLayer::on_tensor(
    const char* op_name,
    const char* tensor_name,
    int n_dims,
    const int64_t* dims,
    int dtype,
    bool is_after
) {
    // We only capture AFTER the operation completes (is_after = true)
    // so we can read the output values
    if (!is_after) return true;

    CaptureEvent e;
    e.op_name     = op_name ? op_name : "?";
    e.tensor_name = tensor_name ? tensor_name : "?";
    e.n_dims      = std::min(n_dims, 4);
    for (int i = 0; i < 4; i++)
        e.dims[i] = (i < n_dims) ? (int)dims[i] : 1;

    // Dtype codes from ggml
    switch (dtype) {
        case 0:  e.dtype = "f32";  break;
        case 1:  e.dtype = "f16";  break;
        case 2:  e.dtype = "q4_0"; break;
        case 3:  e.dtype = "q4_1"; break;
        case 6:  e.dtype = "q5_0"; break;
        case 7:  e.dtype = "q5_1"; break;
        case 8:  e.dtype = "q8_0"; break;
        default: e.dtype = "?";    break;
    }

    // Timing
    double now = now_ms();
    e.timestamp_ms = now;
    e.latency_ms = (last_timestamp_ms > 0) ? (now - last_timestamp_ms) : 0.0;
    last_timestamp_ms = now;

    // Detect if this looks like an attention tensor
    std::string name_lower = e.tensor_name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
    e.is_attention = (name_lower.find("kq") != std::string::npos ||
                      name_lower.find("attn") != std::string::npos ||
                      e.op_name == "SOFT_MAX");

    // Device (simplified — real detection needs ggml backend info)
    e.device = "CPU";

    // Default stats (we don't have raw pointer access through this interface easily)
    e.mean_val = 0.0f;
    e.max_val  = 0.0f;
    e.sparsity = 0.0f;

    check_anomalies(e);
    events.push(e);
    return true;
}

void CaptureLayer::check_anomalies(const CaptureEvent& e) {
    // Get current time as string
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char tbuf[20];
    std::strftime(tbuf, sizeof(tbuf), "%H:%M:%S", std::localtime(&t));
    std::string ts = tbuf;

    std::lock_guard<std::mutex> lock(anomaly_mutex);

    if (e.max_val > 6.0f) {
        anomalies.push_back({ts,
            "⚠ Outlier Feature " + e.tensor_name +
            ": Max=" + std::to_string(e.max_val), false});
    }
    if (e.sparsity > 0.9f) {
        anomalies.push_back({ts,
            "⚠ High Sparsity " + e.tensor_name +
            ": " + std::to_string((int)(e.sparsity * 100)) + "%", false});
    }
    if (e.latency_ms > 100.0) {
        anomalies.push_back({ts,
            "⚠ Slow Layer " + e.tensor_name +
            ": " + std::to_string(e.latency_ms) + "ms", false});
    }

    // Keep anomaly list bounded
    if (anomalies.size() > 50)
        anomalies.erase(anomalies.begin());
}
