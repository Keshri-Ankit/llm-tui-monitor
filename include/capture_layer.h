#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <array>

// One captured event = one tensor operation the model performed
struct CaptureEvent {
    std::string op_name;        // e.g. "MUL_MAT", "SOFT_MAX"
    std::string tensor_name;    // e.g. "layers.1.attn"
    int dims[4];                // tensor shape, e.g. [4096, 4096, 1, 1]
    int n_dims;                 // how many dimensions are meaningful
    std::string dtype;          // data type e.g. "f16", "f32"
    double timestamp_ms;        // when this happened
    double latency_ms;          // how long since the previous event
    float mean_val;             // average value in the tensor
    float max_val;              // maximum value in the tensor
    float sparsity;             // fraction of values near zero (0.0 to 1.0)
    std::string device;         // "CPU" or "CUDA"
    bool is_attention;          // true if this looks like attention scores
    std::vector<float> attn_data;   // attention values (only filled for attention ops)
    int attn_rows;              // rows in attention matrix
    int attn_cols;              // cols in attention matrix
};

// Anomaly = something unusual we flagged
struct AnomalyEvent {
    std::string timestamp;
    std::string message;
    bool is_error;  // true = red X, false = yellow warning
};

// Fixed-size ring buffer — stores last N events, oldest gets overwritten
class RingBuffer {
public:
    static const int CAPACITY = 256;
    
    void push(const CaptureEvent& event);
    std::vector<CaptureEvent> get_all() const;  // returns copy for safe reading
    int size() const;

private:
    std::array<CaptureEvent, CAPACITY> buffer;
    int head = 0;      // where next write goes
    int count = 0;     // how many valid entries
    mutable std::mutex mtx;
};

// The main capture class
class CaptureLayer {
public:
    RingBuffer events;
    std::vector<AnomalyEvent> anomalies;
    std::mutex anomaly_mutex;

    // Called by llama.cpp for every tensor operation
    // Returns true to allow the operation to proceed
    bool on_tensor(
        const char* op_name,
        const char* tensor_name,
        int n_dims,
        const int64_t* dims,
        int dtype,
        bool is_after   // false = before op, true = after op
    );

    // Sample statistics from raw float data
    static void compute_stats(
        const float* data,
        int n_samples,
        float& out_mean,
        float& out_max,
        float& out_sparsity
    );

private:
    double last_timestamp_ms = 0.0;

    double now_ms() {
        using namespace std::chrono;
        return duration<double, std::milli>(
            high_resolution_clock::now().time_since_epoch()
        ).count();
    }

    void check_anomalies(const CaptureEvent& e);
};
