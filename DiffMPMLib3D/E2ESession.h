#pragma once
#include "pch.h"
#include "CompGraph.h"
#include <functional>
#include <chrono>
#include <vector>

namespace DiffMPMLib3D {

// Configuration for E2E session
struct E2EConfig {
    // Physics parameters
    int num_timesteps = 100;
    int control_stride = 1;
    float dt = 1.0f / 120.0f;
    float drag = 0.5f;
    Vec3 f_ext = Vec3::Zero();

    // Optimization parameters
    int max_gd_iters = 5;
    int max_ls_iters = 10;
    float initial_alpha = 1.0f;
    float gd_tol = 1e-6f;
    float smoothing_factor = 0.1f;

    // Adaptive alpha parameters
    bool adaptive_alpha_enabled = true;
    float adaptive_alpha_target_norm = 2500.0f;
    float adaptive_alpha_min_scale = 0.1f;

    // E2E training parameters
    int num_passes_per_episode = 3;
    bool enable_render_grads = true;

    // Performance settings
    int preallocate_buffer_size = 100000;  // Max particles expected
};

// Result of running an episode
struct EpisodeResult {
    float loss_physics = 0.0f;
    int episode_num = 0;
    int num_passes_executed = 0;
    double wall_time_seconds = 0.0;
    bool success = true;
};

// Statistics for monitoring
struct SessionStatistics {
    int total_episodes = 0;
    int total_passes = 0;
    double total_wall_time = 0.0;
    float best_loss = 1e10f;
    int best_episode = -1;
};

// Render gradient callback type
// Python provides: (episode, pass_idx) -> (dLdF, dLdx) as numpy arrays
using RenderGradientCallback = std::function<bool(
    int episode,
    int pass_idx,
    std::vector<float>& out_dLdF,  // (N*9,) flattened
    std::vector<float>& out_dLdx,  // (N*3,) flattened
    size_t& out_N
)>;

// Main E2E session class
class E2ESession {
public:
    E2ESession(
        std::shared_ptr<CompGraph> cg,
        const E2EConfig& config
    );

    // Run complete episode with all passes
    // If callback is provided, it will be called to get render gradients
    EpisodeResult RunEpisode(
        int episode_num,
        RenderGradientCallback render_callback = nullptr
    );

    // Get final point cloud after episode
    std::shared_ptr<PointCloud> GetFinalPointCloud() const;

    // Get current statistics
    SessionStatistics GetStatistics() const {
        return stats_;
    }

    // Reset statistics (e.g., after loading checkpoint)
    void ResetStatistics() {
        stats_ = SessionStatistics();
    }

    // Chamfer target forwarding (delegates to CompGraph)
    void SetChamferTarget(const std::vector<float>& target_points_flat, float weight) {
        if (cg_) cg_->SetChamferTarget(target_points_flat, weight);
    }
    void SetChamferRevWeight(float rev_weight) {
        if (cg_) cg_->SetChamferRevWeight(rev_weight);
    }
    void SetChamferHuberDelta(float delta) {
        if (cg_) cg_->SetChamferHuberDelta(delta);
    }
    void SetChamferRevSmoothing(float radius, float clamp_ratio, bool enabled) {
        if (cg_) cg_->SetChamferRevSmoothing(radius, clamp_ratio, enabled);
    }

    // Physics weight + smoothing override forwarding
    void SetPhysicsWeight(float w) { if (cg_) cg_->SetPhysicsWeight(w); }
    void SetSmoothingOverride(float s) { if (cg_) cg_->SetSmoothingOverride(s); }

    // Checkpoint support (stubs for future implementation)
    void SaveCheckpoint(const std::string& path) const;
    void LoadCheckpoint(const std::string& path);

private:
    std::shared_ptr<CompGraph> cg_;
    E2EConfig config_;
    SessionStatistics stats_;

    // Persistent buffers (reused across episodes)
    std::vector<float> render_grad_F_buffer_;
    std::vector<float> render_grad_x_buffer_;

    // Internal methods
    void InitializeEpisode(int episode_num);
    bool RunSinglePass(
        int episode_num,
        int pass_idx,
        RenderGradientCallback render_callback
    );
    float ComputePhysicsLoss();
    void InjectRenderGradients(const std::vector<float>& dLdF,
                               const std::vector<float>& dLdx,
                               size_t N);
};

} // namespace DiffMPMLib3D
