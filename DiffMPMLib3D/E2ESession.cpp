#include "E2ESession.h"
#include <iostream>
#include <fstream>

namespace DiffMPMLib3D {

E2ESession::E2ESession(
    std::shared_ptr<CompGraph> cg,
    const E2EConfig& config
) : cg_(cg), config_(config) {

    // Preallocate buffers to avoid reallocation during training
    render_grad_F_buffer_.reserve(config_.preallocate_buffer_size * 9);
    render_grad_x_buffer_.reserve(config_.preallocate_buffer_size * 3);

    std::cout << "[E2ESession] Created session with "
              << config_.num_passes_per_episode << " passes per episode"
              << std::endl;
    std::cout << "[E2ESession] Preallocated buffers for "
              << config_.preallocate_buffer_size << " particles"
              << std::endl;
}

void E2ESession::InitializeEpisode(int episode_num) {
    // Setup computation graph layers
    cg_->SetUpCompGraph(config_.num_timesteps);

    // Run initial forward pass to establish starting state
    cg_->ComputeForwardPass(0, episode_num);

    // Clear any stored render gradients from previous episode
    cg_->has_render_grads_ = false;
    cg_->stored_render_grad_F_.clear();
    cg_->stored_render_grad_x_.clear();
}

void E2ESession::InjectRenderGradients(
    const std::vector<float>& dLdF,
    const std::vector<float>& dLdx,
    size_t N
) {
    // Store gradients in CompGraph for next physics pass
    cg_->stored_render_grad_F_ = dLdF;
    cg_->stored_render_grad_x_ = dLdx;
    cg_->has_render_grads_ = true;
    cg_->render_grad_num_points_ = N;
}

bool E2ESession::RunSinglePass(
    int episode_num,
    int pass_idx,
    RenderGradientCallback render_callback
) {
    // [FIX] CRITICAL FIX: Temporal gradient mismatch
    // OLD: Used pass_idx-1, causing render grads from F_{n-1} + physics grads from F_n
    // NEW: Run forward pass first, then get render grads at CURRENT state

    // If not first pass, establish current state before getting render gradients
    if (pass_idx > 0) {
        // Run forward pass to establish state F_n (current pass)
        cg_->ComputeForwardPass(0, episode_num);
        std::cout << "  [Pass " << pass_idx + 1 << "] Forward pass complete (establishing current state)" << std::endl;
    }

    // NOW get render gradients at CURRENT state (if callback provided)
    if (pass_idx > 0 && render_callback && config_.enable_render_grads) {
        size_t N = 0;

        // [FIX] FIXED: Get render gradients from CURRENT pass state (not previous)
        // This ensures: dL = dL_physics(F_n) + dL_render(F_n) <- CONSISTENT!
        bool got_grads = render_callback(
            episode_num,
            pass_idx,  // <- CHANGED: Use current pass, not pass_idx-1
            render_grad_F_buffer_,
            render_grad_x_buffer_,
            N
        );

        if (got_grads && N > 0) {
            InjectRenderGradients(
                render_grad_F_buffer_,
                render_grad_x_buffer_,
                N
            );

            std::cout << "  [Pass " << pass_idx + 1 << "] Injected render gradients for "
                      << N << " particles (computed at CURRENT state)" << std::endl;
        }
    }

    // Run physics optimization with consistent gradients
    // Both physics and render gradients now computed at same state F_n

    // [FIX] ADAM MOMENTUM FIX: Skip setup for pass 2+ to preserve Adam state
    // Pass 1: skip_setup=false → Reset adam_timestep=0 (fresh optimization)
    // Pass 2+: skip_setup=true → Preserve momentum buffers (accumulated optimization)
    bool skip_setup = (pass_idx > 0);

    cg_->OptimizeDefGradControlSequence(
        config_.num_timesteps,
        config_.dt,
        config_.drag,
        config_.f_ext,
        config_.control_stride,
        config_.max_gd_iters,
        config_.max_ls_iters,
        config_.initial_alpha,
        config_.gd_tol,
        config_.smoothing_factor,
        episode_num,
        config_.adaptive_alpha_enabled,
        config_.adaptive_alpha_target_norm,
        config_.adaptive_alpha_min_scale,
        skip_setup  // [FIX] Preserve Adam momentum across passes!
    );

    return true;
}

float E2ESession::ComputePhysicsLoss() {
    try {
        return cg_->EndLayerMassLoss();
    } catch (...) {
        std::cerr << "[E2ESession] Error computing physics loss" << std::endl;
        return 0.0f;
    }
}

EpisodeResult E2ESession::RunEpisode(
    int episode_num,
    RenderGradientCallback render_callback
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    EpisodeResult result;
    result.episode_num = episode_num;

    std::cout << "\n[E2ESession] ===========================================" << std::endl;
    std::cout << "[E2ESession] Episode " << episode_num << " START" << std::endl;
    std::cout << "[E2ESession] ===========================================" << std::endl;

    try {
        // Initialize episode (setup comp graph, run initial forward pass)
        InitializeEpisode(episode_num);

        // Run all passes
        for (int pass_idx = 0; pass_idx < config_.num_passes_per_episode; ++pass_idx) {
            std::cout << "\n[E2ESession] --- Pass " << pass_idx + 1 << "/"
                      << config_.num_passes_per_episode << " ---" << std::endl;

            bool success = RunSinglePass(episode_num, pass_idx, render_callback);
            if (!success) {
                std::cerr << "[E2ESession] Pass " << pass_idx + 1 << " failed!" << std::endl;
                result.success = false;
                break;
            }
            result.num_passes_executed++;
        }

        // Compute final loss
        result.loss_physics = ComputePhysicsLoss();

        // Update statistics
        stats_.total_episodes++;
        stats_.total_passes += result.num_passes_executed;

        if (result.loss_physics < stats_.best_loss) {
            stats_.best_loss = result.loss_physics;
            stats_.best_episode = episode_num;
        }

    } catch (const std::exception& e) {
        std::cerr << "[E2ESession] Error in episode " << episode_num
                  << ": " << e.what() << std::endl;
        result.success = false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.wall_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    stats_.total_wall_time += result.wall_time_seconds;

    std::cout << "\n[E2ESession] Episode " << episode_num << " COMPLETE" << std::endl;
    std::cout << "  Loss (physics): " << result.loss_physics << std::endl;
    std::cout << "  Passes executed: " << result.num_passes_executed << std::endl;
    std::cout << "  Wall time: " << result.wall_time_seconds << "s" << std::endl;
    std::cout << "  Success: " << (result.success ? "YES" : "NO") << std::endl;
    std::cout << "[E2ESession] ===========================================\n" << std::endl;

    return result;
}

std::shared_ptr<PointCloud> E2ESession::GetFinalPointCloud() const {
    if (cg_->layers.empty()) {
        std::cerr << "[E2ESession] No layers in computation graph!" << std::endl;
        return nullptr;
    }
    return cg_->layers.back().point_cloud;
}

void E2ESession::SaveCheckpoint(const std::string& path) const {
    // TODO: Implement checkpoint serialization
    // Could save: config, statistics, best episode state, etc.
    std::cout << "[E2ESession] Checkpoint saving not yet implemented (path: "
              << path << ")" << std::endl;
}

void E2ESession::LoadCheckpoint(const std::string& path) {
    // TODO: Implement checkpoint loading
    std::cout << "[E2ESession] Checkpoint loading not yet implemented (path: "
              << path << ")" << std::endl;
}

} // namespace DiffMPMLib3D
