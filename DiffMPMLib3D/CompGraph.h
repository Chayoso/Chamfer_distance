#pragma once
#include "pch.h"
#include "PointCloud.h"
#include "Grid.h"
#include <math.h>
#include <omp.h>
#include <iostream>
#include <vector>

namespace DiffMPMLib3D {

    // Represents the state of the simulation at a single timestep (or "layer").
    struct CompGraphLayer
    {
        std::shared_ptr<PointCloud> point_cloud = nullptr;
        std::shared_ptr<Grid> grid = nullptr;
        float smoothing_factor_used = 0.9f;  // actual smoothing factor applied in forward pass (for correct backward)
    };
    
    // The main computational graph class that manages the simulation layers and runs the optimization process.
    class CompGraph
    {
    public:
        CompGraph(std::shared_ptr<PointCloud> initial_point_cloud, std::shared_ptr<Grid> grid, std::shared_ptr<const Grid> _target_grid);

        // Main entry point for running the optimization over multiple episodes.
        void OptimizeDefGradControlSequence(
            // SIMULATION PARAMS
            int num_steps, // number of timesteps, aka layers in the comp graph
            float _dt,
            float _drag,
            Vec3 _f_ext,
            // OPTIMIZATION PARAMS
            int control_stride,
            int max_gd_iters,
            int max_line_search_iters,
            float initial_alpha,
            float gd_tol,
            float smoothing_factor,
            int current_episodes,
            // ADAPTIVE ALPHA PARAMS
            bool adaptive_alpha_enabled = true,
            float adaptive_alpha_target_norm = 2500.0f,
            float adaptive_alpha_min_scale = 0.1f,
            // MULTI-PASS SUPPORT
            bool skip_setup = false  // Set true to skip SetUpCompGraph (for pass 2+)
        );

        // Sets up the computational graph by creating copies of the initial state for each layer.
        void SetUpCompGraph(int num_layers);

        // Computes the loss at the final layer based on mass distribution.
        float EndLayerMassLoss();

        // Runs the simulation forward from a given starting layer.
        void ComputeForwardPass(size_t start_layer);
        void ComputeForwardPass(size_t start_layer, int current_episode);
        
        // Runs the backpropagation process from the end of the graph to a given control layer.
        void ComputeBackwardPass(size_t control_layer);

        void OptimizeSingleTimestep(
            int timestep_idx,      // which timestep to optimize
            int max_gd_iters = 1,  // gradient descent iterations
            int current_episode = 0,
            float initial_alpha = 1.0f,  // initial step size
            int max_line_search_iters = 10  // line search iterations
        );

        // Gets the point cloud at a specific timestep.
        std::shared_ptr<PointCloud> GetPointCloudAtTimestep(int timestep_idx);

        // Utility for verifying gradients using finite differences (for debugging).
        void FiniteDifferencesGradientTest(int num_steps, size_t particle_id);

        // Stores all simulation states for each timestep.
        std::vector<CompGraphLayer> layers;
        // A read-only grid representing the target mass distribution.
        std::shared_ptr<const Grid> target_grid;
        std::vector<float> stored_render_grad_F_;  // Flattened (N*9,)
        std::vector<float> stored_render_grad_x_;  // Flattened (N*3,)
        bool has_render_grads_ = false;
        size_t render_grad_num_points_ = 0;
        bool render_grads_injected_this_control_timestep_ = false;  // Prevents double counting
        
        // Runtime scaling for balanced physics/render gradients
        void SetRenderGain(float g) { render_gain_ = g; }
        void SetPhysicsWeight(float w) { physics_weight_ = w; }

        // Smoothing override: when > 0, overrides the episode-based schedule
        void SetSmoothingOverride(float s) { smoothing_override_ = s; }

        // J-min barrier: injects barrier gradient at final layer before backward pass.
        // Set w_barrier=0.0 to disable (default). J_target ~0.8 recommended.
        void SetJBarrier(float J_target, float w_barrier) {
            j_barrier_target_ = J_target;
            j_barrier_weight_ = w_barrier;
        }

        // ── Independent render-Adam ───────────────────────────────────────
        // Backward pass with ONLY render gradients (zeroes physics terminal first).
        // After this call, layers[control_layer].dLdF holds the render-only gradient.
        void ComputeRenderBackwardOnly(size_t control_layer);

        // One render Adam step on the control layer using render-only gradient.
        // Uses momentum_r / vector_r — never touches physics Adam state.
        void ApplyRenderAdamStep(
            size_t control_layer,
            float alpha_r, float beta1, float beta2, float epsilon, int timestep);

        // ||dLdF|| at layers[0] — call after ComputeRenderBackwardOnly.
        float GetRenderGradNorm() const;
        // ─────────────────────────────────────────────────────────────────

        // ── LSRC: Late-Stage Render Control ──────────────────────────────
        // Fc_late is an additive correction applied to F at the FINAL timestep,
        // AFTER ForwardTimeStep. This gives the render gradient ZERO backward-chain
        // attenuation: dL/dFc_late = dL/dF_final (direct, norm ~47 vs ~2.5e-4 for layer[0]).
        // Call EnableFcLate(N) once after SetUpCompGraph, then each episode:
        //   1) ComputeRenderBackwardOnly(layers.size()-1)  -- 0 backward steps
        //   2) ApplyFcLateAdamStep(...)                    -- AMSGrad on fc_late_
        void EnableFcLate(size_t N);
        void ApplyFcLateAdamStep(
            float alpha_late, float beta1, float beta2, float epsilon, int timestep);
        float GetFcLateGradNorm() const;  // ||dL/dFc_late|| = render grad at final layer

        // ── x_late: Late-Stage Position Correction ────────────────────────
        // x_late is an additive correction applied to particle POSITIONS at
        // the FINAL timestep, AFTER ForwardTimeStep.
        // dL/dx_late = dL/dx_final (direct render position grad, norm ~2e-3).
        // Call EnableXLate(N) once after SetUpCompGraph, then each episode:
        //   1) ComputeRenderBackwardOnly(layers.size()-1)  -- 0 backward steps
        //   2) ApplyXLateAdamStep(...)                     -- AMSGrad on x_late_
        void EnableXLate(size_t N);
        void ApplyXLateAdamStep(
            float alpha_late, float beta1, float beta2, float epsilon, int timestep);
        float GetXLateGradNorm() const;  // ||dL/dx_late|| = render pos grad at final layer
        // ─────────────────────────────────────────────────────────────────

        // ── Grid Distance Field ─────────────────────────────────────────
        // Pre-computed distance from each grid node to target surface.
        // Added to EndLayerMassLoss: L += w_d * sum_nodes( mass_i * dist_i^2 )
        void SetDistanceField(const std::vector<float>& dist_field, float weight);

        // ── Native Chamfer Loss ──────────────────────────────────────────
        // Per-particle Chamfer distance to target point cloud.
        // Integrated into line search (combined loss) and backward (proper MPM chain).
        // Call SetChamferTarget() once per episode (or when weight changes).
        void SetChamferTarget(const std::vector<float>& target_points_flat, float weight);
        // Set reverse (target→source) Chamfer weight. 0=unidirectional, 1=equal bidirectional (default).
        void SetChamferRevWeight(float rev_weight) { rev_weight_ = rev_weight; }
        void SetChamferHuberDelta(float delta) { chamfer_huber_delta_ = delta; }
        void SetChamferRevSmoothing(float radius, float clamp_ratio, bool enabled) {
            rev_smooth_radius_ = radius;
            rev_clamp_ratio_ = clamp_ratio;
            rev_smooth_enabled_ = enabled;
        }
        // Compute source→target Chamfer loss on final-layer particles.
        // set_gradients=true: ADDS w_chamfer * dL/dx to existing dLdx (call after EndLayerMassLoss).
        // set_gradients=false: returns loss only (for line search eval).
        float ChamferLoss(bool set_gradients = true);

        // Gradient norm getters for monitoring
        std::pair<double, double> GetLastLayerPhysGradNorm() const;
        std::pair<double, double> GetLayerPhysGradNorm(int layer_idx) const;

        // [FIX] NEW: Get actual physics gradients for PCGrad
        std::pair<std::vector<Mat3>, std::vector<Vec3>> GetLastLayerPhysGradients() const;

    private:
        float render_gain_ = 1.0f;      // Scales injected render grads
        float physics_weight_ = 1.0f;   // Scales physics grads before backprop
        float smoothing_override_ = -1.0f;  // When > 0, overrides episode-based smoothing schedule
        int adam_timestep_ = 0;         // Adam optimizer timestep (persistent across calls)
        int render_adam_timestep_ = 0;  // Independent render Adam timestep

        // LSRC state (persistent across episodes, stored in CompGraph not MaterialPoint)
        bool   fc_late_enabled_  = false;
        size_t fc_late_N_        = 0;
        std::vector<float> fc_late_;        // (N*9) additive F correction at final layer
        std::vector<float> fc_late_m_;      // Adam first moment
        std::vector<float> fc_late_v_;      // Adam second moment
        std::vector<float> fc_late_v_max_;  // AMSGrad max second moment
        int    fc_late_timestep_ = 0;

        // x_late state (persistent across episodes)
        bool   x_late_enabled_  = false;
        size_t x_late_N_        = 0;
        std::vector<float> x_late_;         // (N*3) additive position correction at final layer
        std::vector<float> x_late_m_;       // Adam first moment
        std::vector<float> x_late_v_;       // Adam second moment
        std::vector<float> x_late_v_max_;   // AMSGrad max second moment
        int    x_late_timestep_ = 0;
        float j_barrier_target_ = 0.8f; // Target minimum J = det(F_eff)
        float j_barrier_weight_ = 0.0f; // Barrier weight (0 = disabled)

        // Grid distance field state
        bool   has_distance_field_ = false;
        float  distance_field_weight_ = 0.0f;
        std::vector<float> distance_field_;  // (dim_x * dim_y * dim_z) distances to target surface

        // ── Native Chamfer state ─────────────────────────────────────────
        float  w_chamfer_ = 0.0f;
        float  rev_weight_ = 1.0f;  // Weight for reverse (target→source) Chamfer. 0=unidirectional, 1=equal bidirectional
        float  chamfer_huber_delta_ = 0.0f;  // Huber delta. 0 = disabled (pure L2). >0: Huber(d,δ) for robust CD
        float  rev_smooth_radius_ = 2.0f;   // Gaussian sigma = radius * dx for rev_grad spatial smoothing
        float  rev_clamp_ratio_   = 3.0f;   // Clamp rev_grad magnitudes at ratio × avg magnitude
        bool   rev_smooth_enabled_ = false;  // Enable clamp + spatial smooth of rev_grad
        bool   has_chamfer_target_ = false;
        std::vector<Vec3> chamfer_target_pts_;   // (M,) target positions

        // Spatial hash for O(1) NN lookup (uses simulation grid dimensions)
        struct SpatialHash {
            std::vector<std::vector<size_t>> cells;
            Vec3 min_pt = Vec3::Zero();
            float dx = 1.0f;
            int dims[3] = {0, 0, 0};
            int total_cells() const { return dims[0] * dims[1] * dims[2]; }
            int cell_idx(int i, int j, int k) const {
                return i * dims[1] * dims[2] + j * dims[2] + k;
            }
        };
        SpatialHash chamfer_hash_;
        // Simulation parameters cached for use in member functions.
        Vec3 f_ext = Vec3::Zero();
        float dt = 1.0f / 120.0f;
        float drag = 0.5f;
        float smoothing_factor = 0.1f;
        
        // Helper function to clip gradients to prevent explosions.
        static inline void ClipPointGradients(PointCloud& pc,
            float clip_dLdF = 5e-2f,
            float clip_dLdx = 1e-1f,
            float clip_dLdv = 1e-1f)
        {
    #pragma omp parallel for
            for (int i = 0; i < (int)pc.points.size(); ++i) {
                auto& pt = pc.points[i];
                float nf = pt.dLdF.norm();
                if (nf > clip_dLdF) pt.dLdF *= (clip_dLdF / std::max(nf, 1e-12f));
                // Similar clipping for dLdx and dLdv can be added here if needed.
            }
        }
    };
}