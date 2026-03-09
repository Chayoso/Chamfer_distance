#include "pch.h"
#include "CompGraph.h"
#include "ForwardSimulation.h"
#include "BackPropagation.h"
#include "Interpolation.h"
#include <cmath>
#include <numeric> 
#include <fstream>
#include <filesystem>

#include <algorithm>
#ifdef DIAGNOSTICS
#include <mutex>
#endif

namespace DiffMPMLib3D {

    // Use a 'using namespace' directive to simplify function calls within this .cpp file.
    using namespace SingleThreadMPM;

    CompGraph::CompGraph(std::shared_ptr<PointCloud> initial_point_cloud, std::shared_ptr<Grid> grid, std::shared_ptr<const Grid> _target_grid)
    {
        layers.clear();
        layers.resize(1);
        layers[0].point_cloud = std::make_shared<PointCloud>(*initial_point_cloud);
        layers[0].grid = std::make_shared<Grid>(*grid);
        target_grid = _target_grid;
    }

    void CompGraph::SetUpCompGraph(int num_layers)
    {
        assert(num_layers > 0);
        layers.resize(num_layers);

    #pragma omp parallel for 
        for (int i = 1; i < num_layers; i++) {
            layers[i].point_cloud = std::make_shared<PointCloud>(*layers.front().point_cloud);
            layers[i].grid = std::make_shared<Grid>(*layers.front().grid);
        }
    }

   float CompGraph::EndLayerMassLoss()
    {
        float out_of_target_penalty = 5.f;
        float eps = 1e-4f;
        float min_mass = 0.f;  // disabled: let distance field handle mass distribution
        float penalty_weight = 1.f;

        PointCloud& point_cloud = *layers.back().point_cloud;
        Grid& grid = *layers.back().grid;
        float dx = grid.dx;

        G_Reset(grid);
        P2G(point_cloud, grid, 0.f, 0.f);

        float loss = 0.f;
        float dist_loss = 0.f;
        int dim_x = target_grid->dim_x;
        int dim_y = target_grid->dim_y;
        int dim_z = target_grid->dim_z;
        int total_nodes = dim_x * dim_y * dim_z;

        bool use_dist = has_distance_field_ && distance_field_weight_ > 0.f
                        && (int)distance_field_.size() == total_nodes;

    #pragma omp parallel for reduction(+:loss, dist_loss)
        for (int idx = 0; idx < total_nodes; idx++) {
            int i = idx / (dim_y * dim_z);
            int j = (idx / dim_z) % dim_y;
            int k = idx % dim_z;

            float c_m = grid.GetNode(i, j, k).m;
            float t_m = target_grid->GetNode(i, j, k).m;

            float log_diff = std::log(c_m + 1.f + eps) - std::log(t_m + 1.f + eps);
            loss += 0.5f * log_diff * log_diff;
            grid.GetNode(i, j, k).dLdm = log_diff / (c_m + 1.f + eps);

            if (c_m < min_mass) {
                float diff = min_mass - c_m;
                loss += penalty_weight * diff * diff;
                grid.GetNode(i, j, k).dLdm += -2.f * penalty_weight * diff;
            }

            // Distance field penalty: L_dist = w_d * c_m * dist^2
            // dL_dist/dc_m = w_d * dist^2  (added to dLdm)
            if (use_dist) {
                float d2 = distance_field_[idx] * distance_field_[idx];
                dist_loss += distance_field_weight_ * c_m * d2;
                grid.GetNode(i, j, k).dLdm += distance_field_weight_ * d2;
            }
        }

        if (use_dist) {
            loss += dist_loss;
            // Log once per call for monitoring
            static int dist_log_count = 0;
            if (dist_log_count++ % 50 == 0)
                std::cout << "[DistField] dist_loss=" << dist_loss
                          << " (w=" << distance_field_weight_ << ")" << std::endl;
        }

    #pragma omp parallel for
        for (int p = 0; p < (int)point_cloud.points.size(); p++) {
            MaterialPoint& mp = point_cloud.points[p];
            mp.dLdx.setZero();

            std::vector<std::array<int, 3>> indices;
            auto nodes = grid.QueryPoint_CubicBSpline(mp.x, &indices);

            for (int i = 0; i < (int)nodes.size(); i++) {
                const GridNode& node = nodes[i];
                Vec3 dgp = node.x - mp.x;

                Vec3 dgp_div_dx = dgp / dx;
                Vec3 bspline_vals(CubicBSpline(dgp_div_dx[0]), CubicBSpline(dgp_div_dx[1]), CubicBSpline(dgp_div_dx[2]));
                Vec3 bspline_slopes(CubicBSplineSlope(dgp_div_dx[0]), CubicBSplineSlope(dgp_div_dx[1]), CubicBSplineSlope(dgp_div_dx[2]));

                Vec3 wgpGrad = -1.f / dx * Vec3(
                    bspline_slopes[0] * bspline_vals[1] * bspline_vals[2],
                    bspline_vals[0] * bspline_slopes[1] * bspline_vals[2],
                    bspline_vals[0] * bspline_vals[1] * bspline_slopes[2]
                );

                // Use distance field for continuous penalty (replaces binary in/out)
                float penalty;
                if (use_dist) {
                    int ni = indices[i][0], nj = indices[i][1], nk = indices[i][2];
                    float dist = distance_field_[ni * dim_y * dim_z + nj * dim_z + nk];
                    // Near target: penalty=1, far from target: penalty scales up continuously
                    penalty = 1.0f + (out_of_target_penalty - 1.0f) * std::min(dist / (3.0f * dx), 1.0f);
                } else {
                    const auto& target_node = target_grid->GetNode(indices[i][0], indices[i][1], indices[i][2]);
                    penalty = (target_node.m > 1e-12f) ? 1.0f : out_of_target_penalty;
                }
                mp.dLdx += penalty * mp.m * node.dLdm * wgpGrad;
            }

            mp.dLdF.setZero();
            mp.dLdv.setZero();
            mp.dLdC.setZero();
        }

        return loss;
    }

    void CompGraph::SetDistanceField(const std::vector<float>& dist_field, float weight) {
        distance_field_ = dist_field;
        distance_field_weight_ = weight;
        has_distance_field_ = true;
        std::cout << "[DistField] Set: " << dist_field.size() << " nodes, w=" << weight << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NATIVE CHAMFER LOSS — per-particle NN distance to target point cloud
    // ═══════════════════════════════════════════════════════════════════════

    void CompGraph::SetChamferTarget(const std::vector<float>& target_points_flat, float weight) {
        w_chamfer_ = weight;
        size_t M = target_points_flat.size() / 3;
        chamfer_target_pts_.resize(M);
        for (size_t i = 0; i < M; ++i)
            chamfer_target_pts_[i] = Vec3(target_points_flat[i*3], target_points_flat[i*3+1], target_points_flat[i*3+2]);

        // Build spatial hash using simulation grid parameters
        const Grid& grid = *layers.front().grid;
        auto& h = chamfer_hash_;
        h.min_pt = grid.min_point;
        h.dx = grid.dx;
        // Extend hash dims by +2 in each direction to avoid boundary misses
        h.dims[0] = grid.dim_x + 4;
        h.dims[1] = grid.dim_y + 4;
        h.dims[2] = grid.dim_z + 4;
        h.cells.clear();
        h.cells.resize(h.total_cells());

        // Shift min_pt by -2*dx so extended cells start before grid
        Vec3 hash_origin = h.min_pt - Vec3(2.0f * h.dx, 2.0f * h.dx, 2.0f * h.dx);
        h.min_pt = hash_origin;

        int inserted = 0;
        for (size_t i = 0; i < M; ++i) {
            const Vec3& p = chamfer_target_pts_[i];
            int ci = (int)std::floor((p[0] - h.min_pt[0]) / h.dx);
            int cj = (int)std::floor((p[1] - h.min_pt[1]) / h.dx);
            int ck = (int)std::floor((p[2] - h.min_pt[2]) / h.dx);
            if (ci >= 0 && ci < h.dims[0] && cj >= 0 && cj < h.dims[1] && ck >= 0 && ck < h.dims[2]) {
                h.cells[h.cell_idx(ci, cj, ck)].push_back(i);
                inserted++;
            }
        }
        has_chamfer_target_ = true;
        std::cout << "[Chamfer] Set target: " << M << " pts, w=" << weight
                  << ", hash " << h.dims[0] << "x" << h.dims[1] << "x" << h.dims[2]
                  << ", inserted " << inserted << "/" << M << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BIDIRECTIONAL CHAMFER LOSS
    //
    // Direction 1 (source→target): each particle finds nearest target point
    //   → Ensures all particles are close to SOME target point
    //
    // Direction 2 (target→source): each target point finds nearest particle
    //   → Ensures all target regions are COVERED by particles
    //   → Critical for thin structures (bunny ears, tails)
    //
    // Combined: L = w * [ (1/N)Σ||p_i - nn_t(p_i)||² + (1/M)Σ||t_j - nn_p(t_j)||² ]
    // ═══════════════════════════════════════════════════════════════════════
    float CompGraph::ChamferLoss(bool set_gradients) {
        if (!has_chamfer_target_ || w_chamfer_ <= 0.f || chamfer_target_pts_.empty())
            return 0.f;

        PointCloud& pc = *layers.back().point_cloud;
        const int N = (int)pc.points.size();
        const auto& h = chamfer_hash_;       // spatial hash of TARGET points (static)
        const size_t M = chamfer_target_pts_.size();

        // ── DIRECTION 1: Source → Target ──────────────────────────────────
        double fwd_loss_sum = 0.0;
        const float delta = chamfer_huber_delta_;
        const bool use_huber = (delta > 0.f);

    #pragma omp parallel for reduction(+:fwd_loss_sum)
        for (int p = 0; p < N; ++p) {
            const Vec3& xp = pc.points[p].x;
            int ci = (int)std::floor((xp[0] - h.min_pt[0]) / h.dx);
            int cj = (int)std::floor((xp[1] - h.min_pt[1]) / h.dx);
            int ck = (int)std::floor((xp[2] - h.min_pt[2]) / h.dx);

            float best_sq = std::numeric_limits<float>::max();
            int best_idx = -1;

            for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
            for (int dk = -1; dk <= 1; ++dk) {
                int ni = ci + di, nj = cj + dj, nk = ck + dk;
                if (ni < 0 || ni >= h.dims[0] || nj < 0 || nj >= h.dims[1] || nk < 0 || nk >= h.dims[2])
                    continue;
                const auto& cell = h.cells[h.cell_idx(ni, nj, nk)];
                for (size_t idx : cell) {
                    float sq = (xp - chamfer_target_pts_[idx]).squaredNorm();
                    if (sq < best_sq) { best_sq = sq; best_idx = (int)idx; }
                }
            }

            if (best_idx >= 0) {
                const Vec3 diff = xp - chamfer_target_pts_[best_idx];
                const float d = std::sqrt(best_sq);

                if (use_huber && d >= delta) {
                    // Huber: loss = 2δd - δ², grad_dir = 2δ/d * diff / N
                    fwd_loss_sum += (double)(2.0f * delta * d - delta * delta);
                    if (set_gradients) {
                        Vec3 grad = (2.0f * delta / ((float)N * d)) * diff;
                        pc.points[p].dLdx += w_chamfer_ * grad;
                    }
                } else {
                    // L2: loss = d², grad_dir = 2 * diff / N
                    fwd_loss_sum += (double)best_sq;
                    if (set_gradients) {
                        Vec3 grad = (2.0f / (float)N) * diff;
                        pc.points[p].dLdx += w_chamfer_ * grad;
                    }
                }
            }
        }

        // ── DIRECTION 2: Target → Source ──────────────────────────────────
        // Build particle spatial hash on-the-fly (positions change each step)
        struct ParticleHash {
            Vec3 min_pt;
            float dx;
            int dims[3];
            std::vector<std::vector<int>> cells;
            int cell_idx(int i, int j, int k) const { return i * dims[1] * dims[2] + j * dims[2] + k; }
            int total_cells() const { return dims[0] * dims[1] * dims[2]; }
        } ph;

        ph.min_pt = h.min_pt;
        ph.dx = h.dx;
        ph.dims[0] = h.dims[0];  ph.dims[1] = h.dims[1];  ph.dims[2] = h.dims[2];
        ph.cells.resize(ph.total_cells());

        // Insert particles into hash
        for (int p = 0; p < N; ++p) {
            const Vec3& xp = pc.points[p].x;
            int ci = (int)std::floor((xp[0] - ph.min_pt[0]) / ph.dx);
            int cj = (int)std::floor((xp[1] - ph.min_pt[1]) / ph.dx);
            int ck = (int)std::floor((xp[2] - ph.min_pt[2]) / ph.dx);
            if (ci >= 0 && ci < ph.dims[0] && cj >= 0 && cj < ph.dims[1] && ck >= 0 && ck < ph.dims[2])
                ph.cells[ph.cell_idx(ci, cj, ck)].push_back(p);
        }

        // For each target point, find nearest particle
        double rev_loss_sum = 0.0;
        std::vector<Vec3> rev_grad(N, Vec3::Zero());

    #pragma omp parallel for reduction(+:rev_loss_sum)
        for (int t = 0; t < (int)M; ++t) {
            const Vec3& xt = chamfer_target_pts_[t];
            int ci = (int)std::floor((xt[0] - ph.min_pt[0]) / ph.dx);
            int cj = (int)std::floor((xt[1] - ph.min_pt[1]) / ph.dx);
            int ck = (int)std::floor((xt[2] - ph.min_pt[2]) / ph.dx);

            float best_sq = std::numeric_limits<float>::max();
            int best_idx = -1;

            for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
            for (int dk = -1; dk <= 1; ++dk) {
                int ni = ci + di, nj = cj + dj, nk = ck + dk;
                if (ni < 0 || ni >= ph.dims[0] || nj < 0 || nj >= ph.dims[1] || nk < 0 || nk >= ph.dims[2])
                    continue;
                const auto& cell = ph.cells[ph.cell_idx(ni, nj, nk)];
                for (int pidx : cell) {
                    float sq = (xt - pc.points[pidx].x).squaredNorm();
                    if (sq < best_sq) { best_sq = sq; best_idx = pidx; }
                }
            }

            if (best_idx >= 0) {
                const Vec3 diff = pc.points[best_idx].x - xt;
                const float d = std::sqrt(best_sq);

                if (use_huber && d >= delta) {
                    rev_loss_sum += (double)(2.0f * delta * d - delta * delta);
                    if (set_gradients) {
                        Vec3 grad = (2.0f * delta / ((float)M * d)) * diff;
                        #pragma omp atomic
                        rev_grad[best_idx][0] += grad[0];
                        #pragma omp atomic
                        rev_grad[best_idx][1] += grad[1];
                        #pragma omp atomic
                        rev_grad[best_idx][2] += grad[2];
                    }
                } else {
                    rev_loss_sum += (double)best_sq;
                    if (set_gradients) {
                        Vec3 grad = (2.0f / (float)M) * diff;
                        #pragma omp atomic
                        rev_grad[best_idx][0] += grad[0];
                        #pragma omp atomic
                        rev_grad[best_idx][1] += grad[1];
                        #pragma omp atomic
                        rev_grad[best_idx][2] += grad[2];
                    }
                }
            }
        }

        // Apply reverse gradients (scaled by rev_weight_)
        if (set_gradients && rev_weight_ > 0.f) {
            // ── Clamp + Spatial Smooth (reduces gradient concentration) ──
            if (rev_smooth_enabled_) {
                // Stage 1: Clamp outlier magnitudes
                float mag_sum = 0.f;
                int active = 0;
                for (int p = 0; p < N; ++p) {
                    float m = rev_grad[p].norm();
                    if (m > 1e-10f) { mag_sum += m; active++; }
                }
                float avg_mag = active > 0 ? mag_sum / (float)active : 0.f;
                float clamp_mag = rev_clamp_ratio_ * avg_mag;
                int clamped = 0;
                if (clamp_mag > 1e-10f) {
                    for (int p = 0; p < N; ++p) {
                        float m = rev_grad[p].norm();
                        if (m > clamp_mag) {
                            rev_grad[p] *= clamp_mag / m;
                            clamped++;
                        }
                    }
                }

                // Stage 2: Spatial smooth via particle hash (reuse ph built above)
                float sigma_sq = (rev_smooth_radius_ * ph.dx) * (rev_smooth_radius_ * ph.dx);
                float inv_2sig2 = 0.5f / std::max(sigma_sq, 1e-12f);
                std::vector<Vec3> smoothed(N, Vec3::Zero());

            #pragma omp parallel for
                for (int p = 0; p < N; ++p) {
                    const Vec3& xp = pc.points[p].x;
                    int ci = (int)std::floor((xp[0] - ph.min_pt[0]) / ph.dx);
                    int cj = (int)std::floor((xp[1] - ph.min_pt[1]) / ph.dx);
                    int ck = (int)std::floor((xp[2] - ph.min_pt[2]) / ph.dx);
                    Vec3 wsum = Vec3::Zero();
                    float ws = 0.f;
                    for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 1; ++dj)
                    for (int dk = -1; dk <= 1; ++dk) {
                        int ni = ci + di, nj = cj + dj, nk = ck + dk;
                        if (ni < 0 || ni >= ph.dims[0] || nj < 0 || nj >= ph.dims[1] || nk < 0 || nk >= ph.dims[2])
                            continue;
                        const auto& cell = ph.cells[ph.cell_idx(ni, nj, nk)];
                        for (int q : cell) {
                            float sq = (xp - pc.points[q].x).squaredNorm();
                            float w = std::exp(-sq * inv_2sig2);
                            wsum += w * rev_grad[q];
                            ws += w;
                        }
                    }
                    smoothed[p] = (ws > 1e-12f) ? Vec3(wsum / ws) : Vec3::Zero();
                }
                rev_grad = std::move(smoothed);

                // Diagnostic (sparse)
                static int smooth_log_count_ = 0;
                if (smooth_log_count_++ % 20 == 0) {
                    int nonzero = 0;
                    for (int p = 0; p < N; ++p)
                        if (rev_grad[p].norm() > 1e-10f) nonzero++;
                    std::cout << "[Chamfer-Smooth] clamped=" << clamped
                              << "  active_before=" << active
                              << "  nonzero_after=" << nonzero << "/" << N
                              << "  avg_mag=" << avg_mag
                              << "  sigma=" << (rev_smooth_radius_ * ph.dx)
                              << std::endl;
                }
            }

        #pragma omp parallel for
            for (int p = 0; p < N; ++p)
                pc.points[p].dLdx += w_chamfer_ * rev_weight_ * rev_grad[p];
        }

        // ── COMBINED LOSS ─────────────────────────────────────────────────
        float fwd_loss = (float)(fwd_loss_sum / (double)N);
        float rev_loss = (float)(rev_loss_sum / (double)M);
        float loss = w_chamfer_ * (fwd_loss + rev_weight_ * rev_loss);

        static int chamfer_log_count_ = 0;
        if (chamfer_log_count_++ % 20 == 0)
            std::cout << "[Chamfer] loss=" << loss
                      << " (fwd=" << fwd_loss << ", rev=" << rev_loss
                      << ", w=" << w_chamfer_ << ", rev_w=" << rev_weight_
                      << ", huber_d=" << chamfer_huber_delta_
                      << ", N=" << N << ", M=" << M << ")" << std::endl;

        return loss;
    }

    void CompGraph::ComputeForwardPass(size_t start_layer, int current_episode)
    {
        // Compute the episode-scheduled smoothing factor.
        // MUST match ForwardSimulation.cpp ForwardTimeStep's internal schedule exactly,
        // so we can record the correct value for the backward pass.
        float smoothing_scheduled = (smoothing_override_ > 0.f) ? smoothing_override_
                                     : (current_episode < 10 ? 0.88f :
                                        (current_episode < 30 ? 0.90f : 0.92f));

        for (size_t i = start_layer; i < layers.size() - 1; i++)
        {
            // Record the actual smoothing factor used at this layer transition (n → n+1).
            // ComputeBackwardPass reads layers[i+1].smoothing_factor_used, so store in i+1.
            layers[i + 1].smoothing_factor_used = smoothing_scheduled;

            ForwardTimeStep(
                *layers[i + 1].point_cloud,
                *layers[i].point_cloud,
                *layers[i].grid,
                smoothing_scheduled, dt, drag, f_ext, current_episode);

            // LSRC: apply Fc_late as additive correction to F at the final layer.
            // This runs AFTER the last ForwardTimeStep, so it affects only the rendered F
            // (covariances in 3DGS) with ZERO backward-chain attenuation.
            if (fc_late_enabled_ && fc_late_N_ > 0 && i == layers.size() - 2) {
                auto& pc_final = *layers.back().point_cloud;
                size_t N = std::min(pc_final.points.size(), fc_late_N_);
                for (size_t j = 0; j < N; ++j)
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            pc_final.points[j].F(r, c) += fc_late_[j * 9 + r * 3 + c];
            }

            // x_late: apply additive position correction at the final layer.
            // dL/dx_late = dL/dx_final (direct render position gradient, ~2e-3 norm).
            if (x_late_enabled_ && x_late_N_ > 0 && i == layers.size() - 2) {
                auto& pc_final = *layers.back().point_cloud;
                size_t N = std::min(pc_final.points.size(), x_late_N_);
                for (size_t j = 0; j < N; ++j)
                    for (int d = 0; d < 3; ++d)
                        pc_final.points[j].x[d] += x_late_[j * 3 + d];
            }
        }
    }

    void CompGraph::ComputeBackwardPass(size_t control_layer)
    {
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STEP 1: Standard backward propagation (physics)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        // [DEBUG] DEBUG: Print physics_weight_ value at the start of backward pass
        static bool first_time = true;
        if (first_time) {
            std::cout << "[DEBUG] physics_weight_ = " << physics_weight_ << std::endl;
            std::cout << "[DEBUG] has_render_grads_ = " << (has_render_grads_ ? "true" : "false") << std::endl;
            first_time = false;
        }

        // ════════════════════════════════════════════════════════════════════════
        // [FIX] GRADIENT INJECTION FIX: Add render gradients to final layer
        // [CRITICAL] Only inject ONCE per pass using flag check!
        // ════════════════════════════════════════════════════════════════════════
        if (has_render_grads_ && render_grad_num_points_ > 0 && !render_grads_injected_this_control_timestep_) {
            size_t final_layer = layers.size() - 1;
            auto& final_pc = layers[final_layer].point_cloud;
            size_t num_points = std::min(final_pc->points.size(), render_grad_num_points_);

            // ── Step 1: Compute physics and render gradient norms ──
            double phys_dx_norm = 0.0, render_dx_norm = 0.0;

            #pragma omp parallel for reduction(+:phys_dx_norm, render_dx_norm)
            for (size_t i = 0; i < num_points; ++i) {
                phys_dx_norm += final_pc->points[i].dLdx.norm();
                double rx = 0.0;
                for (int d = 0; d < 3; d++) rx += stored_render_grad_x_[i*3+d] * stored_render_grad_x_[i*3+d];
                render_dx_norm += std::sqrt(rx);
            }

            // ── Step 2: Adaptive scale — normalize render to match physics ──
            // render_gain_ acts as α (interpretable fraction):
            //   α=0.1 → Chamfer = 10% of physics, α=0.5 → 50%, etc.
            float alpha = render_gain_;
            float scale = (render_dx_norm > 1e-12)
                        ? (float)(phys_dx_norm / render_dx_norm)
                        : 0.0f;
            float effective_gain = alpha * scale;

            // ── Step 3: Inject normalized render gradients ──
            #pragma omp parallel for
            for (size_t i = 0; i < num_points; ++i) {
                auto& pt = final_pc->points[i];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        pt.dLdF(r, c) += effective_gain * stored_render_grad_F_[i * 9 + r * 3 + c];
                for (int d = 0; d < 3; d++)
                    pt.dLdx[d] += effective_gain * stored_render_grad_x_[i * 3 + d];
            }

            // ── Diagnostic ──
            static int inject_log_count_ = 0;
            if (inject_log_count_ < 20) {  // Log first 20 injections, then quiet
                std::cout << "[Inject] α=" << alpha
                          << "  phys=" << phys_dx_norm
                          << "  render=" << render_dx_norm
                          << "  scale=" << scale
                          << "  gain=" << effective_gain << std::endl;
                inject_log_count_++;
            }

            render_grads_injected_this_control_timestep_ = true;
        }
        // ════════════════════════════════════════════════════════════════════════

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // J-min BARRIER: inject barrier gradient into the final layer.
        // Gets propagated backward through all timesteps alongside physics grad,
        // directly penalizing over-compression (J = det(F_eff) < J_target).
        // Enable via SetJBarrier(J_target, w_barrier); disabled when w_barrier=0.
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if (j_barrier_weight_ > 0.f && !layers.empty()) {
            layers.back().point_cloud->AddJBarrierGradient(j_barrier_target_, j_barrier_weight_);
            std::cout << "[Backward] [J-barrier] target=" << j_barrier_target_
                      << " weight=" << j_barrier_weight_ << std::endl;
        }

        for (int i = (int)layers.size() - 2; i >= (int)control_layer; i--)
        {
            layers[i].grid->ResetGradients();
            layers[i].point_cloud->ResetGradients();

            // Use the smoothing factor that was actually applied in the forward pass at this layer.
            // layers[i+1].smoothing_factor_used was recorded by ComputeForwardPass.
            // This ensures the backward pass is the exact mathematical inverse of the forward.
            float s = layers[i + 1].smoothing_factor_used;
            Back_Timestep(layers[i + 1], layers[i], drag, dt, s);
            
            // [FIX] NEW: Apply physics_weight to balance physics/render signals
            // [WARN]  Only apply if render gradients are present (E2E mode)
            if (physics_weight_ != 1.0f && has_render_grads_) {
                auto& pc = *layers[i].point_cloud;
                auto& grid = *layers[i].grid;

                // Scale point cloud gradients
                #pragma omp parallel for
                for (int p = 0; p < (int)pc.points.size(); ++p) {
                    auto& mp = pc.points[p];
                    mp.dLdF *= physics_weight_;
                    mp.dLdx *= physics_weight_;
                    mp.dLdv *= physics_weight_;
                    mp.dLdC *= physics_weight_;
                }

                // Scale grid gradients
                #pragma omp parallel for
                for (int idx = 0; idx < grid.dim_x * grid.dim_y * grid.dim_z; ++idx) {
                    int ii = idx / (grid.dim_y * grid.dim_z);
                    int jj = (idx / grid.dim_z) % grid.dim_y;
                    int kk = idx % grid.dim_z;
                    auto& node = grid.GetNode(ii, jj, kk);
                    node.dLdv *= physics_weight_;
                    node.dLdm *= physics_weight_;
                    node.dLdp *= physics_weight_;
                }
            }
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // [DISABLED] STEP 2: Control layer injection is INCORRECT
        // Render gradients are from FINAL state - they should ONLY be injected to final layer (line 147)
        // Backward propagation will naturally distribute them to control layers
        // This block is now disabled by the flag check
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if (false && has_render_grads_ && !render_grads_injected_this_control_timestep_ && control_layer < layers.size()) {
            std::shared_ptr<PointCloud> pc_control = layers[control_layer].point_cloud;

            if (!pc_control) {
                std::cerr << "[WARN] Control layer point cloud is null, skipping render gradient injection" << std::endl;
            } else {
                const size_t N = pc_control->points.size();

                if (render_grad_num_points_ != N) {
                    std::cerr << "[WARN] Render gradient size mismatch: stored "
                              << render_grad_num_points_ << " but control layer has " << N
                              << " points. Skipping injection." << std::endl;
                } else {
                    std::cout << "[C++] Injecting render gradients to control layer " << control_layer
                              << " (" << N << " points)" << std::endl;

                    #pragma omp parallel for
                    for (int i = 0; i < (int)N; ++i) {
                        MaterialPoint& pt = pc_control->points[i];

                        // [OK] Build Mat3 from stored render gradient (dLdF)
                        // [FIX] FIX: Python already normalizes gradients - no gain multiplication needed!
                        Mat3 dF_render;
                        dF_render(0,0) = stored_render_grad_F_[i*9 + 0];
                        dF_render(0,1) = stored_render_grad_F_[i*9 + 1];
                        dF_render(0,2) = stored_render_grad_F_[i*9 + 2];
                        dF_render(1,0) = stored_render_grad_F_[i*9 + 3];
                        dF_render(1,1) = stored_render_grad_F_[i*9 + 4];
                        dF_render(1,2) = stored_render_grad_F_[i*9 + 5];
                        dF_render(2,0) = stored_render_grad_F_[i*9 + 6];
                        dF_render(2,1) = stored_render_grad_F_[i*9 + 7];
                        dF_render(2,2) = stored_render_grad_F_[i*9 + 8];

                        // [OK] Build Vec3 from stored render gradient (dLdx)
                        Vec3 dx_render;
                        dx_render(0) = stored_render_grad_x_[i*3 + 0];
                        dx_render(1) = stored_render_grad_x_[i*3 + 1];
                        dx_render(2) = stored_render_grad_x_[i*3 + 2];

                        // [OK] ADD to existing physics gradients (already propagated backward)
                        pt.dLdF += dF_render;
                        pt.dLdx += dx_render;
                    }

                    // Mark as injected to prevent double counting
                    render_grads_injected_this_control_timestep_ = true;

                    std::cout << "[C++] Render gradients injected (L_tot = L_phys_propagated + L_render)"
                              << std::endl;
                }
            }
        }
    }

    void CompGraph::OptimizeDefGradControlSequence(
        int num_steps, float _dt, float _drag, Vec3 _f_ext,
        int control_stride, int max_gd_iters, int max_line_search_iters,
        float initial_alpha, float gd_tol, float _smoothing_factor, int current_episodes,
        bool adaptive_alpha_enabled, float adaptive_alpha_target_norm, float adaptive_alpha_min_scale,
        bool skip_setup)
    {
        dt = _dt;
        drag = _drag;
        smoothing_factor = _smoothing_factor;
        f_ext = _f_ext;

        // ═══════════════════════════════════════════════════════════════════════
        // [CONFIG] OPTIMIZATION CONFIGURATION SUMMARY
        // ═══════════════════════════════════════════════════════════════════════
        std::cout << "\n" << std::string(75, '=') << std::endl;
        std::cout << "[CONFIG] PHYSICS OPTIMIZATION CONFIGURATION" << std::endl;
        std::cout << std::string(75, '=') << std::endl;

        // Simulation Parameters
        std::cout << "[INFO] Simulation Parameters:" << std::endl;
        std::cout << "  ├─ Timesteps:        " << num_steps << std::endl;
        std::cout << "  ├─ dt:               " << dt << std::endl;
        std::cout << "  ├─ Drag:             " << drag << std::endl;
        std::cout << "  ├─ Smoothing (cfg):  " << smoothing_factor
                  << " (override: " << (smoothing_override_ > 0.f ? std::to_string(smoothing_override_) : "off") << ")" << std::endl;
        std::cout << "  └─ External force:   [" << f_ext[0] << ", " << f_ext[1] << ", " << f_ext[2] << "]" << std::endl;

        // Optimization Parameters
        std::cout << "\n[OPT]  Optimization Parameters:" << std::endl;
        std::cout << "  ├─ Control stride:   " << control_stride << std::endl;
        std::cout << "  ├─ Max GD iters:     " << max_gd_iters << std::endl;
        std::cout << "  ├─ Max LS iters:     " << max_line_search_iters << std::endl;
        std::cout << "  ├─ Initial alpha:    " << initial_alpha << std::endl;
        std::cout << "  ├─ GD tolerance:     " << gd_tol << std::endl;
        std::cout << "  └─ Adaptive alpha:   " << (adaptive_alpha_enabled ? "ENABLED" : "DISABLED") << std::endl;

        if (adaptive_alpha_enabled) {
            std::cout << "      ├─ Target norm:  " << adaptive_alpha_target_norm << std::endl;
            std::cout << "      └─ Min scale:    " << adaptive_alpha_min_scale << std::endl;
        }

        // Adam Optimizer Parameters (hardcoded)
        const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-3f;
        std::cout << "\n[ADAM] Adam Optimizer:" << std::endl;
        std::cout << "  ├─ beta1:            " << beta1 << std::endl;
        std::cout << "  ├─ beta2:            " << beta2 << std::endl;
        std::cout << "  └─ epsilon:          " << epsilon << std::endl;

        std::cout << std::string(75, '=') << "\n" << std::endl;

        // [FIX] MULTI-PASS FIX: Only setup on first pass
        if (!skip_setup) {
            std::cout << "[Setup] Initializing computation graph..." << std::endl;
            SetUpCompGraph(num_steps);
            // Reset Adam optimizer state for fresh optimization
            adam_timestep_ = 0;
            std::cout << "[Adam] Reset adam_timestep to 0 (fresh optimization)" << std::endl;
        } else {
            std::cout << "[Setup] Skipping SetUpCompGraph (using existing state from previous pass)" << std::endl;
            std::cout << "[Adam] Continuing from adam_timestep = " << adam_timestep_ << " (preserving momentum)" << std::endl;
        }

        // [FIX] CRITICAL FIX: Reset render gradient injection flag ONCE per pass
        // Must be BEFORE any ComputeBackwardPass calls to prevent double injection
        render_grads_injected_this_control_timestep_ = false;

        ComputeForwardPass(0, current_episodes);

        auto eval_loss = [&]() -> float {
            float phys = EndLayerMassLoss();
            float chamfer = (w_chamfer_ > 0.f && has_chamfer_target_)
                            ? ChamferLoss(false)  // loss only, no gradient
                            : 0.f;
            return phys + chamfer;
        };
        float initial_loss = eval_loss();
        std::cout << "Initial loss = " << initial_loss;
        if (w_chamfer_ > 0.f && has_chamfer_target_)
            std::cout << " (includes Chamfer w=" << w_chamfer_ << ")";
        std::cout << std::endl;

        ComputeBackwardPass(0);
        float initial_norm_global = layers.front().point_cloud->Compute_dLdF_Norm();
        std::cout << "Initial global gradient norm = " << initial_norm_global << std::endl;

        size_t num_points = layers.front().point_cloud->points.size();
        std::vector<Mat3> dFc_bak(num_points), m_bak(num_points), v_bak(num_points), vmax_bak(num_points);

        // =======================================================================
        // Single-pass optimization (multipass handled in Python E2E loop)
        // Python controls Pass 1/2/3, C++ does one optimization pass per call
        // =======================================================================
        int totalTemporalIterations = 1;
        std::cout << "Starting optimization (Episode " << current_episodes << ")" << std::endl;
        std::cout << std::string(75, '-') << std::endl;

        for (int temporalIter = 0; temporalIter < totalTemporalIterations; ++temporalIter)
        {
            int num_control_steps = (num_steps - 1 + control_stride - 1) / control_stride;
            for (int control_timestep = 0; control_timestep < num_steps - 1; control_timestep += control_stride)
            {
                int step_number = control_timestep / control_stride + 1;
                std::cout << "\n[STEP] Control Step " << step_number << "/" << num_control_steps
                          << " (Timestep " << control_timestep << ")" << std::endl;

                // [FIX] Reset per control timestep so render grads contribute to ALL timesteps.
                render_grads_injected_this_control_timestep_ = false;

                // ── PHYSICS-ONLY BACKWARD: get norm for Adam normalization ──
                // By suppressing render injection here, we get the pure physics
                // gradient norm.  This norm is passed to Descend_Adam instead of
                // the combined norm, so the Chamfer signal is NOT erased by
                // inv_gn = 1/gradient_norm  (the root cause of convergence-floor).
                float phys_grad_norm_for_adam = 0.f;
                {
                    bool saved_has_render = has_render_grads_;
                    has_render_grads_ = false;            // suppress render injection
                    ComputeBackwardPass(control_timestep);
                    phys_grad_norm_for_adam = std::max(
                        layers[control_timestep].point_cloud->Compute_dLdF_Norm(), 1e-12f);
                    has_render_grads_ = saved_has_render;  // restore
                    render_grads_injected_this_control_timestep_ = false;  // reset for next backward
                }

                // ── Adaptive Chamfer Weight ─────────────────────────────────
                // Scale w_chamfer so ChamferLoss ≈ w_user × PhysicsLoss.
                // This makes w_chamfer an intuitive relative weight (1.0 = equal)
                // and fixes the ~5000:1 magnitude gap between physics and chamfer.
                float saved_w_chamfer = w_chamfer_;
                if (w_chamfer_ > 0.f && has_chamfer_target_) {
                    float ph = EndLayerMassLoss();
                    float ch_raw = ChamferLoss(false) / std::max(w_chamfer_, 1e-12f);
                    if (ch_raw > 1e-12f) {
                        w_chamfer_ = saved_w_chamfer * ph / ch_raw;
                        std::cout << "  [AdaptiveW] phys=" << ph << "  ch_raw=" << ch_raw
                                  << "  w_adaptive=" << w_chamfer_
                                  << " (user=" << saved_w_chamfer << ")" << std::endl;
                    }
                }

                // ADAPTIVE INITIAL_ALPHA: calibrate against physics-only gradient
                float alpha = initial_alpha;

                if (adaptive_alpha_enabled) {
                    float current_grad_norm = phys_grad_norm_for_adam;

                    float alpha_scale = std::min(1.0f, adaptive_alpha_target_norm / std::max(current_grad_norm, 1e-6f));
                    alpha_scale = std::max(alpha_scale, adaptive_alpha_min_scale);

                    alpha = initial_alpha * alpha_scale;

                    if (alpha_scale < 1.0f) {
                        std::cout << "  [Adaptive Alpha] grad_norm=" << current_grad_norm
                                  << ", target=" << adaptive_alpha_target_norm
                                  << ", scale=" << alpha_scale
                                  << ", alpha=" << alpha << " (reduced from " << initial_alpha << ")" << std::endl;
                    }
                } else {
                    // Physics-only backward already done above — nothing extra needed
                    std::cout << "  [Fixed Alpha] alpha=" << alpha << " (adaptive disabled)" << std::endl;
                }

                float initial_norm_local = 0.f;

                for (int gd_iter = 0; gd_iter < max_gd_iters; ++gd_iter)
                {
                    // [DEBUG] Track Adam timestep for momentum debugging
                    if (gd_iter == 0) {
                        std::cout << "  [Adam] Current adam_timestep = " << adam_timestep_
                                  << " (next update will use timestep " << (adam_timestep_ + 1) << ")" << std::endl;
                    }

                    ComputeForwardPass(control_timestep, current_episodes);
                    float gd_loss = eval_loss();

#ifdef DIAGNOSTICS
                    // --- DIAG: Log J statistics for the current forward state
                    {
                        static std::once_flag header_once;
                        static std::mutex io_mtx;
                        auto& pc_last = *layers.back().point_cloud;
                        std::vector<float> J = pc_last.GetPointDeterminants(); // det(F + dFc)
                        if (!J.empty()) {
                            std::vector<float> JJ = J; std::sort(JJ.begin(), JJ.end());
                            auto P = [&](float q){ size_t i = (size_t)std::floor(q * (JJ.size()-1));
                                                   return JJ[std::min(i, JJ.size()-1)]; };
                            const float j_min = JJ.front();
                            const float j_mean = std::accumulate(J.begin(), J.end(), 0.f) / float(J.size());
                            std::ofstream ofs("diag_opt.csv", std::ios::app);
                            std::call_once(header_once, [&](){
                                ofs << "pass,step,gd_iter,phase,loss,j_min,j_mean,j_p01,j_p50,j_p99,alpha_try,ls_iters,accepted\n";
                            });
                            ofs << temporalIter << "," << control_timestep << "," << gd_iter
                                << ",pre_ls," << gd_loss << ","
                                << j_min << "," << j_mean << "," << P(0.01f) << "," << P(0.50f) << "," << P(0.99f)
                                << "," << initial_alpha << "," << 0 << ",-1\n";
                        }
                    }
#endif

                    if (!std::isfinite(gd_loss)) {
                        std::cout << "Warning: Non-finite loss detected. Aborting step." << std::endl;
                        break;
                    }

                    // ── COMBINED BACKWARD: physics + native Chamfer ──
                    // 1. EndLayerMassLoss sets physics dLdx (zeros first)
                    // 2. ChamferLoss ADDS chamfer dLdx on top
                    // 3. BackwardPass propagates combined gradient through MPM chain
                    EndLayerMassLoss();  // sets dLdx_physics on final layer
                    if (w_chamfer_ > 0.f && has_chamfer_target_)
                        ChamferLoss(true);  // ADDS w_chamfer * dL_chamfer/dx

                    render_grads_injected_this_control_timestep_ = false;
                    ComputeBackwardPass(control_timestep);

                    auto& pc = *layers[control_timestep].point_cloud;

                    float gradient_norm = std::max(pc.Compute_dLdF_Norm(), 1e-12f);

                    // ── DIAGNOSTIC ──
                    std::cout << "  [DIAG] ||dLdFc|| combined=" << gradient_norm
                              << "  phys_only=" << phys_grad_norm_for_adam
                              << "  ratio=" << (gradient_norm / phys_grad_norm_for_adam)
                              << ((w_chamfer_ > 0.f) ? "  +Chamfer" : "")
                              << std::endl;

                    if (gd_iter == 0) {
                        initial_norm_local = gradient_norm;
                    }

                    if (gradient_norm < gd_tol * initial_norm_local) {
                        std::cout << "Converged at GD iteration " << gd_iter << "." << std::endl;
                        break;
                    }

                #pragma omp parallel for
                    for (int i = 0; i < num_points; ++i) {
                        dFc_bak[i] = pc.points[i].dFc;
                        m_bak[i] = pc.points[i].momentum;
                        v_bak[i] = pc.points[i].vector;
                        vmax_bak[i] = pc.points[i].vector_max;
                    }

                    bool step_accepted = false;
                    float alpha_try = alpha;

                    for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter)
                    {
                        // [KEY FIX] Use physics-only norm for Adam normalization.
                        // Combined norm erases Chamfer signal (inv_gn = 1/||phys+chamfer||).
                        // Physics-only norm preserves it: grad = (phys+chamfer)/||phys||
                        //   → phys part ≈ unit direction (unchanged)
                        //   → chamfer part = α × render_unit_dir (PRESERVED)
                        pc.Descend_Adam(alpha_try, phys_grad_norm_for_adam, beta1, beta2, epsilon, adam_timestep_ + 1);

                        // ── DIAGNOSTIC: 4-value logging ──
                        {
                            float actual_step_sq = 0.f;
                            #pragma omp parallel for reduction(+:actual_step_sq)
                            for (int i = 0; i < (int)num_points; ++i) {
                                actual_step_sq += (pc.points[i].dFc - dFc_bak[i]).squaredNorm();
                            }
                            float actual_step_norm = std::sqrt(actual_step_sq);
                            float inv_gn = 1.0f / phys_grad_norm_for_adam;
                            std::cout << "    [STEP-DIAG] ||phys||=" << phys_grad_norm_for_adam
                                      << "  ||combined||=" << gradient_norm
                                      << "  inv_gn=" << inv_gn
                                      << "  ||actual_step||=" << actual_step_norm
                                      << "  (alpha_try=" << alpha_try << ")" << std::endl;
                        }

                        ComputeForwardPass(control_timestep, current_episodes);
                        float new_loss = eval_loss();

                        if (std::isfinite(new_loss) && new_loss < gd_loss) {
                            adam_timestep_++;
                            std::cout << "    [Adam] Step accepted! adam_timestep incremented: "
                                      << (adam_timestep_ - 1) << " → " << adam_timestep_ << std::endl;
                            alpha = std::min(alpha_try * 1.1f, initial_alpha);
                            step_accepted = true;
#ifdef DIAGNOSTICS
                            // mark acceptance
                            std::ofstream ofs("diag_opt.csv", std::ios::app);
                            ofs << temporalIter << "," << control_timestep << "," << gd_iter
                                << ",ls_accept," << new_loss << ",,,,,,"
                                << "," << alpha << "," << 0 << "," << 1 << "\n";
#endif
                            
                            break;
                        }

                    #pragma omp parallel for
                        for (int i = 0; i < num_points; ++i) {
                            pc.points[i].dFc = dFc_bak[i];
                            pc.points[i].momentum = m_bak[i];
                            pc.points[i].vector = v_bak[i];
                            pc.points[i].vector_max = vmax_bak[i];
                        }
                        alpha_try *= 0.5f;
                    }

                    if (!step_accepted) {
                        std::cout << "Line search failed. Moving to next control step." << std::endl;

#ifdef DIAGNOSTICS
                        std::ofstream ofs("diag_opt.csv", std::ios::app);
                        ofs << temporalIter << "," << control_timestep << "," << gd_iter
                            << ",ls_fail," << gd_loss << ",,,,,,"
                            << "," << alpha << "," << max_line_search_iters << "," << 0 << "\n";
#endif
                        break;
                    }
                } // End gradient descent loop

                // Restore original w_chamfer (adaptive override was per-control-step)
                w_chamfer_ = saved_w_chamfer;
            } // End control timestep loop
        } // --- End temporalIter (multipass) loop ---
        
        float final_loss = eval_loss();
        std::cout << "\n" << std::string(75, '=') << std::endl;
        std::cout << "[OK] Optimization Complete " << std::endl;
        std::cout << std::string(75, '=') << std::endl;
        std::cout << "  Initial loss: " << initial_loss << std::endl;
        std::cout << "  Final loss:   " << final_loss << std::endl;
        std::cout << "  Reduction:    " << (initial_loss - final_loss) << " ("
                  << (100.0f * (initial_loss - final_loss) / std::max(initial_loss, 1e-6f)) << "%)" << std::endl;
        std::cout << std::string(75, '=') << "\n" << std::endl;
    }
    void CompGraph::OptimizeSingleTimestep(
        int timestep_idx,
        int max_gd_iters,
        int current_episode,
        float initial_alpha,
        int max_line_search_iters)
    {
        // Validate index
        if (timestep_idx < 0 || timestep_idx >= (int)layers.size() - 1) {
            std::cerr << "Invalid timestep index: " << timestep_idx << std::endl;
            return;
        }
        
        std::cout << "\n[Single Timestep Optimization]" << std::endl;
        std::cout << "  ├─ Timestep:      " << timestep_idx << std::endl;
        std::cout << "  ├─ Initial alpha: " << initial_alpha << std::endl;
        std::cout << "  ├─ Max GD iters:  " << max_gd_iters << std::endl;
        std::cout << "  └─ Max LS iters:  " << max_line_search_iters << std::endl;
        
        auto& pc = *layers[timestep_idx].point_cloud;
        size_t num_points = pc.points.size();
        
        // Backup for line search
        std::vector<Mat3> dFc_bak(num_points);
        std::vector<Mat3> m_bak(num_points);
        std::vector<Mat3> v_bak(num_points);
        std::vector<Mat3> vmax_bak(num_points);
        
        // Adam parameters
        const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-3f;
        // NOTE: Using member variable adam_timestep_ (persistent across calls)
        
        float alpha = initial_alpha;
        
        // Gradient descent loop
        for (int gd_iter = 0; gd_iter < max_gd_iters; ++gd_iter)
        {
            // 1. Forward pass from this timestep
            ComputeForwardPass(timestep_idx, current_episode);
            float gd_loss = EndLayerMassLoss();
            
            if (!std::isfinite(gd_loss)) {
                std::cout << "Warning: Non-finite loss. Aborting." << std::endl;
                break;
            }
            
            // 2. Backward pass to this timestep
            ComputeBackwardPass(timestep_idx);
            
            float gradient_norm = std::max(pc.Compute_dLdF_Norm(), 1e-12f);
            
            // 3. Backup current state
            #pragma omp parallel for
            for (int i = 0; i < (int)num_points; ++i) {
                dFc_bak[i] = pc.points[i].dFc;
                m_bak[i] = pc.points[i].momentum;
                v_bak[i] = pc.points[i].vector;
                vmax_bak[i] = pc.points[i].vector_max;
            }
            
            // 4. Line search
            bool step_accepted = false;
            float alpha_try = alpha;
            
            for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter)
            {
                // Try Adam update
                pc.Descend_Adam(alpha_try, gradient_norm, beta1, beta2, epsilon, adam_timestep_ + 1);
                
                // Forward pass to evaluate new loss
                ComputeForwardPass(timestep_idx, current_episode);
                float new_loss = EndLayerMassLoss();
                
                if (std::isfinite(new_loss) && new_loss < gd_loss) {
                    // Accept step
                    adam_timestep_++;
                    alpha = std::min(alpha_try * 1.1f, initial_alpha);
                    step_accepted = true;
                    std::cout << "  Step accepted at ls_iter=" << ls_iter 
                              << ", loss=" << new_loss << std::endl;
                    break;
                }
                
                // Restore state and reduce step size
                #pragma omp parallel for
                for (int i = 0; i < (int)num_points; ++i) {
                    pc.points[i].dFc = dFc_bak[i];
                    pc.points[i].momentum = m_bak[i];
                    pc.points[i].vector = v_bak[i];
                    pc.points[i].vector_max = vmax_bak[i];
                }
                alpha_try *= 0.5f;
            }
            
            if (!step_accepted) {
                std::cout << "  Line search failed." << std::endl;
                break;
            }
        }
    }
    
    std::shared_ptr<PointCloud> CompGraph::GetPointCloudAtTimestep(int timestep_idx)
    {
        if (timestep_idx < 0 || timestep_idx >= (int)layers.size()) {
            std::cerr << "Invalid timestep index: " << timestep_idx << std::endl;
            return nullptr;
        }
        return layers[timestep_idx].point_cloud;
    }
    
    // ============================================================================
    // Independent Render-Adam
    // ============================================================================

    void CompGraph::ComputeRenderBackwardOnly(size_t control_layer)
    {
        if (!has_render_grads_ || render_grad_num_points_ == 0) {
            std::cout << "[RenderBackward] No render grads stored — skipping." << std::endl;
            return;
        }

        // 1. Clear ALL gradients on the final layer (remove any physics terminal).
        {
            auto& final_pc   = *layers.back().point_cloud;
            auto& final_grid = *layers.back().grid;
            for (auto& mp : final_pc.points) {
                mp.dLdF.setZero();
                mp.dLdx.setZero();
                mp.dLdv.setZero();
                mp.dLdC.setZero();
            }
            final_grid.ResetGradients();
        }

        // 2. Inject render gradients (render_gain_ already baked in during store step).
        {
            auto& final_pc = *layers.back().point_cloud;
            size_t num_pts = std::min(final_pc.points.size(), render_grad_num_points_);

            #pragma omp parallel for
            for (int i = 0; i < (int)num_pts; ++i) {
                auto& pt = final_pc.points[i];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        pt.dLdF(r, c) = stored_render_grad_F_[i * 9 + r * 3 + c];
                for (int d = 0; d < 3; d++)
                    pt.dLdx[d] = stored_render_grad_x_[i * 3 + d];
            }
        }

        // 3. Backward through all layers — render gradient only.
        for (int i = (int)layers.size() - 2; i >= (int)control_layer; i--)
        {
            layers[i].grid->ResetGradients();
            layers[i].point_cloud->ResetGradients();
            float s = layers[i + 1].smoothing_factor_used;
            Back_Timestep(layers[i + 1], layers[i], drag, dt, s);
        }

        // Diagnostic
        double norm_F = 0.0, norm_x = 0.0;
        const auto& pc0 = *layers[control_layer].point_cloud;
        #pragma omp parallel for reduction(+:norm_F, norm_x)
        for (int i = 0; i < (int)pc0.points.size(); ++i) {
            norm_F += pc0.points[i].dLdF.squaredNorm();
            norm_x += pc0.points[i].dLdx.squaredNorm();
        }
        std::cout << "[RenderBackward] ||dLdF||=" << std::sqrt(norm_F)
                  << "  ||dLdx||=" << std::sqrt(norm_x) << std::endl;
    }

    void CompGraph::ApplyRenderAdamStep(
        size_t control_layer,
        float alpha_r, float beta1, float beta2, float epsilon, int timestep)
    {
        if (control_layer >= layers.size() || !layers[control_layer].point_cloud) return;

        auto& pc = *layers[control_layer].point_cloud;
        float grad_norm = std::max(pc.Compute_dLdF_Norm(), 1e-12f);

        std::cout << "[RenderAdam] alpha_r=" << alpha_r
                  << "  grad_norm=" << grad_norm
                  << "  t=" << timestep << std::endl;

        pc.Descend_Adam_Render(alpha_r, grad_norm, beta1, beta2, epsilon, timestep);
    }

    float CompGraph::GetRenderGradNorm() const
    {
        if (layers.empty() || !layers.front().point_cloud) return 0.f;
        const auto& pc = *layers.front().point_cloud;
        double sq = 0.0;
        #pragma omp parallel for reduction(+:sq)
        for (int i = 0; i < (int)pc.points.size(); ++i)
            sq += pc.points[i].dLdF.squaredNorm();
        return (float)std::sqrt(sq);
    }

    // ============================================================================
    // [FIX] NEW: Gradient Norm Monitoring
    // ============================================================================
    
    std::pair<double, double> CompGraph::GetLastLayerPhysGradNorm() const {
        if (layers.empty() || !layers.back().point_cloud) {
            return {0.0, 0.0};
        }
        
        const auto& pc = *layers.back().point_cloud;
        double gF2 = 0.0, gx2 = 0.0;
        
        #pragma omp parallel for reduction(+:gF2,gx2)
        for (int i = 0; i < (int)pc.points.size(); ++i) {
            const auto& p = pc.points[i];
            
            // dLdF Frobenius norm
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double v = p.dLdF(r, c);
                    gF2 += v * v;
                }
            }
            
            // dLdx L2 norm
            for (int k = 0; k < 3; ++k) {
                double v = p.dLdx(k);
                gx2 += v * v;
            }
        }
        
        return {std::sqrt(gF2), std::sqrt(gx2)};
    }
    
    std::pair<double, double> CompGraph::GetLayerPhysGradNorm(int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= (int)layers.size() || !layers[layer_idx].point_cloud) {
            return {0.0, 0.0};
        }

        const auto& pc = *layers[layer_idx].point_cloud;
        double gF2 = 0.0, gx2 = 0.0;

        #pragma omp parallel for reduction(+:gF2,gx2)
        for (int i = 0; i < (int)pc.points.size(); ++i) {
            const auto& p = pc.points[i];

            // dLdF Frobenius norm
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double v = p.dLdF(r, c);
                    gF2 += v * v;
                }
            }

            // dLdx L2 norm
            for (int k = 0; k < 3; ++k) {
                double v = p.dLdx(k);
                gx2 += v * v;
            }
        }

        return {std::sqrt(gF2), std::sqrt(gx2)};
    }

    // [FIX] NEW: Get actual physics gradients for PCGrad
    std::pair<std::vector<Mat3>, std::vector<Vec3>> CompGraph::GetLastLayerPhysGradients() const {
        if (layers.empty() || !layers.back().point_cloud) {
            return {{}, {}};
        }

        const auto& pc = *layers.back().point_cloud;
        return {pc.GetPointDefGradGradients(), pc.GetPointPositionGradients()};
    }

    // ============================================================================
    // LSRC: Late-Stage Render Control
    // ============================================================================

    void CompGraph::EnableFcLate(size_t N) {
        // Idempotent: if already initialized with the same N, do not reset Adam state.
        if (fc_late_enabled_ && fc_late_N_ == N) {
            // Already initialized — just ensure the fc_late is applied this episode.
            return;
        }
        fc_late_enabled_ = true;
        fc_late_N_       = N;
        fc_late_.assign(N * 9, 0.0f);
        fc_late_m_.assign(N * 9, 0.0f);
        fc_late_v_.assign(N * 9, 0.0f);
        fc_late_v_max_.assign(N * 9, 0.0f);
        fc_late_timestep_ = 0;
        std::cout << "[LSRC] EnableFcLate: N=" << N
                  << "  (fc_late_ applied at final layer, 0 backward steps)" << std::endl;
    }

    void CompGraph::ApplyFcLateAdamStep(
        float alpha, float beta1, float beta2, float epsilon, int timestep)
    {
        if (!fc_late_enabled_ || fc_late_N_ == 0 || layers.empty()) return;

        // Gradient = dLdF at the FINAL layer (render grad, zero attenuation).
        const auto& pc_final = *layers.back().point_cloud;
        size_t N = std::min(pc_final.points.size(), fc_late_N_);

        // Compute gradient norm for normalisation (same as physics Adam).
        double gnorm2 = 0.0;
        #pragma omp parallel for reduction(+:gnorm2)
        for (int j = 0; j < (int)N; ++j)
            gnorm2 += pc_final.points[j].dLdF.squaredNorm();
        float gnorm  = std::max((float)std::sqrt(gnorm2), 1e-12f);
        float inv_gn = 1.0f / gnorm;

        constexpr float kMaxStepAbs = 5e-3f;
        constexpr float kMaxStepRel = 5e-2f;
        constexpr float kVFloor     = 1e-8f;
        int t = std::max(1, timestep);
        ++fc_late_timestep_;

        #pragma omp parallel for
        for (int j = 0; j < (int)N; ++j) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    int   idx  = j * 9 + r * 3 + c;
                    float g    = pc_final.points[j].dLdF(r, c) * inv_gn;
                    float& m   = fc_late_m_[idx];
                    float& v   = fc_late_v_[idx];
                    float& vm  = fc_late_v_max_[idx];
                    float& x   = fc_late_[idx];

                    m  = beta1 * m + (1.0f - beta1) * g;
                    v  = beta2 * v + (1.0f - beta2) * g * g;
                    vm = std::max(vm, v);  // AMSGrad

                    float m_hat = m  / (1.0f - std::pow(beta1, t));
                    float v_hat = vm / (1.0f - std::pow(beta2, t));

                    float step = m_hat / (std::sqrt(v_hat) + epsilon);
                    float cap  = std::max(kMaxStepAbs, kMaxStepRel * std::max(1.0f, std::abs(x)));
                    float scaled = alpha * std::abs(step);
                    if (scaled > cap) step *= (cap / scaled);

                    x -= alpha * step;
                }
            }
        }

        std::cout << "[LSRC] ApplyFcLateAdam: alpha=" << alpha
                  << "  ||dL/dFc_late||=" << gnorm
                  << "  t=" << t << std::endl;
    }

    float CompGraph::GetFcLateGradNorm() const {
        if (!fc_late_enabled_ || layers.empty() || !layers.back().point_cloud) return 0.f;
        const auto& pc_final = *layers.back().point_cloud;
        size_t N = std::min(pc_final.points.size(), fc_late_N_);
        double sq = 0.0;
        #pragma omp parallel for reduction(+:sq)
        for (int j = 0; j < (int)N; ++j)
            sq += pc_final.points[j].dLdF.squaredNorm();
        return (float)std::sqrt(sq);
    }

    // ── x_late: Late-Stage Position Correction ────────────────────────────

    void CompGraph::EnableXLate(size_t N) {
        if (x_late_enabled_ && x_late_N_ == N) return;  // idempotent
        x_late_enabled_ = true;
        x_late_N_       = N;
        x_late_.assign(N * 3, 0.0f);
        x_late_m_.assign(N * 3, 0.0f);
        x_late_v_.assign(N * 3, 0.0f);
        x_late_v_max_.assign(N * 3, 0.0f);
        x_late_timestep_ = 0;
        std::cout << "[x_late] Enabled for N=" << N << " particles"
                  << "  (x_late_ applied at final layer, 0 backward steps)" << std::endl;
    }

    void CompGraph::ApplyXLateAdamStep(
        float alpha, float beta1, float beta2, float epsilon, int timestep)
    {
        if (!x_late_enabled_ || x_late_N_ == 0 || layers.empty()) return;

        const auto& pc_final = *layers.back().point_cloud;
        size_t N = std::min(pc_final.points.size(), x_late_N_);

        // Gradient = dL/dx at the FINAL layer (direct render position gradient).
        double gnorm2 = 0.0;
        #pragma omp parallel for reduction(+:gnorm2)
        for (int j = 0; j < (int)N; ++j)
            gnorm2 += pc_final.points[j].dLdx.squaredNorm();
        float gnorm  = std::max((float)std::sqrt(gnorm2), 1e-12f);
        float inv_gn = 1.0f / gnorm;

        constexpr float kMaxStepAbs = 5e-3f;
        constexpr float kMaxStepRel = 5e-2f;
        int t = std::max(1, timestep);
        ++x_late_timestep_;

        #pragma omp parallel for
        for (int j = 0; j < (int)N; ++j) {
            for (int d = 0; d < 3; ++d) {
                int   idx = j * 3 + d;
                float g   = pc_final.points[j].dLdx[d] * inv_gn;
                float& m  = x_late_m_[idx];
                float& v  = x_late_v_[idx];
                float& vm = x_late_v_max_[idx];
                float& x  = x_late_[idx];

                m  = beta1 * m + (1.0f - beta1) * g;
                v  = beta2 * v + (1.0f - beta2) * g * g;
                vm = std::max(vm, v);  // AMSGrad

                float m_hat = m  / (1.0f - std::pow(beta1, t));
                float v_hat = vm / (1.0f - std::pow(beta2, t));

                float step  = m_hat / (std::sqrt(v_hat) + epsilon);
                float cap   = std::max(kMaxStepAbs, kMaxStepRel * std::max(1.0f, std::abs(x)));
                float scaled = alpha * std::abs(step);
                if (scaled > cap) step *= (cap / scaled);

                x -= alpha * step;
            }
        }

        std::cout << "[x_late] ApplyXLateAdam: alpha=" << alpha
                  << "  ||dL/dx_late||=" << gnorm
                  << "  t=" << t << std::endl;
    }

    float CompGraph::GetXLateGradNorm() const {
        if (!x_late_enabled_ || layers.empty() || !layers.back().point_cloud) return 0.f;
        const auto& pc_final = *layers.back().point_cloud;
        size_t N = std::min(pc_final.points.size(), x_late_N_);
        double sq = 0.0;
        #pragma omp parallel for reduction(+:sq)
        for (int j = 0; j < (int)N; ++j)
            sq += pc_final.points[j].dLdx.squaredNorm();
        return (float)std::sqrt(sq);
    }

}
