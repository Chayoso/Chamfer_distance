#include "pch.h"
#include "ForwardSimulation.h"
#include "Elasticity.h"
#include "Interpolation.h"
#include <cmath>
#include <algorithm>


namespace DiffMPMLib3D::SingleThreadMPM {

    void SingleParticle_op_1(MaterialPoint& mp)
    {
        mp.P = PK_FixedCorotatedElasticity(mp.F + mp.dFc, mp.lam, mp.mu);
    }

    void SingleParticle_op_1(MaterialPoint& mp, int current_episode)
    {
        mp.P = PK_FixedCorotatedElasticity(mp.F + mp.dFc, mp.lam, mp.mu);
    }

    void SingleParticle_to_grid(const MaterialPoint& mp, Grid& grid, float dt, float drag)
    {
        float dx = grid.dx;
        Vec3 relative_point = mp.x - grid.min_point;
        int bot_left_index[3];

        for (int i = 0; i < 3; i++) {
            bot_left_index[i] = (int)std::floor(relative_point[i] / grid.dx) - 1;
        }

        // This function is for a single particle; parallelism is handled by the calling P2G function.
        // Spawning threads for a 64-iteration loop is extremely inefficient.
        for (int idx = 0; idx < 64; idx++) {
            int i_offset = idx / 16;
            int j_offset = (idx % 16) / 4;
            int k_offset = idx % 4;

            int i = bot_left_index[0] + i_offset;
            int j = bot_left_index[1] + j_offset;
            int k = bot_left_index[2] + k_offset;

            if (0 <= i && i < grid.dim_x &&
                0 <= j && j < grid.dim_y &&
                0 <= k && k < grid.dim_z)
            {
                // [MODIFIED] Use GetNode helper for safe, efficient access to the 1D data array.
                GridNode& node = grid.GetNode(i, j, k);
                Vec3 dgp = node.x - mp.x;
                float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

                // Atomic operations are still necessary because the outer P2G loop is parallel.
                #pragma omp atomic
                node.m += wgp * mp.m;

                const float C0 = 3.f / (dx * dx);
                
                Vec3 delta_p = wgp * (
                    mp.m * mp.v * (1.f - dt * drag)
                    + ( -C0 * dt * mp.vol * mp.P * (mp.F + mp.dFc).transpose()
                        + mp.m * mp.C
                      ) * dgp
                );

                #pragma omp atomic
                node.p[0] += delta_p[0];
                #pragma omp atomic
                node.p[1] += delta_p[1];
                #pragma omp atomic
                node.p[2] += delta_p[2];
            }
        }
    }

    void SingleNode_op(GridNode& node, float dt, Vec3 f_ext)
    {
        if (node.m > 1e-12f) { // Use a small epsilon for floating point comparison
            node.v = node.p / node.m + dt * f_ext;
        }
    }

    void Grid_to_SingleParticle(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, const Grid& grid)
    {
        float dx = grid.dx;
        next_timestep_mp.v.setZero();
        next_timestep_mp.C.setZero();

        Vec3 relative_point = curr_timestep_mp.x - grid.min_point;
        int bot_left_index[3];
        for (int i = 0; i < 3; i++) {
            bot_left_index[i] = static_cast<int>(std::floor(relative_point[i] / dx)) - 1;
        }

        // This is a simple reduction over 64 nodes for a single particle and should be serial.
        for (int idx = 0; idx < 64; idx++) {
            int i_offset = idx / 16;
            int j_offset = (idx % 16) / 4;
            int k_offset = idx % 4;

            int i = bot_left_index[0] + i_offset;
            int j = bot_left_index[1] + j_offset;
            int k = bot_left_index[2] + k_offset;

            if (0 <= i && i < grid.dim_x &&
                0 <= j && j < grid.dim_y &&
                0 <= k && k < grid.dim_z)
            {
                const GridNode& node = grid.GetNode(i, j, k);
                Vec3 dgp = node.x - curr_timestep_mp.x;
                float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

                next_timestep_mp.v += wgp * node.v;
                next_timestep_mp.C += 3.0f / (dx * dx) * wgp * node.v * dgp.transpose();
            }
        }
    }

    void SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float dt)
    {
        next_timestep_mp.F = (Mat3::Identity() + dt * next_timestep_mp.C) * (curr_timestep_mp.F + curr_timestep_mp.dFc);
        next_timestep_mp.x = curr_timestep_mp.x + dt * next_timestep_mp.v;
    }

    void smooth_deformation_gradient(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor) {
        next_timestep_mp.F = next_timestep_mp.F * (1.0f - smoothing_factor) + curr_timestep_mp.F * smoothing_factor;
    }

    void SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor, float dt)
    {
        next_timestep_mp.F = (Mat3::Identity() + dt * next_timestep_mp.C) * (curr_timestep_mp.F + curr_timestep_mp.dFc);
        smooth_deformation_gradient(next_timestep_mp, curr_timestep_mp, smoothing_factor);
        next_timestep_mp.x = curr_timestep_mp.x + dt * next_timestep_mp.v;
    }

    void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float smoothing_factor, float dt, float drag, Vec3 f_ext, int current_episode)
    {
        // smoothing_factor is now passed in from CompGraph (respects override).
        // No internal schedule — the caller (CompGraph::ComputeForwardPass) owns the schedule.

        P_op_1(curr_point_cloud, current_episode);
        G_Reset(grid);
        P2G(curr_point_cloud, grid, dt, drag);

    #pragma omp parallel for
        for (int i = 0; i < grid.dim_x * grid.dim_y * grid.dim_z; ++i) {
            if (grid.nodes[i].m < 0.f) {
                grid.nodes[i].m = 0.f;
            }
        }

        G_op(grid, dt, f_ext);
        G2P(next_point_cloud, curr_point_cloud, grid);
        P_op_2(next_point_cloud, curr_point_cloud, smoothing_factor, dt);
    }
    
    // High-level functions that orchestrate the substeps
    void P_op_1(PointCloud& curr_point_cloud, int current_episode)
    {
    #pragma omp parallel for
        for (int p = 0; p < curr_point_cloud.points.size(); p++) {
            SingleParticle_op_1(curr_point_cloud.points[p], current_episode);
        }
    }

    void G_Reset(Grid& grid)
    {
        grid.ResetValues();
    }

    void P2G(const PointCloud& curr_point_cloud, Grid& grid, float dt, float drag)
    {
    #pragma omp parallel for
        for (int p = 0; p < curr_point_cloud.points.size(); p++) {
            SingleParticle_to_grid(curr_point_cloud.points[p], grid, dt, drag);
        }
    }

    void G_op(Grid& grid, float dt, Vec3 f_ext)
    {
    #pragma omp parallel for
        for (int idx = 0; idx < grid.nodes.size(); idx++) {
            SingleNode_op(grid.nodes[idx], dt, f_ext);
        }
    }

    void G2P(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, Grid& grid)
    {
    #pragma omp parallel for 
        for (int p = 0; p < curr_point_cloud.points.size(); p++) {
            Grid_to_SingleParticle(next_point_cloud.points[p], curr_point_cloud.points[p], grid);
        }
    }

    void P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float smoothing_factor, float dt)
    {
    #pragma omp parallel for
        for (int p = 0; p < next_point_cloud.points.size(); p++) {
            SingleParticle_op_2(next_point_cloud.points[p], curr_point_cloud.points[p], smoothing_factor, dt);
        }
    }
    
    /* 06/24 -- reduction */
    void CalculatePointCloudVolumes(PointCloud& curr_point_cloud, Grid& grid)
    {
        float dx = grid.dx;
        P2G(curr_point_cloud, grid, 0.f, 0.f);

    #pragma omp parallel for //schedule(guided, 24) 
        for (int p = 0; p < curr_point_cloud.points.size(); p++) {
            MaterialPoint& mp = curr_point_cloud.points[p];

            std::vector<std::array<int, 3>> indices;
            auto nodes = grid.QueryPoint_CubicBSpline(mp.x, &indices);

            float mass_from_grid = 0.f;

            for (int i = 0; i < nodes.size(); i++) {
                GridNode& node = nodes[i];

                Vec3 xg = node.x;
                Vec3 xp = mp.x;
                Vec3 dgp = xg - xp;
                float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

    #pragma omp atomic
                mass_from_grid += wgp * node.m;
            }
            // Unit-correct volume estimation:
            // mass_from_grid = Σ_i w_ip * m_i   [mass]
            // local density  rho_p ~= mass_from_grid / dx^3
            // rest volume    V_p0 = m_p / rho_p = m_p * dx^3 / mass_from_grid
            float cell_volume = dx * dx * dx;
            float rho_p = mass_from_grid / std::max(cell_volume, 1e-12f);
            if (rho_p > std::numeric_limits<float>::epsilon()) {
                mp.vol = mp.m / rho_p; // == mp.m * dx^3 / mass_from_grid
            } else {
                mp.vol = 0.f;
            }
        }
    }
    // Other overloads are omitted for brevity but should be updated similarly.
}