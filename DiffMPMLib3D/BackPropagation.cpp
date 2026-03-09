#include "pch.h"
#include "BackPropagation.h"
#include "Elasticity.h"
#include "Interpolation.h"

namespace DiffMPMLib3D {

    void Back_Timestep(CompGraphLayer& layer_nplus1, CompGraphLayer& layer_n, float drag, float dt, float smoothing_factor)
    {
        const PointCloud& pc = *layer_nplus1.point_cloud;
        PointCloud& pc_prev = *layer_n.point_cloud;
        Grid& grid = *layer_n.grid;
        float dx = grid.dx;

        // OpenMP requires signed integral types for loop variables.
        // Casting from size_t to int to acknowledge the conversion and silence warnings.
        int num_points = (int)pc.points.size();

        // Back prop P_op_2: Propagate gradients from the next state (n+1) to the current state's particles (n).
        // This reverses the particle state update step (F, v, C).
    #pragma omp parallel for
        for (int p = 0; p < num_points; p++) {
            const MaterialPoint& mp = pc.points[p];
            MaterialPoint& mp_prev = pc_prev.points[p];

            mp_prev.dLdF = (Mat3::Identity() + dt * mp.C).transpose() * mp.dLdF;
            mp_prev.dLdF = mp_prev.dLdF * (1.0f - smoothing_factor) + mp.dLdF * smoothing_factor;

            mp_prev.dLdv_next = dt * mp.dLdx + mp.dLdv;
            mp_prev.dLdC_next = dt * mp.dLdF * (mp_prev.F + mp_prev.dFc).transpose() + mp.dLdC;
        }

        // Back prop G2P: Propagate gradients from particle states back to grid node velocities.
        // This reverses the Grid-to-Particle (G2P) velocity transfer.
    #pragma omp parallel for
        for (int p = 0; p < num_points; p++) {
            const MaterialPoint& mp_prev = pc_prev.points[p];
            auto nodes = grid.QueryPoint_CubicBSpline(mp_prev.x);

            for (size_t i = 0; i < nodes.size(); i++) {
                GridNode& node = nodes[i];

                Vec3 dgp = node.x - mp_prev.x;
                float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

                Vec3 dLdv_update = mp_prev.dLdv_next * wgp + 3.f / (dx * dx) * wgp * mp_prev.dLdC_next * dgp;

                // [OK] Use atomic operations instead of critical section (10-100x faster!)
                // Each component can be updated independently with atomic add
                #pragma omp atomic
                node.dLdv[0] += dLdv_update[0];
                #pragma omp atomic
                node.dLdv[1] += dLdv_update[1];
                #pragma omp atomic
                node.dLdv[2] += dLdv_update[2];
            }
        }

        // Back prop G_op: Propagate gradients from grid velocities back to grid momentum and mass.
        // This reverses the grid velocity update (v = p/m).
    #pragma omp parallel for 
        for (int idx = 0; idx < grid.dim_x * grid.dim_y * grid.dim_z; idx++) {
            int i = idx / (grid.dim_y * grid.dim_z);
            int j = (idx / grid.dim_z) % grid.dim_y;
            int k = idx % grid.dim_z;

            // [MODIFIED] Use GetNode helper for safe, efficient access to the flattened 1D grid data.
            GridNode& node = grid.GetNode(i, j, k);

            if (fabs(node.m) > 1e-12f) {
                node.dLdp = node.dLdv / node.m;
                node.dLdm = -1.f / node.m * node.v.dot(node.dLdv);
            }
        }

        // Back prop P2G: Propagate gradients from grid states back to particle states (F, v, x, C, P).
        // This reverses the Particle-to-Grid (P2G) transfer.
    #pragma omp parallel for 
        for (int p = 0; p < num_points; p++) {
            const MaterialPoint& mp = pc.points[p];
            MaterialPoint& mp_prev = pc_prev.points[p];

            const Vec3& xp = mp_prev.x;
            const Mat3 F_total_transpose = (mp_prev.F + mp_prev.dFc).transpose();

            auto nodes = grid.QueryPoint_CubicBSpline(xp);

            // This inner loop is serial, which is correct. The outer loop parallelizes over particles.
            for (size_t i = 0; i < nodes.size(); i++) {
                GridNode& node = nodes[i];
                const Vec3& xg = node.x;
                Vec3 dgp = xg - xp;

                // Use constants and precompute b-spline values for clarity.
                const float C0 = 3.f / (dx * dx);
                
                Vec3 dgp_div_dx = dgp / dx;
                Vec3 bspline_vals(CubicBSpline(dgp_div_dx[0]), CubicBSpline(dgp_div_dx[1]), CubicBSpline(dgp_div_dx[2]));
                Vec3 bspline_slopes(CubicBSplineSlope(dgp_div_dx[0]), CubicBSplineSlope(dgp_div_dx[1]), CubicBSplineSlope(dgp_div_dx[2]));

                float wgp = bspline_vals[0] * bspline_vals[1] * bspline_vals[2];
                Vec3 wgpGrad = -1.f / dx * Vec3(
                    bspline_slopes[0] * bspline_vals[1]   * bspline_vals[2],
                    bspline_vals[0]   * bspline_slopes[1] * bspline_vals[2],
                    bspline_vals[0]   * bspline_vals[1]   * bspline_slopes[2]
                );

                Mat3 G = -C0 * dt * mp_prev.vol * mp_prev.P * F_total_transpose + mp_prev.m * mp_prev.C;

                // Accumulate gradients for particle properties
                mp_prev.dLdP -= wgp * C0 * dt * mp_prev.vol * node.dLdp * (F_total_transpose * dgp).transpose();
                mp_prev.dLdF -= wgp * C0 * dt * mp_prev.vol * dgp * (mp_prev.P.transpose() * node.dLdp).transpose();
                mp_prev.dLdC += wgp * mp_prev.m * node.dLdp * dgp.transpose();

                // Decompose complex dLdx update into meaningful parts.
                Vec3 dLdx_from_mass     = mp_prev.m * node.dLdm * wgpGrad;
                Mat3 momentum_term      = wgpGrad * (mp_prev.m * mp_prev.v + G * dgp).transpose() - wgp * G.transpose();
                Vec3 dLdx_from_momentum = momentum_term * node.dLdp;
                Vec3 dLdx_from_v_next   = wgpGrad * node.v.transpose() * mp_prev.dLdv_next;
                
                Vec3 temp = InnerProduct(mp_prev.dLdC_next, node.v * dgp.transpose()) * wgpGrad - wgp * mp_prev.dLdC_next.transpose() * node.v;
                Vec3 dLdx_from_C_next   = C0 * temp;

                mp_prev.dLdx += dLdx_from_mass + dLdx_from_momentum + dLdx_from_v_next + dLdx_from_C_next;

                mp_prev.dLdv += wgp * mp_prev.m * (1.f - dt * drag) * node.dLdp;
            }

            // Back prop P_op_1: Final gradient accumulation for dLdF and dLdx within the particle loop.
            // mp_prev.dLdF += (Mat3::Identity() + dt * pc.points[p].C).transpose() * pc.points[p].dLdF;
            mp_prev.dLdF += d2_FCE_psi_dF2_mult_by_dF(mp_prev.F + mp_prev.dFc, mp_prev.lam, mp_prev.mu, mp_prev.dLdP);
            mp_prev.dLdx += pc.points[p].dLdx;
        }
    }

} // namespace DiffMPMLib3D