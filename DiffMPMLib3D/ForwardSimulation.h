#pragma once

#include "PointCloud.h"
#include "Grid.h"

namespace DiffMPMLib3D {
    namespace SingleThreadMPM {
        // Particle operations (stress computation)
        void SingleParticle_op_1(MaterialPoint& mp);
        void SingleParticle_op_1(MaterialPoint& mp, int current_episode);

        // Particle to Grid transfer (P2G)
        void SingleParticle_to_grid(const MaterialPoint& mp, Grid& grid, float dt, float drag);

        // Grid operations (velocity update)
        void SingleNode_op(GridNode& node, float dt, Vec3 f_ext);

        // Grid to Particle transfer (G2P)
        void Grid_to_SingleParticle(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, const Grid& grid);
        
        // Particle operations (deformation and position update)
        void SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float dt);
        void SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor, float dt);
        void smooth_deformation_gradient(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor);

        // --- Main Timestep Function ---
        void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float dt, float drag, Vec3 f_ext);
        void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float smoothing_factor, float dt, float drag, Vec3 f_ext);
        void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float smoothing_factor, float dt, float drag, Vec3 f_ext, int current_episode);

        // --- High-level MPM Steps (called by ForwardTimeStep) ---
        void P_op_1(PointCloud& curr_point_cloud);
        void P_op_1(PointCloud& curr_point_cloud, int current_episode);
        void G_Reset(Grid& grid);
        void P2G(const PointCloud& curr_point_cloud, Grid& grid, float dt, float drag);
        void G_op(Grid& grid, float dt, Vec3 f_ext);
        void G_op(Grid& grid, float dt, Vec3 gravity_point, float gravity_mag);
        void G2P(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, Grid& grid);
        void P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float dt);
        void P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float smoothing_factor, float dt);

        // --- Utility Functions ---
        void CalculatePointCloudVolumes(PointCloud& curr_point_cloud, Grid& grid);
        void P2G_Mass(const std::vector<Vec3> points, Grid& grid, float mp_m);
        void G2G_Mass(Grid& grid_1, Grid& grid_2);

        // --- Material and Contact Properties ---
        struct MaterialProperties {
            float lam, mu;
        };

        const std::vector<MaterialProperties> MATERIALS = {
            {38888.8f, 58333.3f},  // 0: refer
            {28846.1f, 19231.7f},  // 1: soft
            {186207.9f, 48276.8f}, // 2: bouncy 
            {155556.5f, 233333.3f},// 3: stiff
        };

        struct ContactMaterial {
            float restitution;   // Coefficient of restitution
            float softness;      // Penalty spring stiffness
            float sigmoid_k;     // Smoothing: penetration -> weight
        };
        
        static const ContactMaterial MAT_REFERENCE{ 0.f, 0.f, 15.f };
    }
}