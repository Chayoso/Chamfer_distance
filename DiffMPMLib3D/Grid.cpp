#include "pch.h"
#include "Grid.h"
#include <iostream>

namespace DiffMPMLib3D {

    Grid::Grid(int _dim_x, int _dim_y, int _dim_z, float _dx, Vec3 _min_point)
        : dim_x(_dim_x), dim_y(_dim_y), dim_z(_dim_z), dx(_dx), min_point(_min_point)
    {
        // [OPTIMIZATION] Allocate all nodes in a single, contiguous block of memory.
        size_t total_nodes = (size_t)dim_x * dim_y * dim_z;
        nodes.resize(total_nodes);

        // Initialize node positions in parallel.
    #pragma omp parallel for
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < dim_y; ++j) {
                for (int k = 0; k < dim_z; ++k) {
                    GetNode(i, j, k).x = min_point + Vec3((float)i, (float)j, (float)k) * dx;
                }
            }
        }

        auto total_bytes = sizeof(GridNode) * total_nodes;
        std::cout << "Generated grid of size: " << (total_bytes >> 10) << " KB (" << total_nodes << " nodes)" << std::endl;
    }

    Grid::Grid(const Grid& other)
    {
        // Copy constructor, now simpler with a single vector.
        dim_x = other.dim_x;
        dim_y = other.dim_y;
        dim_z = other.dim_z;
        dx = other.dx;
        min_point = other.min_point;
        nodes = other.nodes;
    }

    std::vector<std::reference_wrapper<GridNode>> Grid::QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices)
    {
        /*
        * Returns all nodes which are within the interpolation range of the point position.
        * NOTE: This function still performs a dynamic allocation for the return vector, which can be a bottleneck.
        * A further optimization would be to pass a pre-allocated array (e.g., std::array<GridNode*, 64>)
        * to be filled by this function.
        */

        std::vector<std::reference_wrapper<GridNode>> ret;
        ret.reserve(64); // 4*4*4, reserve memory to avoid multiple reallocations.

        if (indices != nullptr) {
            indices->clear();
            indices->reserve(64);
        }

        Vec3 relative_point = point - min_point;
        int bot_left_index[3];
        for (int i = 0; i < 3; i++) {
            bot_left_index[i] = (int)std::floor(relative_point[i] / dx) - 1;
        }

        // [BUG FIX] Removed incorrect #pragma omp parallel for.
        // 'push_back' on std::vector is not thread-safe and would cause a race condition.
        // This loop is small (64 iterations) and should be executed serially for each point query.
        for (int i = 0; i <= 3; i++) {
            for (int j = 0; j <= 3; j++) {
                for (int k = 0; k <= 3; k++) {
                    int final_i = bot_left_index[0] + i;
                    int final_j = bot_left_index[1] + j;
                    int final_k = bot_left_index[2] + k;

                    // Check if the node index is within the grid bounds.
                    if (0 <= final_i && final_i < dim_x &&
                        0 <= final_j && final_j < dim_y &&
                        0 <= final_k && final_k < dim_z)
                    {
                        // [MODIFIED] Use the GetNode helper for 3D indexing.
                        ret.push_back(std::ref(GetNode(final_i, final_j, final_k)));
                        if (indices != nullptr) {
                            indices->push_back({ final_i, final_j, final_k });
                        }
                    }
                }
            }
        }
        return ret;
    }

    std::vector<std::reference_wrapper<const GridNode>> Grid::QueryPointConst_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices) const
    {
        // Const version of the query function.
        std::vector<std::reference_wrapper<const GridNode>> ret;
        ret.reserve(64);

        if (indices != nullptr) {
            indices->clear();
            indices->reserve(64);
        }
        
        Vec3 relative_point = point - min_point;
        int bot_left_index[3];
        for (int i = 0; i < 3; i++) {
            bot_left_index[i] = (int)std::floor(relative_point[i] / dx) - 1;
        }

        // [BUG FIX] Removed incorrect #pragma omp parallel for for the same reason as above.
        for (int i = 0; i <= 3; i++) {
            for (int j = 0; j <= 3; j++) {
                for (int k = 0; k <= 3; k++) {
                    int final_i = bot_left_index[0] + i;
                    int final_j = bot_left_index[1] + j;
                    int final_k = bot_left_index[2] + k;

                    if (0 <= final_i && final_i < dim_x &&
                        0 <= final_j && final_j < dim_y &&
                        0 <= final_k && final_k < dim_z)
                    {
                        ret.push_back(std::cref(GetNode(final_i, final_j, final_k)));
                        if (indices != nullptr) {
                            indices->push_back({ final_i, final_j, final_k });
                        }
                    }
                }
            }
        }
        return ret;
    }

    std::vector<Vec3> Grid::GetNodePositions() const
    {
        size_t total_nodes = (size_t)dim_x * dim_y * dim_z;
        std::vector<Vec3> ret(total_nodes);

    #pragma omp parallel for
        for (int idx = 0; idx < total_nodes; idx++) {
            ret[idx] = nodes[idx].x;
        }
        return ret;
    }

    std::vector<float> Grid::GetNodeMasses() const
    {
        size_t total_nodes = (size_t)dim_x * dim_y * dim_z;
        std::vector<float> ret(total_nodes);

    #pragma omp parallel for
        for (int idx = 0; idx < total_nodes; idx++) {
            ret[idx] = nodes[idx].m;
        }
        return ret;
    }

    std::vector<Vec3> Grid::GetNodeVelocities() const
    {
        size_t total_nodes = (size_t)dim_x * dim_y * dim_z;
        std::vector<Vec3> ret(total_nodes);

    #pragma omp parallel for
        for (int idx = 0; idx < total_nodes; idx++) {
            ret[idx] = nodes[idx].v;
        }
        return ret;
    }

    void Grid::GetMassSDF(Eigen::MatrixXf& GV, Eigen::VectorXf& Gf) const
    {
        size_t total_nodes = (size_t)dim_x * dim_y * dim_z;
        GV.resize(total_nodes, 3);
        Gf.resize(total_nodes);

        // This loop is hard to parallelize safely due to the single 'index++'.
        // It can be parallelized with care, but for simplicity, we keep it serial.
        int index = 0;
        for (int i = 0; i < dim_x; i++) {
            for (int j = 0; j < dim_y; j++) {
                for (int k = 0; k < dim_z; k++) {
                    GV.row(index) = GetNode(i, j, k).x;
                    Gf(index) = GetNode(i, j, k).m;
                    index++;
                }
            }
        }
    }

    void Grid::ResetGradients()
    {
    #pragma omp parallel for
        for (int idx = 0; idx < nodes.size(); idx++) {
            nodes[idx].ResetGradients();
        }
    }

    void Grid::ResetValues()
    {	
    #pragma omp parallel for 
        for (int idx = 0; idx < nodes.size(); idx++) {
            nodes[idx].m = 0.0;
            nodes[idx].v.setZero();
            nodes[idx].p.setZero();
        }
    }
}