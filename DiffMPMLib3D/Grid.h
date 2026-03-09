// #pragma once
// #include "GridNode.h"
// #include <functional>


// namespace DiffMPMLib3D {
// 	struct Grid
// 	{
// 		/*
// 		* Grid stores a 3D uniform grid of nodal interpolation points.
// 		* The convention is that i, j, k corresponds the the positive x, y, z axes.
// 		*/

// 		Grid(int _dim_x, int _dim_y, int dim_z, float _dx, Vec3 _min_point);
// 		Grid(const Grid& grid);

// 		//std::vector<std::reference_wrapper<GridNode>> QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr);
// 		std::vector<std::reference_wrapper<DiffMPMLib3D::GridNode>> QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr);
// 		std::vector<std::reference_wrapper<const GridNode>> QueryPointConst_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr) const;
// 		//std::vector<DiffMPMLib3D::GridNode*> Grid::QueryPoint_CubicBSpline_with_kernel(Vec3 point, std::vector<std::array<int, 3>>* indices);

// 		std::vector<GridNode> flatten() const {
// 			std::vector<GridNode> flat_nodes(dim_x * dim_y * dim_z);
// 			for (int idx = 0; idx < dim_x * dim_y * dim_z; ++idx) {
// 				int x = idx / (dim_y * dim_z);
// 				int y = (idx / dim_z) % dim_y;
// 				int z = idx % dim_z;
// 				flat_nodes[idx] = nodes[x][y][z];
// 			}
// 			return flat_nodes;
// 		}

// 		void unflatten(const std::vector<GridNode>& flat_nodes) {
// 			nodes = std::vector<std::vector<std::vector<GridNode>>>(dim_x, std::vector<std::vector<GridNode>>(dim_y, std::vector<GridNode>(dim_z)));
// 			for (int idx = 0; idx < flat_nodes.size(); ++idx) {
// 				int x = idx / (dim_y * dim_z);
// 				int y = (idx / dim_z) % dim_y;
// 				int z = idx % dim_z;
// 				nodes[x][y][z] = flat_nodes[idx];
// 			}
// 		}

// 		std::vector<Vec3> GetNodePositions() const;
// 		std::vector<float> GetNodeMasses() const;
// 		std::vector<Vec3> GetNodeVelocities() const;
// 		void GetMassSDF(Eigen::MatrixXf& GV, Eigen::VectorXf& Gf) const;

// 		void ResetGradients();
// 		void ResetValues();

// 		int dim_x;
// 		int dim_y;
// 		int dim_z;
// 		float dx;
// 		Vec3 min_point;
// 		std::vector<std::vector<std::vector<GridNode>>> nodes;
// 	};
// }

#pragma once
#include "GridNode.h"
#include <functional>
#include <vector>

namespace DiffMPMLib3D {
    struct Grid
    {
        /*
        * Grid stores a 3D uniform grid of nodal interpolation points.
        * The convention is that i, j, k corresponds to the positive x, y, z axes.
        */

        Grid(int _dim_x, int _dim_y, int _dim_z, float _dx, Vec3 _min_point);
        Grid(const Grid& grid);

        std::vector<std::reference_wrapper<GridNode>> QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr);
        std::vector<std::reference_wrapper<const GridNode>> QueryPointConst_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr) const;

        std::vector<Vec3> GetNodePositions() const;
        std::vector<float> GetNodeMasses() const;
        std::vector<Vec3> GetNodeVelocities() const;
        void GetMassSDF(Eigen::MatrixXf& GV, Eigen::VectorXf& Gf) const;

        void ResetGradients();
        void ResetValues();

        // Helper function for accessing nodes using 3D indices on the flattened 1D vector.
        // This is crucial for maintaining readable code while gaining performance.
        inline GridNode& GetNode(int i, int j, int k) {
            return nodes[i * dim_y * dim_z + j * dim_z + k];
        }
        inline const GridNode& GetNode(int i, int j, int k) const {
            return nodes[i * dim_y * dim_z + j * dim_z + k];
        }

        int dim_x;
        int dim_y;
        int dim_z;
        float dx;
        Vec3 min_point;

        // [OPTIMIZATION] Changed from a 3D vector of vectors to a single 1D vector.
        // This ensures contiguous memory allocation, which is critical for CPU cache performance
        // and allows for better vectorization by the compiler. This is the most
        // significant performance optimization for the grid structure.
        std::vector<GridNode> nodes;
    };
}