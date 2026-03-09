#pragma once
#include "pch.h"

namespace DiffMPMLib3D
{
	void SphereUnionMarchingCubesSurfaceFromPointCloud(const std::vector<Vec3>& _points,
		float radius, float grid_dx, float iso_mass, 
		int blur_iterations,
		Vec3 grid_min_point, Vec3 grid_max_point,
		Eigen::MatrixXf& mcV, // mesh vertices
		Eigen::MatrixXi& mcF // mesh faces
	);
}