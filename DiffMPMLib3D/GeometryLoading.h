#pragma once
#include "pch.h"
#include "PointCloud.h"
#include <igl/readOBJ.h>
#include <cfloat> 

namespace DiffMPMLib3D {
	namespace GeometryLoading
	{

	std::vector<Vec3> GeneratePointCloudFromWatertightTriangleMesh(
		const Eigen::MatrixXf& V,
		const Eigen::MatrixXi& F,
		Vec3 min_point,
		Vec3 max_point,
		float sampling_dx,
		bool apply_jitter = true
	);

	bool LoadMPMPointCloudFromObj(
		std::string obj_path,
		std::shared_ptr<PointCloud>& mpm_point_cloud,
		float point_dx,
		float density,
		float lam,
		float mu,
		bool apply_jitter = true
	);
	}
}