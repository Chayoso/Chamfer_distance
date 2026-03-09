#pragma once
#include "pch.h"

#include "MaterialPoint.h"
#include <omp.h>
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"

namespace DiffMPMLib3D {
	struct PointCloud
	{
		void ResetGradients();
		float Compute_dLdF_Norm();
		void Descend_dLdF(float alpha, float gradient_norm);
		void Descend_Adam(float alpha, float gradient_norm, float beta1, float beta2, float epsilon, int timestep);
		std::vector<int> generateMinibatchIndices(int total_size, int minibatch_size);

		void Descend_AMSGrad(
			float alpha, float gradient_norm,
			float beta1, float beta2, float epsilon, int timestep);

		void Descend_GradientDescent(
			float alpha,
			float gradient_norm);

	// Independent render Adam: uses separate momentum_r / vector_r state.
	// dLdF must contain ONLY the render gradient (set by ComputeRenderBackwardOnly).
	void Descend_Adam_Render(
		float alpha, float gradient_norm,
		float beta1, float beta2, float epsilon, int timestep);

		inline float SignFloat(float x) {
			return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
		}

	void Descend_Lion(
		float alpha,   
		float beta,  
		float gradient_norm);

		void RemovePoint(size_t point_index);

		float Compute_dLdF_Norm_Stochastic(const std::vector<int>& minibatch_indices);
		void Descend_Adam_Stochastic(float alpha, float gradient_norm, float beta1, float beta2, float epsilon, int timestep, const std::vector<int>& minibatch_indices);

		bool ReadFromOBJ(std::string obj_path, float point_mass);
		void WriteToOBJ(std::string obj_path); // just writes vertex positions
		void WriteMassVelocityDefgradsToFile(std::string file_path);
		/*void WriteEntirePointCloudToFile(std::string file_path);
		bool ReadEntirePointCloudFromFile(std::string file_path);*/
		void WriteEntirePointCloudToBinaryFile(std::string file_path);
		bool ReadEntirePointCloudFromBinaryFile(std::string file_path);

		bool IsEqualToOtherPointCloud(const PointCloud& other_pc);

		std::vector<Vec3> GetPointPositions() const;
		std::vector<float> GetPointMasses() const;
		std::vector<float> GetPointVolumes() const;
		std::vector<Mat3> GetPointDefGrads() const;
		std::vector<float> GetPointDeterminants() const;
		std::vector<float> GetPointElasticEnergies() const;

		// [FIX] NEW: Gradient getters for PCGrad
		std::vector<Mat3> GetPointDefGradGradients() const;  // dLdF
		std::vector<Vec3> GetPointPositionGradients() const; // dLdx

		// J-min barrier: adds dL_barrier/dF to dLdF for particles where det(F+dFc) < J_target
		// Barrier: L = (w/2) * max(0, J_target - J)^2  =>  pushes J up toward J_target
		void AddJBarrierGradient(float J_target, float w_barrier);

		std::vector<MaterialPoint> points;

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(
				CEREAL_NVP(points)
			);
		}
	};
}