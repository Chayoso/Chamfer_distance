#include "pch.h"
#include "PointCloud.h"
#include "Elasticity.h"
#include <fstream>
#include <random>
#include <iostream>
#include <chrono>

template <typename ForwardIt, typename T>
void custom_iota(ForwardIt first, ForwardIt last, T value) {
	while (first != last) {
		*first++ = value++;
	}
}

void DiffMPMLib3D::PointCloud::ResetGradients()
{
#pragma omp parallel for
	for (int p = 0; p < points.size(); p++) {
		points[p].ResetGradients();
	}
}

float DiffMPMLib3D::PointCloud::Compute_dLdF_Norm()
{
	float norm = 0.f;

#pragma omp parallel for reduction(+:norm)
	for (int i = 0; i < points.size(); ++i) {
		norm += points[i].dLdF.squaredNorm();
	}

	norm = sqrt(norm);
	return norm;
}

std::vector<int> DiffMPMLib3D::PointCloud::generateMinibatchIndices(int total_size, int minibatch_size)
{
	std::vector<int> indices(total_size);
	custom_iota(indices.begin(), indices.end(), 0); 
	std::random_device rd;
	std::mt19937 g(rd());
	g.seed(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())); // set seed
	std::shuffle(indices.begin(), indices.end(), g);
	indices.resize(minibatch_size);

	return indices;
}


float DiffMPMLib3D::PointCloud::Compute_dLdF_Norm_Stochastic(const std::vector<int>& minibatch_indices)
{
	float norm = 0;

#pragma omp parallel for
	for (int idx = 0; idx < minibatch_indices.size(); ++idx) {
		int i = minibatch_indices[idx];
		norm += points[i].dLdF.squaredNorm();
	}

	norm = std::sqrt(norm);
	return norm;
}

void DiffMPMLib3D::PointCloud::Descend_Adam_Stochastic(float alpha, float gradient_norm, float beta1,
	float beta2, float epsilon, int timestep, const std::vector<int>& minibatch_indices)
{
#pragma omp parallel for
	for (int idx = 0; idx < minibatch_indices.size(); ++idx) {
		int i = minibatch_indices[idx];
		Mat3 grad = points[i].dLdF / gradient_norm;

		points[i].momentum = beta1 * points[i].momentum + (1.0f - beta1) * grad;
		points[i].vector = beta2 * points[i].vector + (1.0f - beta2) * (grad.array().square().matrix());

		Mat3 m_hat = points[i].momentum / (1.0f - std::pow(beta1, timestep + 1));
		Mat3 v_hat = points[i].vector / (1.0f - std::pow(beta2, timestep + 1));

		points[i].dFc -= alpha * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
	}
}

void DiffMPMLib3D::PointCloud::Descend_Adam(
	float alpha, float gradient_norm,
	float beta1, float beta2, float epsilon, int timestep)
{
	// ---- Stabilization parameters ----
	constexpr bool  kUseAMSGrad = true;
	constexpr float kWeightDecayAdamW = 0.0f;
	constexpr float kVFloor = 1e-8f;
	constexpr float kMaxStepAbs = 5e-3f;
	constexpr float kMaxStepRel = 5e-2f;
	constexpr float kEMA_K = 0.20f;
	constexpr float kSmallStepAbs = 1e-4f;
	constexpr float kSmallStepRel = 1e-3f;

	const float inv_gn = 1.0f / std::max(gradient_norm, 1e-12f);
	const int   t = std::max(1, timestep);

#pragma omp parallel for
	for (int i = 0; i < (int)points.size(); ++i)
	{
		Mat3 grad = points[i].dLdF * inv_gn;

		Mat3& m = points[i].momentum;
		Mat3& v = points[i].vector;
		Mat3& vmax = points[i].vector_max;   // for AMSGrad
		Mat3& x = points[i].dFc;

		m = beta1 * m + (1.0f - beta1) * grad;
		v = beta2 * v + (1.0f - beta2) * (grad.array().square().matrix());

		Mat3 m_hat = m / (1.0f - std::pow(beta1, t));
		Mat3 v_hat = v / (1.0f - std::pow(beta2, t));

		if (kUseAMSGrad) vmax = vmax.cwiseMax(v_hat);
		const Mat3& denom = kUseAMSGrad ? vmax : v_hat;

		// √v + floor + epsilon  (no lambda)
		Mat3 denom_sqrt = denom.cwiseSqrt().cwiseMax(kVFloor);

		Mat3 step_core = (m_hat.array() / (denom_sqrt.array() + epsilon)).matrix();

		// per-step clipping (abs + relative)
		float step_norm = step_core.norm();
		float x_norm = x.norm();
		float rel_cap = kMaxStepRel * std::max(1.0f, x_norm);
		float cap = std::max(kMaxStepAbs, rel_cap);
		Mat3  step = step_core;
		if (step_norm > 0.f) {
			float scaled = alpha * step_norm;
			if (scaled > cap) step *= (cap / scaled);
		}

		Mat3 x_old = x;

		if (kWeightDecayAdamW > 0.f) x *= (1.f - alpha * kWeightDecayAdamW);
		x -= alpha * step;

		// weak EMA only when near-stationary
		float small_cap = std::max(kSmallStepAbs, kSmallStepRel * std::max(1.0f, x_old.norm()));
		bool  near_stationary = (alpha * step_core.norm() <= small_cap);
		if (kEMA_K > 0.f && near_stationary) {
			x = x_old + kEMA_K * (x - x_old);
		}
	}
}

void DiffMPMLib3D::PointCloud::Descend_Adam_Render(
	float alpha, float gradient_norm,
	float beta1, float beta2, float epsilon, int timestep)
{
	// Same AMSGrad logic as Descend_Adam, but uses momentum_r / vector_r
	// so render updates never corrupt the physics Adam state.
	constexpr bool  kUseAMSGrad = true;
	constexpr float kVFloor = 1e-8f;
	constexpr float kMaxStepAbs = 5e-3f;
	constexpr float kMaxStepRel = 5e-2f;

	const float inv_gn = 1.0f / std::max(gradient_norm, 1e-12f);
	const int   t = std::max(1, timestep);

#pragma omp parallel for
	for (int i = 0; i < (int)points.size(); ++i)
	{
		Mat3 grad = points[i].dLdF * inv_gn;

		Mat3& m    = points[i].momentum_r;
		Mat3& v    = points[i].vector_r;
		Mat3& x    = points[i].dFc;

		m = beta1 * m + (1.0f - beta1) * grad;
		v = beta2 * v + (1.0f - beta2) * (grad.array().square().matrix());

		Mat3 m_hat = m / (1.0f - std::pow(beta1, t));
		Mat3 v_hat = v / (1.0f - std::pow(beta2, t));

		Mat3 denom_sqrt = v_hat.cwiseSqrt().cwiseMax(kVFloor);
		Mat3 step_core  = (m_hat.array() / (denom_sqrt.array() + epsilon)).matrix();

		float step_norm = step_core.norm();
		float rel_cap   = kMaxStepRel * std::max(1.0f, x.norm());
		float cap       = std::max(kMaxStepAbs, rel_cap);
		Mat3  step      = step_core;
		if (step_norm > 0.f) {
			float scaled = alpha * step_norm;
			if (scaled > cap) step *= (cap / scaled);
		}

		x -= alpha * step;
	}
}

void DiffMPMLib3D::PointCloud::Descend_AMSGrad(
	float alpha, float gradient_norm,
	float beta1, float beta2, float epsilon, int timestep)
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		// 1) Gradient (scaled by gradient_norm if needed)
		Mat3 grad = points[i].dLdF / gradient_norm;

		// 2) m, v update (Adam)
		points[i].momentum = beta1 * points[i].momentum + (1.f - beta1) * grad;  // m_t
		points[i].vector = beta2 * points[i].vector
			+ (1.f - beta2) * (grad.array().square().matrix());  // v_t

		// 3) bias correction
		Mat3 m_hat = points[i].momentum / (1.f - std::pow(beta1, timestep));
		Mat3 v_hat = points[i].vector / (1.f - std::pow(beta2, timestep));

		// 4) AMSGrad: v_max update
		points[i].vector_max = points[i].vector_max.cwiseMax(v_hat);

		// 5) parameter(dFc) update
		// correct example
		Mat3 update = (
			m_hat.array()
			/ (points[i].vector_max.array().sqrt() + epsilon)
			).matrix();

		points[i].dFc -= alpha * update;
	}
}

void DiffMPMLib3D::PointCloud::Descend_Lion(
	float alpha,   // learning rate
	float beta,    // momentum coefficient
	float gradient_norm)
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		// 1) get grad (can scale with gradient_norm if desired)
		//    Lion paper itself doesn't use gradient_norm, but can apply if needed.
		Eigen::Matrix3f grad = points[i].dLdF / gradient_norm;

		// 2) calculate sign(grad)
		Eigen::Matrix3f signGrad = grad.unaryExpr(
			[this](float x) {
				return SignFloat(x);
			}
		);

		// 3) Lion momentum update
		//    m_t = beta * m_{t-1} + (1-beta) * sign(grad)
		points[i].momentum = beta * points[i].momentum
			+ (1.f - beta) * signGrad;

		// 4) parameter(dFc) update
		//    theta <- theta - alpha * m_t
		points[i].dFc -= alpha * points[i].momentum;
	}
}

void DiffMPMLib3D::PointCloud::Descend_GradientDescent(
	float alpha,         // learning rate (step size)
	float gradient_norm) // scale with gradient_norm if needed
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		// 1) get gradient (dLdF), scaling
		//    (if gradient_norm is 1, just use raw gradient)
		Eigen::Matrix3f grad = points[i].dLdF / gradient_norm;

		// 2) GD update: theta <- theta - alpha * grad
		points[i].dFc -= alpha * grad;
	}
}


void DiffMPMLib3D::PointCloud::Descend_dLdF(float alpha, float gradient_norm)
{
	float momentum_factor = 0.9f;

	float min_clip_value = 1e-3f;
	float max_clip_value = 1.0f;

#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		Mat3 dir = points[i].dLdF / gradient_norm;

		// Clipping Gradient;
		dir = dir.unaryExpr([=](float val) {
			if (val > max_clip_value) return max_clip_value;
			if (val < -max_clip_value) return -max_clip_value;
			if (std::abs(val) < min_clip_value) return (val < 0 ? -min_clip_value : min_clip_value);
			return val;
			});
	
		points[i].momentum = momentum_factor * points[i].momentum + alpha * dir;
		points[i].dFc -= points[i].momentum;
	}
}

void DiffMPMLib3D::PointCloud::RemovePoint(size_t point_index)
{
	points.erase(points.begin() + point_index);
}

bool DiffMPMLib3D::PointCloud::ReadFromOBJ(std::string obj_path, float point_mass)
{
	// only material parameter assigned will be mass, since we only use this for rendering data (not simulation)

	points.resize(0);


	std::ifstream ifs;
	ifs.open(obj_path);
	if (ifs.good()) {
		std::string junk;
		float x, y, z;

		while (ifs >> junk >> x >> y >> z)
		{
			Eigen::Vector3f pos(x, y, z);
			MaterialPoint point;
			point.x = pos;
			point.m = point_mass;
			points.emplace_back(point);
		}
		ifs.close();
		return true;
	}
	else {
		std::cout << "couldn't open file: " << obj_path << std::endl;
		ifs.close();
		return false;
	}
}

void DiffMPMLib3D::PointCloud::WriteToOBJ(std::string obj_path)
{
	std::ofstream ofs(obj_path, std::ios::out);

	if (!ofs) {
		std::cerr << "Could not open file for writing: " << obj_path << std::endl;
		return;
	}

	for (size_t v = 0; v < points.size(); v++) {
		ofs << "v " << points[v].x[0] << " " << points[v].x[1] << " " << points[v].x[2] << "\n";
	}

	ofs.close();
}

void DiffMPMLib3D::PointCloud::WriteMassVelocityDefgradsToFile(std::string file_path)
{
	std::ofstream ofs;
	ofs.open(file_path);

	if (ofs.good()) {
		for (size_t v = 0; v < points.size(); v++) {
			ofs << "x " << points[v].x[0] << " " << points[v].x[1] << " " << points[v].x[2] << "\n";
			ofs << "v " << points[v].v[0] << " " << points[v].v[1] << " " << points[v].v[2] << "\n";


			// I don't plan on doing any visualizations that actually use the 3x3 deformation gradients,
			// just maybe their elastic energy and their magnitudes?

			ofs << "F";
			for (size_t i = 0; i < 9; i++) {
				ofs << " " << points[v].F(i);
			}
			ofs << "\n";
			ofs << "dFc";
			for (size_t i = 0; i < 9; i++) {
				ofs << " " << points[v].dFc(i);
			}
			ofs << "\n";
			ofs << "dLdF";
			for (size_t i = 0; i < 9; i++) {
				ofs << " " << points[v].dLdF(i);
			}
			ofs << "\n";
		}
	}
	ofs.close();
}

//void DiffMPMLib3D::PointCloud::WriteEntirePointCloudToFile(std::string file_path)
//{
//	// not sure how much I want to invest into this codebase in the future, so just doing the easy thing here
//	std::ofstream ofs;
//	ofs.open(file_path);
//
//	if (ofs.good()) {
//		// first line: number of points
//		ofs.precision(17);
//		ofs << points.size() << std::endl;
//		for (size_t v = 0; v < points.size(); v++) {
//			points[v].WriteEntirePointToFile(ofs);
//		}
//	}
//	ofs.close();
//}
//
//bool DiffMPMLib3D::PointCloud::ReadEntirePointCloudFromFile(std::string file_path)
//{
//	std::ifstream ifs;
//	ifs.open(file_path);
//
//	if (ifs.good()) {
//		// first line: number of points
//		size_t num_points;
//		ifs >> num_points;
//
//		points.resize(num_points);
//
//		for (size_t v = 0; v < num_points; v++) {
//			points[v].ReadEntirePointFromFile(ifs);
//		}
//
//		ifs.close();
//		return true;
//	}
//	else {
//		std::cout << "couldn't read file: " << file_path << std::endl;
//		return false;
//	}
//}

void DiffMPMLib3D::PointCloud::WriteEntirePointCloudToBinaryFile(std::string file_path)
{
	std::ofstream ofs;
	ofs.open(file_path, std::ios::binary);

	if (ofs.good()) {
		cereal::BinaryOutputArchive oarchive(ofs); // Create an output archive

		oarchive(*this);
	}
	ofs.close();
}

bool DiffMPMLib3D::PointCloud::ReadEntirePointCloudFromBinaryFile(std::string file_path)
{
	std::ifstream ifs;
	ifs.open(file_path, std::ios::binary);
	if (ifs.good()) {
		cereal::BinaryInputArchive iarchive(ifs); // Create an input archive

		iarchive(*this);
	}
	else {
		std::cout << "couldn't read file: " << file_path << std::endl;
		return false;
	}
	ifs.close();
	return true;
}

bool DiffMPMLib3D::PointCloud::IsEqualToOtherPointCloud(const PointCloud& other_pc)
{
	for (int i = 0; i < points.size(); i++) {
		const MaterialPoint& mp = points[i];
		const MaterialPoint& other_mp = other_pc.points[i];
		if (!mp.IsEqualToOtherPoint(other_mp)) {

			std::cout << "Not equal at point " << i << std::endl;
			std::cout << "*** original MP ***" << std::endl;
			mp.PrintMP();
			std::cout << "*** other MP ***" << std::endl;
			other_mp.PrintMP();
			return false;
		}
	}

	std::cout << "Point clouds are equal" << std::endl;
	return true;
}


std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::PointCloud::GetPointPositions() const
{
	std::vector<Vec3> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].x;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointMasses() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].m;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointVolumes() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].vol;
	}

	return ret;
}

std::vector<DiffMPMLib3D::Mat3> DiffMPMLib3D::PointCloud::GetPointDefGrads() const
{
	std::vector<Mat3> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].F;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointDeterminants() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = (points[p].F + points[p].dFc).determinant();
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointElasticEnergies() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = FixedCorotatedElasticity(points[p].F + points[p].dFc, points[p].lam, points[p].mu);
	}

	return ret;
}

// [FIX] NEW: Gradient getter methods for PCGrad
std::vector<DiffMPMLib3D::Mat3> DiffMPMLib3D::PointCloud::GetPointDefGradGradients() const
{
	std::vector<Mat3> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].dLdF;
	}

	return ret;
}

std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::PointCloud::GetPointPositionGradients() const
{
	std::vector<Vec3> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].dLdx;
	}

	return ret;
}

void DiffMPMLib3D::PointCloud::AddJBarrierGradient(float J_target, float w_barrier)
{
	// Quadratic penalty barrier that pushes J = det(F_eff) toward J_target from below.
	//
	// Barrier loss:  L_b = (w/2) * max(0, J_target - J)^2
	// dL_b/dJ      = -w * max(0, J_target - J)          (negative when J < J_target)
	// dJ/dF_eff    = J * F_eff^{-T}                      (Jacobi's formula)
	// dL_b/dF_eff  = dL_b/dJ * dJ/dF_eff
	//              = -w*(J_target-J) * J * F_eff^{-T}    (injected into dLdF)
	//
	// Effect in Adam: dFc -= lr * step(dLdF)
	//   => negative dLdF contribution  => dFc increases => J increases  (desired)

#pragma omp parallel for
	for (int i = 0; i < (int)points.size(); ++i)
	{
		const Mat3 F_eff = points[i].F + points[i].dFc;
		const float J = F_eff.determinant();

		// Only apply barrier when J < J_target and F_eff is non-degenerate
		if (J >= J_target || J < 1e-6f) continue;

		const float violation = J_target - J;              // > 0
		const float dLdJ = -w_barrier * violation;         // < 0  (want J to increase)

		// dJ/dF = J * F^{-T}  (Jacobi's formula)
		// Use Eigen's inverse (safe for 3x3)
		const Mat3 F_inv_T = F_eff.inverse().transpose();
		const Mat3 barrier_grad = (dLdJ * J) * F_inv_T;    // (N/m, typically small)

		points[i].dLdF += barrier_grad;
	}
}
