// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#ifndef PCH_H
#define PCH_H

// add headers that you want to pre-compile here
#include "framework.h"

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <math.h>
#include <iostream>
#include <array>

#include <limits>

namespace DiffMPMLib3D {
	typedef Eigen::Vector3f Vec3;
	typedef Eigen::Matrix3f Mat3;

	inline float InnerProduct(const Mat3& A, const Mat3& B)
	{
		return (A.array() * B.array()).sum();
	}

	inline Mat3 CofactorMatrix(const Mat3& A)
	{
		return A.adjoint().transpose();
	}
}

#endif //PCH_H
