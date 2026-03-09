#pragma once
#include "pch.h"

namespace DiffMPMLib3D {
	inline float CubicBSpline(float x) {
		x = abs(x);
		if (0.f <= x && x < 1.f) {
			return 0.5f * x * x * x - x * x + 2.f / 3.f;
		}
		else if (1.f <= x && x < 2.f) {
			return (2.f - x) * (2.f - x) * (2.f - x) / 6.f;
		}
		else {
			return 0.f;
		}
	}

	inline float CubicBSplineSlope(float x)
	{
		float absx = abs(x);
		if (0.f <= absx && absx < 1.f) {
			return 1.5f * x * absx - 2.f * x;
		}
		else if (1.f <= absx && absx < 2.f) {
			return -x * absx / 2.f + 2.f * x - 2.f * x / absx;
		}
		else {
			return 0.f;
		}
	}
}