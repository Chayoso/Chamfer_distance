#pragma once
#include "pch.h"

namespace DiffMPMLib3D {
	struct GridNode
	{
		void ResetGradients();

		Vec3 x;
		Vec3 v;
		Vec3 p;

		float m;

		Vec3 dLdx;
		Vec3 dLdv;
		Vec3 dLdp;

		float dLdm;

		/*GridNode()
			: x(Vec3::Zero()), v(Vec3::Zero()), p(Vec3::Zero()), m(0.0f),
			dLdx(Vec3::Zero()), dLdv(Vec3::Zero()), dLdp(Vec3::Zero()), dLdm(0.0f) {}*/
	};
}