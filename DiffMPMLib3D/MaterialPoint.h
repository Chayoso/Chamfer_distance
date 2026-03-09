#pragma once
#include "pch.h"
#include <fstream>
#include "cereal/archives/binary.hpp"
#include "cereal_eigen.h"

namespace DiffMPMLib3D {
	struct MaterialPoint
	{
		void ResetGradients();
		void WriteEntirePointToFile(std::ofstream& ofs);
		void ReadEntirePointFromFile(std::ifstream& ifs);

		bool IsEqualToOtherPoint(const MaterialPoint& other_mp) const;
		void PrintMP() const;

		// stores every intermediate value in computation graph
		Vec3 x = Vec3::Zero();
		Vec3 v = Vec3::Zero();
		Mat3 F = Mat3::Identity();
		Mat3 C = Mat3::Zero();
		Mat3 P = Mat3::Zero();

		float m = 0.f;
		float vol = 0.f;

		float lam = 0.f;
		float mu = 0.f;

		// control variable
		Mat3 dFc = Mat3::Zero(); // How much to change deformation gradient in this timestep

		// gradients
		Vec3 dLdx = Vec3::Zero();
		Vec3 dLdv = Vec3::Zero();
		Mat3 dLdF = Mat3::Zero();
		Mat3 dLdC = Mat3::Zero();
		Mat3 dLdP = Mat3::Zero();

		Vec3 dLdv_next = Vec3::Zero();
		Mat3 dLdC_next = Mat3::Zero();
		

		Mat3 momentum = Mat3::Zero();
		Mat3 vector = Mat3::Zero();

		Mat3 vector_max = Mat3::Zero();

		// Independent render Adam state (separate from physics Adam)
		Mat3 momentum_r = Mat3::Zero();
		Mat3 vector_r   = Mat3::Zero();

		// these don't do anything, I don't know why I added them in
		/*float dLdm = 0.0;
		float dLdvol = 0.0;*/

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(
				CEREAL_NVP(x),
				CEREAL_NVP(v),
				CEREAL_NVP(F),
				CEREAL_NVP(C),
				CEREAL_NVP(P),
				CEREAL_NVP(m),
				CEREAL_NVP(vol),
				CEREAL_NVP(lam),
				CEREAL_NVP(mu),
				CEREAL_NVP(dFc),
				CEREAL_NVP(dLdx),
				CEREAL_NVP(dLdv),
				CEREAL_NVP(dLdF),
				CEREAL_NVP(momentum),
				CEREAL_NVP(vector),
				CEREAL_NVP(vector_max)
				// other stuff i doubt i would care about?
			);
		}
	};
}