#pragma once
#include "pch.h"
#include "CompGraph.h"


namespace DiffMPMLib3D {
	void Back_Timestep(CompGraphLayer& layer_nplus1, CompGraphLayer& layer_n, float drag, float dt, float smoothing_factor);
}