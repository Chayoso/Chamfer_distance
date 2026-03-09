#pragma once
#include "pch.h"
#include "Tensor3x3x3x3.h"

// Ensure Tensor3x3x3x3 is available
using DiffMPMLib3D::Tensor3x3x3x3;

namespace DiffMPMLib3D {
    // Finite difference versions (primarily for debugging/testing)
    Tensor3x3x3x3 d_JFit_dF_FD(const Mat3& F);
    Tensor3x3x3x3 d2_FCE_psi_dF2_FD(const Mat3& F, float lam, float mu);

    // Analytical functions (for performance)
    float FixedCorotatedElasticity(const Mat3& F, float lam, float mu);
    Mat3 PK_FixedCorotatedElasticity(const Mat3& F, float lam, float mu);
    Mat3 d2_FCE_psi_dF2_mult_by_dF(const Mat3& F, float lam, float mu, const Mat3& dF);
    
    // [NEW] Analytical derivative for d(J*F^{-T})/dF
    Tensor3x3x3x3 d_JFit_dF(const Mat3& F);

    // Utility
    void CalculateLameParameters(float young_mod, float poisson, float& lam, float& mu);
}