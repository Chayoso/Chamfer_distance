#include "pch.h"
#include "Elasticity.h"
#include "qrsvd/ImplicitQRSVD.h"
#include <chrono>
#include <unsupported/Eigen/KroneckerProduct>

namespace DiffMPMLib3D {

    // [OPTIMIZATION] Analytical implementation of d(J*F^{-T})/dF.
    // This is significantly faster than the finite difference version.
    // The formula comes from continuum mechanics.
    Tensor3x3x3x3 d_JFit_dF(const Mat3& F)
    {
        Tensor3x3x3x3 dJFit_dF;
        Mat3 Fit = F.inverse().transpose();
        float J = F.determinant();

        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        float val = J * (Fit(i, k) * Fit(m, n) - Fit(m, k) * Fit(i, n));
                        // This corresponds to d(J*Fit)_{ik} / dF_{mn}
                        // To match the Tensor3x3x3x3 layout dJFit_dF[i][k](m,n)
                        dJFit_dF[i][k](m, n) = val;
                    }
                }
            }
        }
        return dJFit_dF;
    }

    float FixedCorotatedElasticity(const Mat3& F, float lam, float mu)
    {
        Mat3 U, V;
        Vec3 S;
        JIXIE::singularValueDecomposition(F, U, S, V);

        float J = F.determinant();
        float s_sum = (S.array() - 1.0f).square().sum();

        return mu * s_sum + 0.5f * lam * (J - 1.0f) * (J - 1.0f);
    }

    Mat3 PK_FixedCorotatedElasticity(const Mat3& F, float lam, float mu)
    {
        Mat3 R, S;
        JIXIE::polarDecomposition(F, R, S);

        float J = F.determinant();
        // Clamp J to avoid issues with inversion or negative determinants
        J = std::max(J, 1e-6f);

        return 2.f * mu * (F - R) + lam * (J - 1.f) * J * F.inverse().transpose();
    }

    Mat3 d2_FCE_psi_dF2_mult_by_dF(const Mat3& F, float lam, float mu, const Mat3& dF)
    {
        Mat3 R, S;
        JIXIE::polarDecomposition(F, R, S);
        float J = F.determinant();
        Mat3 Fit = F.inverse().transpose();
        J = std::max(J, 1e-6f);

        Mat3 ret = Mat3::Zero();

        // Term 1: 2 * mu * dF
        ret += 2.f * mu * dF;

        // Term 2: From d/dF(lam * (J-1) * J * F^{-T}) contracted with dF
        ret += lam * J * Fit * InnerProduct(J * Fit, dF);

        // [OPTIMIZATION] Call the fast analytical function instead of the slow finite difference one.
        Tensor3x3x3x3 dJFit_dF_tensor = d_JFit_dF(F);

        Mat3 term = Mat3::Zero();
    #pragma omp parallel for
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                // Contract the 4th-order tensor with the matrix dF
                term(a, b) = (dJFit_dF_tensor[a][b].array() * dF.array()).sum();
            }
        }
        ret += lam * (J - 1.f) * term;


        // Term 3: From the derivative of the rotation matrix R (-2 * mu * dR)
        float s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
        float s11 = S(1, 1), s12 = S(1, 2);
        float s22 = S(2, 2);
        
        Mat3 A;
        A(0,0) = s11+s22; A(0,1) = -s01;   A(0,2) = -s02;
        A(1,0) = -s01;   A(1,1) = s00+s22; A(1,2) = -s12;
        A(2,0) = -s02;   A(2,1) = -s12;   A(2,2) = s00+s11;
        
        // Regularize if matrix is close to singular
        if (std::abs(A.determinant()) < 1e-6) {
            A += Mat3::Identity() * 1e-4f;
        }
        
        Mat3 temp = R.transpose() * dF;
        Vec3 b(temp(2, 1) - temp(1, 2), temp(0, 2) - temp(2, 0), temp(1, 0) - temp(0, 1));

        Vec3 w = A.colPivHouseholderQr().solve(b);

        Mat3 dR_dw; // Skew-symmetric matrix from w
        dR_dw << 0, -w.z(), w.y(),
                 w.z(), 0, -w.x(),
                -w.y(), w.x(), 0;
        
        Mat3 dR = R * dR_dw;

        ret -= 2.f * mu * dR;

        return ret;
    }

    // This function can be kept for debugging and verification purposes.
    Tensor3x3x3x3 d2_FCE_psi_dF2_FD(const Mat3& F, float lam, float mu)
    {
        Tensor3x3x3x3 ret;
        float delta = 1e-6f;
        Mat3 temp_F = F;

    #pragma omp parallel for
        for (int ij = 0; ij < 9; ++ij) {
            int i = ij / 3;
            int j = ij % 3;
            
            Mat3 F_plus = F, F_minus = F;
            F_plus(i, j) += delta;
            F_minus(i, j) -= delta;

            Mat3 P_forward = PK_FixedCorotatedElasticity(F_plus, lam, mu);
            Mat3 P_backward = PK_FixedCorotatedElasticity(F_minus, lam, mu);
            
            Mat3 grad = (P_forward - P_backward) / (2.f * delta);
            
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    ret[a][b](i, j) = grad(a, b);
                }
            }
        }
        return ret;
    }
    
    void CalculateLameParameters(float young_mod, float poisson, float& lam, float& mu)
    {
        lam = young_mod * poisson / ((1.f + poisson) * (1.f - 2.f * poisson));
        mu = young_mod / (2.f + 2.f * poisson);
    }
}