#include "pch.h"
#include "GeometryLoading.h"
#include "igl/point_mesh_squared_distance.h"
#include "igl/signed_distance.h"
#include <vector>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <random>

std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::GeometryLoading::GeneratePointCloudFromWatertightTriangleMesh(
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXi& F,
    Vec3 min_point,
    Vec3 max_point,
    float sampling_dx,
    bool apply_jitter)
{
    using namespace Eigen;

    // 1. Generate uniform grid sample points with optional perturbation
    int dims[3];
    for (int i = 0; i < 3; i++) {
        dims[i] = static_cast<int>(std::ceil((max_point[i] - min_point[i]) / sampling_dx)) + 1; // +1 to include max_point
    }

    std::vector<Vec3> sample_points;
    sample_points.reserve(dims[0] * dims[1] * dims[2]);
    
    // [OK] Random number generator for perturbation (only if apply_jitter is true)
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    const float perturbation_scale = 0.1f;  // 10% of grid spacing
    std::uniform_real_distribution<float> dis(-perturbation_scale * sampling_dx, 
                                               perturbation_scale * sampling_dx);
    
//#pragma omp parallel for collapse(3)
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                // [OK] Add random perturbation only if apply_jitter is true
                Vec3 perturbation = apply_jitter ? Vec3(dis(gen), dis(gen), dis(gen)) : Vec3(0.0f, 0.0f, 0.0f);
                Vec3 point = min_point + sampling_dx * Vec3(float(i), float(j), float(k)) + perturbation;
                sample_points.emplace_back(point);
            }
        }
    }
    
    if (apply_jitter) {
        std::cout << "Applied perturbation: ±" << (perturbation_scale * sampling_dx) 
                  << " (±" << (perturbation_scale * 100) << "% of dx)" << std::endl;
    } else {
        std::cout << "No perturbation applied (apply_jitter=false)" << std::endl;
    }

    // Convert sample_points to Eigen matrix
    MatrixXf P(sample_points.size(), 3);
    for (size_t i = 0; i < sample_points.size(); ++i) {
        P.row(i) = sample_points[i];
    }

    // Debug: Output the range of sample points
    std::cout << "Sample points range: ["
        << P.col(0).minCoeff() << ", " << P.col(0).maxCoeff() << "] x ["
        << P.col(1).minCoeff() << ", " << P.col(1).maxCoeff() << "] x ["
        << P.col(2).minCoeff() << ", " << P.col(2).maxCoeff() << "]" << std::endl;

    // Debug: Output the range of mesh vertices
    std::cout << "Mesh vertices range: ["
        << V.col(0).minCoeff() << ", " << V.col(0).maxCoeff() << "] x ["
        << V.col(1).minCoeff() << ", " << V.col(1).maxCoeff() << "] x ["
        << V.col(2).minCoeff() << ", " << V.col(2).maxCoeff() << "]" << std::endl;

    // 2. Get signed distances
    VectorXf S;
    VectorXi I;
    MatrixXf C;
    MatrixXf N;

    igl::SignedDistanceType sign_type = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER;
    igl::signed_distance(P, V, F, sign_type, S, I, C, N);

    // Debug: Output some signed distances
    std::cout << "Signed distances: " << std::endl;
    for (int i = 0; i < std::min(10, (int)S.size()); ++i) {
        std::cout << "P[" << i << "] = " << P.row(i) << ", S = " << S[i] << std::endl;
    }

    // 3. Store all points with negative signed distance
    std::vector<Vec3> points;
    points.reserve(P.rows());
    for (int i = 0; i < P.rows(); i++) {
        if (S[i] <= 0.0) {
            points.push_back(P.row(i));
        }
    }

    // Debug: Output the number of points inside the mesh
    std::cout << "Number of points inside the mesh: " << points.size() << std::endl;

    return points;
}

bool DiffMPMLib3D::GeometryLoading::LoadMPMPointCloudFromObj(
    std::string obj_path,
    std::shared_ptr<PointCloud>& mpm_point_cloud,
    float point_dx,
    float density,
    float lam,
    float mu,
    bool apply_jitter
)
{
    std::cout << "reading " << obj_path << "..." << std::endl;
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(obj_path, V, F)) {
        std::cout << "error reading " << obj_path << std::endl;
        return false;
    }

    Vec3 min_point = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 max_point = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (size_t i = 0; i < (size_t)V.rows(); i++) {
        for (size_t j = 0; j < 3; j++) {
            min_point[j] = std::fmin(min_point[j], V(i, j));
            max_point[j] = std::fmax(max_point[j], V(i, j));
        }
    }

    std::cout << obj_path << "'s max point is: " << max_point.transpose() << std::endl;
    std::cout << obj_path << "'s min point is: " << min_point.transpose() << std::endl;

    std::cout << "generating point cloud..." << std::endl;

    std::vector<Vec3> points = GeometryLoading::GeneratePointCloudFromWatertightTriangleMesh(V, F, min_point, max_point, point_dx, apply_jitter);


    std::cout << obj_path << "'s max y point is: " << max_point[1] << std::endl;
    std::cout << obj_path << "'s min Y point is: " << min_point[1] << std::endl;

    // MPM Point Cloud
    std::cout << "generating mpm points..." << std::endl;
    mpm_point_cloud = std::make_shared<PointCloud>();
    mpm_point_cloud->points.resize(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        const auto& p = points[i];
        mpm_point_cloud->points[i].x = points[i];
        mpm_point_cloud->points[i].v = Vec3(0.0, 0.0, 0.0);
        mpm_point_cloud->points[i].F = Mat3::Identity();
        
         // Mass is consistent with a uniform voxel of size point_dx^3 at density 'density'
        mpm_point_cloud->points[i].m = point_dx * point_dx * point_dx * density;
        
        // REST VOLUME (V_p0): keep the classical MPM convention: vol = m / rho0 = point_dx^3
        // This ensures the stress term uses the correct rest volume scale from the start.
        mpm_point_cloud->points[i].vol = mpm_point_cloud->points[i].m / density; // == point_dx^3
        mpm_point_cloud->points[i].lam = lam;
        mpm_point_cloud->points[i].mu = mu;
    }

    return true;
}