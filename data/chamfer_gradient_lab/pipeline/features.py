"""Observable FPFH likelihoods, frozen during each matching proposal."""
from .model import np,torch


def descriptor(points,spacing):
    import open3d as o3d
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.detach().cpu().numpy()))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*spacing,max_nn=30))
    cloud.orient_normals_towards_camera_location(cloud.get_center())
    raw=np.asarray(o3d.pipelines.registration.compute_fpfh_feature(
        cloud,o3d.geometry.KDTreeSearchParamHybrid(radius=5*spacing,max_nn=100)).data).T.copy()
    if not np.isfinite(raw).all():raise ValueError('Nonfinite FPFH descriptor')
    norms=np.linalg.norm(raw,axis=1,keepdims=True)
    return torch.as_tensor(raw/np.maximum(norms,1e-30),device=points.device,dtype=points.dtype),int((norms==0).sum())


def feature_penalty(p,q):
    nearest=[]
    for points in (p,q):
        distance=torch.cdist(points.detach(),points.detach(),compute_mode='donot_use_mm_for_euclid_dist')
        distance.fill_diagonal_(float('inf'));nearest.append(distance.min(1).values)
    spacing=float(torch.cat(nearest).mean())
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Invalid sampling spacing')
    a,az=descriptor(p,spacing);b,bz=descriptor(q,spacing)
    penalty=.5*torch.cdist(a,b,compute_mode='donot_use_mm_for_euclid_dist').square()/.1**2
    return penalty,dict(spacing=spacing,source_zero_features=az,target_zero_features=bz)
