"""Fixed FGR positive control; GT is accessed only after all poses are saved."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/fgr_20261018'
PAIRS = (100, 101, 124, 125, 200, 201)


def load(pair, evaluate=False):
    folder = '4dmatch_replication' if pair < 200 else 'hf_3dmatch'
    path = ROOT/f'assets/{folder}/{pair}.npz'
    with np.load(path, allow_pickle=False) as z:
        if pair < 200:
            x, y = z['s_pc'].astype(float), z['t_pc'].astype(float)
            g = (x+z['s2t_flow'])@z['rot'].T+z['trans'].reshape(1, 3) if evaluate else None
        else:
            x, y = z['source']/z['scale'], z['target']/z['scale']
            g = z['truth']/z['scale'] if evaluate else None
    return x, y, g, path


def prepare(x, y):
    import open3d as o3d
    spacing = np.concatenate([cKDTree(z).query(z, k=2)[0][:, 1] for z in (x, y)]).mean()
    voxel = float(2*spacing)
    assert voxel > 0
    clouds, features = [], []
    for z in (x, y):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(z)).voxel_down_sample(voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel, max_nn=30))
        feature = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel, max_nn=100))
        clouds.append(cloud)
        features.append(feature)
    return voxel, clouds, features


def voxel_truth(x, truth, down, voxel):
    """Average truth over the exact source voxel members, retaining output order."""
    x = np.asarray(x, dtype=np.float64)
    origin = x.min(0)-.5*voxel
    keys, inverse = np.unique(np.floor((x-origin)/voxel).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    means, mapped = np.zeros((len(keys), 3)), np.zeros((len(keys), 3))
    np.add.at(means, inverse, x)
    np.add.at(mapped, inverse, truth)
    means /= counts[:, None]
    mapped /= counts[:, None]
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    order = [lookup[tuple(key)] for key in np.floor((down-origin)/voxel).astype(np.int64)]
    assert len(order) == len(keys) and len(set(order)) == len(order)
    np.testing.assert_allclose(means[order], down, rtol=0, atol=1e-12)
    return mapped[order]


def descriptor_audit(source_features, target_features, truth, target, threshold):
    """Independent raw-feature NN audit; does not emulate native FGR filtering."""
    k = min(10, len(target))
    neighbors = cKDTree(target_features).query(source_features, k=list(range(1, k+1)))[1]
    reverse = cKDTree(source_features).query(target_features)[1]
    correct = np.linalg.norm(target[neighbors]-truth[:, None], axis=2) <= threshold
    supported = cKDTree(target).query(truth)[0] <= threshold
    mutual = reverse[neighbors[:, 0]] == np.arange(len(truth))
    result = dict(source_voxels=len(truth), supported_count=int(supported.sum()),
                  support_fraction=float(supported.mean()), top1_precision=float(correct[:, 0].mean()),
                  mutual_count=int(mutual.sum()), mutual_correct=int(correct[mutual, 0].sum()),
                  mutual_precision=float(correct[mutual, 0].mean()) if mutual.any() else None,
                  source_zero_features=int((np.linalg.norm(source_features, axis=1) == 0).sum()))
    for requested in (1, 5, 10):
        found = correct[:, :min(requested, k)].any(1)
        result[f'top{requested}_coverage'] = float(found.mean())
        result[f'top{requested}_supported_coverage'] = float(found[supported].mean()) if supported.any() else None
    return result


def metrics(points, truth):
    distance = np.linalg.norm(points-truth, axis=1)
    return dict(mse=float(((points-truth)**2).sum(1).mean()), epe=float(distance.mean()))


def main():
    import open3d as o3d
    assert o3d.__version__ == '0.19.0'
    assert not (OUT/'observed.json').exists()
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
    prior = json.loads((ROOT/'paper_revision/joint_20261018/pose_observed.json').read_text())
    old = {r['pair']: r for r in prior}
    rows, prepared = [], {}
    reg = o3d.pipelines.registration
    for pair in PAIRS:
        x, y, _, path = load(pair)
        start = time.perf_counter()
        voxel, clouds, features = prepare(x, y)
        options = dict(division_factor=1.4, use_absolute_scale=False, decrease_mu=False,
                       maximum_correspondence_distance=1.5*voxel, iteration_number=64,
                       tuple_scale=.95, maximum_tuple_count=1000, tuple_test=True)
        assert voxel == old[pair]['voxel']
        assert [len(c.points) for c in clouds] == old[pair]['sampled_points']
        o3d.utility.random.seed(18000+pair)
        result = reg.registration_fgr_based_on_feature_matching(
            *clouds, *features, reg.FastGlobalRegistrationOption(**options))
        assert np.isfinite(result.transformation).all()
        row = dict(pair=pair, transform=result.transformation.tolist(), fitness=result.fitness,
                   inlier_rmse=result.inlier_rmse, voxel=voxel, options=options,
                   sampled_points=[len(c.points) for c in clouds], seed=18000+pair,
                   seconds=time.perf_counter()-start, asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
        prepared[pair] = clouds, features
        print(json.dumps(dict(pair=pair, fitness=row['fitness'], seconds=row['seconds'])), flush=True)
    observed = dict(rows=rows, open3d_version=o3d.__version__)
    (OUT/'observed.json').write_text(json.dumps(observed, indent=2)+'\n')
    observed_hash = hashlib.sha256((OUT/'observed.json').read_bytes()).hexdigest()
    # No truth has been accessed until this point. All observed poses are frozen.
    prior = json.loads((ROOT/'paper_revision/joint_20261018/pose_evaluation.json').read_text())
    old = {r['pair']: r for r in prior['rows']}
    for row in rows:
        pair = row['pair']
        x, y, truth, _ = load(pair, True)
        t = np.array(row['transform'])
        for name, points in [('identity', x), ('centroid', x-x.mean(0)+y.mean(0)),
                             ('fgr', x@t[:3, :3].T+t[:3, 3])]:
            row[name] = metrics(points, truth)
            if name != 'fgr':
                for metric in ('mse', 'epe'):
                    np.testing.assert_allclose(row[name][metric], old[pair][f'{name}_{metric}'], rtol=1e-12, atol=1e-14)
        row['ransac'] = {key: old[pair]['ransac_'+key] for key in ('mse', 'epe')}
        row['ndp_epe'] = old[pair]['ndp_epe']
        clouds, features = prepared[pair]
        mapped = voxel_truth(x, truth, np.asarray(clouds[0].points), row['voxel'])
        row['descriptors'] = descriptor_audit(np.asarray(features[0].data).T,
            np.asarray(features[1].data).T, mapped, np.asarray(clouds[1].points), 1.5*row['voxel'])
    wins = [r['pair'] for r in rows if r['fgr']['mse'] < r['centroid']['mse']]
    passed = all(p in wins for p in (200, 201)) and sum(p < 200 for p in wins) >= 3
    evaluation = dict(rows=rows, mse_wins_vs_centroid=wins, preliminary_screen=passed,
                      adopted=False, observed_sha256=observed_hash, open3d_version=o3d.__version__)
    (OUT/'evaluation.json').write_text(json.dumps(evaluation, indent=2)+'\n')
    assert hashlib.sha256((OUT/'observed.json').read_bytes()).hexdigest() == observed_hash
    print('Preliminary proposal screen:', passed, flush=True)


if __name__ == '__main__':
    main()
