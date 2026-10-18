"""Check GT voxel membership and diagnostic coverage without running registration."""
import numpy as np
from global_registration_control import descriptor_audit, voxel_truth


def test_voxel_truth_retains_membership_and_output_order():
    x = np.array([[0., 0, 0], [.1, 0, 0], [1., 0, 0]])
    truth = np.array([[0., 0, 0], [.4, 0, 0], [3., 0, 0]])
    down = np.array([[1., 0, 0], [.05, 0, 0]])
    np.testing.assert_allclose(voxel_truth(x, truth, down, 1.), [[3., 0, 0], [.2, 0, 0]])


def test_descriptor_coverage_separates_support_and_precision():
    target = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    truth = np.array([[1., 0, 0], [0., 0, 0], [8., 0, 0]])
    source_features = np.array([[0.], [1.], [2.]])
    result = descriptor_audit(source_features, source_features, truth, target, .1)
    assert result['supported_count'] == 2
    assert result['mutual_count'] == 3
    assert result['top1_precision'] == result['mutual_precision'] == 0
    assert result['top5_coverage'] == 2/3
    assert result['top5_supported_coverage'] == 1
    assert result['top10_coverage'] == result['top5_coverage']


def test_descriptor_no_gt_support_is_defined():
    result = descriptor_audit(np.array([[0.]]), np.array([[0.]]),
        np.array([[5., 0, 0]]), np.array([[0., 0, 0]]), .1)
    assert result['supported_count'] == 0
    assert result['top1_supported_coverage'] is None
    assert result['mutual_correct'] == 0
