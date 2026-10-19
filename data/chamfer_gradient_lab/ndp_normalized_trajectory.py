"""Add one frozen normalized-mixture arm to the existing nine-by16 study."""
import json
import sys
import ndp_dense_trajectory as experiment

experiment.OUT=experiment.ROOT/'paper_revision/normalized_trajectory_20261018'
experiment.MATCHED_REFERENCE=experiment.ROOT/'paper_revision/dense_pcg_20261018/runs'
experiment.PREREQUISITE=experiment.ROOT/'paper_revision/normalized_timing_20261018/summary.json'
experiment.PRIMARY='short_fgr_normmix'
experiment.ARMS=(*experiment.ARMS,experiment.PRIMARY)
experiment.KINDS=dict(experiment.KINDS,short_fgr_normmix='fgr_normmix')


if __name__=='__main__':
    prior=experiment.PREREQUISITE.parent
    assert json.loads((prior/'VALIDATION.json').read_text())['source_files']==177
    assert json.loads(experiment.PREREQUISITE.read_text())['advance_to_cost_based_schedule']
    experiment.torch.set_num_threads(1);experiment.torch.backends.cuda.matmul.allow_tf32=False
    if '--report-only' in sys.argv:
        from dense_trajectory_report import main
        main()
    elif '--preflight' in sys.argv:experiment.preflight()
    else:experiment.main()
