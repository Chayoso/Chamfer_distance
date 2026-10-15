"""The expanded solver grid and same-correspondence attribution must be explicit."""
import copy
import pytest
from test_capacity_report import complete_records
from capacity_report import verify_coverage,KINDS,METHODS
from weighted_report import COMPARISONS,CANDIDATES


def test_weighted_solver_rows_are_required_in_the_expanded_grid():
    dr,fr,sr,solvers,integrity=complete_records()
    for row in sr:
        new=dict(row,method=row['method']+'_weighted');dr.append(new);solvers.append(copy.deepcopy(new))
        for policy in ('raw','trust','cd_trust'):
            for cap in (1/16,.25,1.):fr.append(dict(new,policy=policy,cap=cap))
    integrity['rows']={n:len(v) for n,v in zip(('directions','finite','states','solver'),(dr,fr,sr,solvers))}
    methods=METHODS+tuple(k+'_weighted' for k in KINDS)
    solver_methods=tuple(k+'_'+m for k in KINDS for m in ('lift','weighted'))
    verify_coverage(dr,fr,sr,solvers,integrity,KINDS,methods,solver_methods)
    solvers[-1]=copy.deepcopy(solvers[0])
    with pytest.raises(AssertionError,match='solver'):
        verify_coverage(dr,fr,sr,solvers,integrity,KINDS,methods,solver_methods)


def test_attribution_preserves_both_features_and_marginal_model():
    for candidate in CANDIDATES:
        field=candidate.removesuffix('_weighted')
        assert COMPARISONS['lift'][candidate]==field+'_lift'
        assert COMPARISONS['gd'][candidate]==field+'_gd'
