"""Same correspondence and gates; replace force fitting by weighted displacement fitting."""
from functools import partial
from pipeline.model import ROOT
from ndp_capacity_probe import run
from ndp_feature_probe import build_fields

OUT = ROOT/'paper_revision/weighted_20261015'


if __name__ == '__main__':
    run(OUT, partial(build_fields, out=OUT), weighted=True)
