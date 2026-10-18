"""Evaluation-only proposal/transfer decomposition; never an update selector."""
from pipeline.model import torch


def measure(p, truth, force, mass, motion):
    task = 2*(p-truth)/len(p)
    active = mass > 0
    assert (mass >= 0).all() and (force[~active] == 0).all()
    proposal = torch.zeros_like(force)
    proposal[active] = force[active]/mass[active, None]
    correction = motion-proposal
    def slope(v):
        return float((task*v).sum())
    def cosine(v):
        norm = float(task.norm()*v.norm())
        return slope(v)/norm if norm else 0.
    def unit(v):
        norm = float(v.norm())
        return v/norm if norm else torch.zeros_like(v)
    root = mass.sqrt()[:, None]
    weighted = float((root*proposal).norm())
    plain = float(proposal.norm())
    result = dict(force_slope=slope(force), proposal_slope=slope(proposal),
        projected_slope=slope(motion), correction_slope=slope(correction),
        force_cosine=cosine(force), proposal_cosine=cosine(proposal),
        projected_cosine=cosine(motion), zero_mass_rows=int((~active).sum()),
        supported_correction_slope=slope(correction*active[:, None]),
        unsupported_motion_slope=slope(motion*(~active)[:, None]),
        raw_transfer_error=float(correction.norm())/plain if plain else None,
        weighted_transfer_error=float((root*correction).norm())/weighted if weighted else None,
        direction_gap=float((unit(motion)-unit(proposal)).norm()),
        positive_mass_min=float(mass[active].min()),
        proposal_norm=plain, projected_norm=float(motion.norm()), task_norm=float(task.norm()))
    result['slope_identity_error'] = abs(result['projected_slope']-result['proposal_slope']-result['correction_slope'])
    result['directional_descent_certificate'] = bool(plain and result['projected_norm'] and
        result['direction_gap'] < -result['proposal_cosine'])
    return result
