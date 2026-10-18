import torch
from pipeline.refresh import accepted_path


class Warp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([.05, -.1], dtype=torch.float64))

    def forward(self, x):
        return x+self.theta.tanh()+.1*x.square()*self.theta.square(), None


def test_accepted_refresh_uses_actual_network_and_shared_objective():
    model = Warp(); saved = model.theta.detach().clone()
    x = torch.tensor([[-.3, .2], [.4, -.1], [.2, .5]], dtype=torch.float64)
    q = torch.tensor([[.8, .7], [1.2, .1], [.7, 1.3], [1.6, .9]], dtype=x.dtype)
    for kind in ('cd', 'uot'):
        arms = [list(accepted_path(model, x, q, kind, flag, 13, .05)) for flag in (False, True)]
        observations = []
        observed = list(accepted_path(model, x, q, kind, True, 13, .05,
            observe=lambda *args: observations.append(tuple(v.clone() for v in args))))
        assert len(observations) == 4
        for original, repeated in zip(arms[1], observed):
            assert original[0] == repeated[0]
            for a, b in zip(original[1:], repeated[1:]):
                if a is not None: torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(arms[0][0][4], arms[1][0][4], rtol=0, atol=0)
        for rows in arms:
            assert len(rows) == 4 and len({r[0]['solve']['damping'] for r in rows}) == 1
            for i, (r, initial, p, velocity, after, old) in enumerate(rows):
                assert r['fraction'] > 0 and r['linearization_error'] <= .1
                assert r['objective_change'] <= .1*r['fraction']*r['slope']+1e-12
                if i: torch.testing.assert_close(p, rows[i-1][4], rtol=0, atol=0)
                torch.testing.assert_close(initial, model(x)[0])
            assert rows[-1][0]['force_refresh_relative'] > 0
        # Each policy really follows its own accepted nonlinear trajectory.
        assert (arms[0][-1][4]-arms[1][-1][4]).norm() > 1e-8
    torch.testing.assert_close(model.theta, saved, rtol=0, atol=0)
