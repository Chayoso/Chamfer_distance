from pipeline.selection import select


def test_explicit_cd_ridge_baseline_can_be_a_ray_rule_fallback():
    def row(method, change):
        return dict(method=method, cd_derivative=-1., cd_change=change, accepted=True,
                    linearization_error=0., ray_error=None)
    records = [row('adam', -.1), row('cd_gd', -.2), row('cd_ridge', -.3)]
    assert select(records, 'ray_cd', ('adam', 'cd_gd', 'cd_ridge')) == 'cd_ridge'
    assert select(records, 'cd_trial', ('adam', 'cd_gd', 'cd_ridge')) == 'cd_ridge'
