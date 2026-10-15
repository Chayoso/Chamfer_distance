from rigid_control import np,align


def test_proper_rigid_alignment_and_held_points():
    rng=np.random.default_rng(871);x=rng.normal(size=(50,3))
    r,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(r)<0:r[:,0]*=-1
    q=np.einsum('ij,nj->ni',r,x)+[.2,-.3,.8]
    assert np.allclose(align(x[:20],q[:20],x),q,rtol=1e-12,atol=1e-12)
