import resolve as rve
import nifty8 as ift
import matplotlib.pyplot as plt
import astropy.units as u


observation = rve.Observation.load("./data/synthetic_observation/m51ha.npz")

shape = 256
sdom = ift.RGSpace(shape=(shape,)*2, distances=(10*u.arcsec/shape).to(u.rad))
pdom = rve.PolarizationSpace(['I'])
tdom = fdom = rve.IRGSpace([1.0])
domain = ift.makeDomain((pdom, tdom, fdom, sdom))


R = rve.InterferometryResponse(
    observation,
    domain,
    do_wgridding=False,
    epsilon=1e-5)


Radjoint_vis = R.adjoint(observation.vis)
plt.imshow(Radjoint_vis.val[0, 0, 0], origin='lower')
plt.show()
