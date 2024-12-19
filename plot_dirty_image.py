import resolve as rve
import matplotlib.pyplot as plt
import nifty8 as ift
import astropy.units as u


observation = rve.Observation.load("./data/synthetic_observation/m51ha.npz")

sdom = ift.RGSpace(shape=(1024,)*2, distances=(10*u.arcsec).to(u.rad))
pdom = rve.PolarizationSpace(['I'])
tdom = fdom = rve.IRGSpace([1.0])
domain = ift.makeDomain((pdom, tdom, fdom, sdom))


R = rve.InterferometryResponse(
    observation,
    domain,
    do_wgridding=False,
    epsilon=1e-5)


Radjoint_vis = R.adjoint(observation.vis)

plt.imshow(Radjoint_vis.val[0, 0, 0])
plt.show()
