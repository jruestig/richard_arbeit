import resolve as rve
import nifty8 as ift
from astropy import units as u
import matplotlib.pyplot as plt

import numpy as np
from radio_project_helpers.data_loading import freq_average

from charm_lensing.utils import load_fits


file = './data/real_observation/uid___A002_Xb66ea7_X4dc3.ms.split.cal_only_spw0_SPT0418-47_spw0.npz'
fits = './data/fitsfiles/M51ha.fits'
map_I = load_fits(fits)
observation = rve.Observation.load(file)
observation = freq_average(observation, 3)


sdom = ift.RGSpace(shape=(1024,)*2, distances=(10*u.arcsec/1024).to(u.rad))
pdom = rve.PolarizationSpace(['I'])
tdom = fdom = rve.IRGSpace([1.0])
domain = ift.makeDomain((pdom, tdom, fdom, sdom))

field = ift.makeField(domain, map_I[None, None, None])

R = rve.InterferometryResponse(
    observation,
    domain,
    do_wgridding=False,
    epsilon=1e-5)

vis = R(field)
noise_std = np.max(np.abs(vis.val)) * 0.005
visibilities_corrupted = ift.makeField(
    R.target,
    np.array(vis.val + np.random.normal(0, noise_std, vis.shape),
             dtype=np.complex64)
)
map_Radjoint_vis = R.adjoint(vis)
map_Radjoint_viscorr = R.adjoint(visibilities_corrupted)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(map_I, origin='lower')
axes[1].imshow(map_Radjoint_vis.val[0, 0, 0], origin='lower')
axes[2].imshow(map_Radjoint_viscorr.val[0, 0, 0], origin='lower')
plt.show()


weight = 1/noise_std**2
weights = ift.makeField(
    observation.weight.domain,
    np.full(observation.weight.shape, weight))

synthetic_observation = rve.Observation(
    observation.antenna_positions,
    visibilities_corrupted.val,
    np.full(observation.weight.shape, weight, dtype=np.float32),
    observation.polarization,
    observation.freq,
    observation._auxiliary_tables
)

synthetic_observation.save(
    './data/synthetic_observation/m51ha.npz', compress=False
)
