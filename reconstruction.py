import resolve as rve
import nifty8 as ift

from utils import (
    save_config_copy_easy, build_callback, build_samples, build_minimizer,
    build_sampling,
)

from os import makedirs
from os.path import join

import configparser

observation = rve.Observation.load("./data/synthetic_observation/m51ha.npz")


cfg = configparser.ConfigParser()
config_path = "./m51.cfg"
cfg.read(config_path)

epsilon = 1e-4
nthreads = 1


sky, sky_diffuse_operators = rve.sky_model_diffuse(cfg['sky'])

lh = rve.ImagingLikelihood(
    observation,
    sky_operator=sky,
    epsilon=epsilon,
    do_wgridding=False,
    nthreads=nthreads,
    verbosity=False)

output_name = cfg['General']['output name'].format(
    npix=cfg['sky']['space npix x'],
    fov=cfg['sky']['space fov x'],
)
output_directory = f"output/{output_name}"
makedirs(output_directory, exist_ok=True)
save_config_copy_easy(config_path, join(output_directory, 'config.cfg'))

callback = build_callback(sky, output_directory, True, save_fits=False)
resume = eval(cfg['minimization']['resume'])
n_iterations = int(cfg['minimization']['iterations'])
n_samples = build_samples(cfg)
minimizer = build_minimizer(cfg)
ic_sampling = build_sampling(cfg)

samples, sci_position = ift.optimize_kl(
    lh,
    n_iterations,
    n_samples,
    minimizer,
    ic_sampling,
    None,
    return_final_position=True,
    output_directory=output_directory,
    comm=None,
    inspect_callback=callback,
    export_operator_outputs={
        key: val for key, val in sky_diffuse_operators.items() if 'power' not in key
    },
    resume=cfg['minimization']['resume'],
)
