import resolve as rve
import nifty8 as ift

from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

import numpy as np

from os import makedirs
from os.path import join


def save_config_copy_easy(path_to_file: str, path_to_save_file: str):
    from shutil import copy, SameFileError
    try:
        copy(path_to_file, path_to_save_file)
        print(f"Config file saved to: {path_to_save_file}.")
    except SameFileError:
        pass


def build_minimizer(cfg: dict) -> Callable:
    cfg_mini = cfg['minimization']

    de, ce, ie = [float(v) for v in cfg_mini['newton early'].split(', ')]
    dl, cl, il = [float(v) for v in cfg_mini['newton late'].split(', ')]

    ic_newton_early = ift.AbsDeltaEnergyController(
        name="Newton", deltaE=de, convergence_level=ce, iteration_limit=ie
    )
    ic_newton_late = ift.AbsDeltaEnergyController(
        name="Newton", deltaE=dl, convergence_level=cl, iteration_limit=il
    )

    switch = eval(cfg_mini['switch'])
    minimizer_early = ift.NewtonCG(ic_newton_early)
    minimizer_late = ift.NewtonCG(ic_newton_late)

    return lambda i: minimizer_early if i < switch else minimizer_late


def build_sampling(cfg: dict) -> Callable:
    cfg_mini = cfg['minimization']
    switch = eval(cfg_mini['switch'])

    de, ie = [float(v) for v in cfg_mini['sampling early'].split(', ')]
    dl, il = [float(v) for v in cfg_mini['sampling early'].split(', ')]
    print(f'Sampling (early): DeltaE {de}, iteration limit {ie}')
    print(f'Sampling (late): DeltaE {dl}, iteration limit {il}')

    ic_sampling_early = ift.AbsDeltaEnergyController(
        name="Sampling (linear)", deltaE=de, iteration_limit=ie
    )
    ic_sampling_late = ift.AbsDeltaEnergyController(
        name="Sampling (linear)", deltaE=dl, iteration_limit=il
    )

    return lambda i: ic_sampling_early if i < switch else ic_sampling_late


def build_samples(cfg: dict) -> Callable:
    switch = eval(cfg['samples']['switch'])
    samples_early = eval(cfg['samples']['samples early'])
    samples_late = eval(cfg['samples']['samples late'])
    return lambda i:  samples_early if i < switch else samples_late


def build_callback(sky, output_directory, master, save_fits=True):
    print('Output:', output_directory)
    makedirs(output_directory, exist_ok=True)

    def callback(samples: ift.SampleList, i: int):
        print('Plotting iteration', i, 'in: ', output_directory)

        sky_mean = samples.average(sky)
        pols, ts, freqs, *_ = sky_mean.shape
        fig, axes = plt.subplots(pols, freqs, figsize=(freqs*4, pols*3))

        if save_fits:
            rve.ubik_tools.field2fits(sky_mean, join(
                output_directory, f'sky_reso_iter{i}.fits'))

        if freqs == 1:
            if pols == 1:
                axes = [axes]

            for poli, ax in enumerate(axes):
                f = sky_mean.val[poli, 0, 0].T
                if poli > 0:
                    f = np.abs(f)

                im = ax.imshow(f, origin="lower", norm=LogNorm())
                plt.colorbar(im, ax=ax)

        elif pols == 1:
            if freqs == 1:
                axes = [axes]

            for freqi, ax in enumerate(axes):
                im = ax.imshow(
                    sky_mean.val[0, 0, freqi].T, origin="lower", norm=LogNorm())
                plt.colorbar(im, ax=ax)

        else:
            for poli, pol_axes in enumerate(axes):
                for freqi, ax in enumerate(pol_axes):
                    if poli > 0:
                        f = np.abs(f)
                    f = sky_mean.val[poli, 0, freqi].T
                    im = ax.imshow(f, origin="lower", norm=LogNorm())
                    plt.colorbar(im, ax=ax)

        plt.tight_layout()
        if master:
            plt.savefig(f"{output_directory}/resolve_iteration_{i}.png")
        plt.close()

    return callback


def load_fits(path_to_file, fits_number=0, sum_axis=None, get_header=False):
    if sum_axis is not None:
        with fits.open(path_to_file) as hdul:
            header = hdul[0].header
            data = hdul[0].data.sum(axis=sum_axis)
            multiplicity = hdul[0].data.shape[sum_axis]
        if get_header:
            return np.array(data), multiplicity, header
        return np.array(data), multiplicity

    with fits.open(path_to_file) as hdul:
        headers = [h.header for h in hdul if isinstance(h, ImageHDU)]
        datas = [h.data for h in hdul if isinstance(h, ImageHDU)]
        if len(headers) == 0:
            header = hdul[0].header
            data = hdul[0].data
        else:
            header = headers[fits_number]
            data = datas[fits_number]
    if get_header:
        return np.array(data).astype(np.float64), header
    return np.array(data).astype(np.float64)
