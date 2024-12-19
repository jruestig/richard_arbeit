import nifty8 as ift
from os.path import join, isfile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import configparser
import resolve as rve

from utils import load_fits

import sys
from sys import exit


def load_samples_ift(output_directory: str) -> ift.SampleList:
    fname = join(output_directory, "pickle", "last")
    if isfile(fname + ".mean.pickle"):
        # mean = ift.ResidualSampleList.load_mean(fname)
        sl = ift.ResidualSampleList.load(fname)
    else:
        raise FileNotFoundError
    return sl


config_file = sys.argv[1]
output_directory = join(*config_file.split('/')[:-1])
samples = load_samples_ift(output_directory)
cfg = configparser.ConfigParser()
cfg.read(f"./{config_file}")
print(f'Config read {config_file}')


sky, sky_diffuse_operators = rve.sky_model_diffuse(cfg['sky'])
sky_mean, sky_variance = samples.sample_stat(sky)
sky_mean, sky_std = sky_mean.val[0, 0, 0].T, sky_variance.sqrt().val[0, 0, 0].T

# ground_truth = load_fits(...)
