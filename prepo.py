# Mert Hacıahmetoğlu
# 03.08.2019

"""hyperparameters script"""

#---------------------------------------------------------------------
from __future__ import print_function

from araclar import spektrogram_yukle
import os
from data_yukle import yukle_data
import numpy as np
import tqdm

# Load data
fpaths, _, _ = yukle_data() # list

for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = spektrogram_yukle(fpath)
    if not os.path.exists("mels"): os.mkdir("mels")
    if not os.path.exists("mags"): os.mkdir("mags")

    np.save("mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("mags/{}".format(fname.replace("wav", "npy")), mag)