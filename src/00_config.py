"""
Global config, device setup, and reproducibility seed.

Original notebook cell(s): [0, 1, 2]
"""

# ----- (notebook cell 0) -----
import os, math, random, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomErasing, RandomResizedCrop
import torchvision.models as models
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_curve
)
import rasterio

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
torch.backends.cudnn.benchmark = True

# ----- (notebook cell 1) -----
class CFG:
    ROOT          = Path('/kaggle/input/datasets/shreyansdeshpande/200x200/FINALCNNDATA_200')
    OUTPUTDIR     = Path('/kaggle/working')

    TRAINSITE     = ROOT / 'train' / 'site'
    TRAINNOSITE   = ROOT / 'train' / 'no_site'
    VALSITE       = ROOT / 'validation' / 'site'
    VALNOSITE     = ROOT / 'validation' / 'no_site'
    TESTSITE      = ROOT / 'test' / 'site'
    TESTNOSITE    = ROOT / 'test' / 'no_site'

    IMGSUFFIX     = '.tif'
    SATBANDS      = [0, 1, 2, 3]
    LIDARBANDS    = [4, 5, 6, 7, 8, 9, 10]
    INCHANNELS    = len(SATBANDS) + len(LIDARBANDS)

    NUMCLASSES    = 1
    IMGSIZE       = 200
    BATCHSIZE     = 32
    NUMEPOCHS     = 40
    BASELR        = 5e-5
    WEIGHTDECAY   = 0.1
    DROPOUT       = 0.6
    SEED          = 42
    NUMWORKERS    = 2
    USEAMP        = torch.cuda.is_available()

    PATIENCE      = 3
    MINDELTA      = 1e-3
    SCHED_FACTOR  = 0.5
    SCHED_PATIENCE = 2

    USEPOSWEIGHT  = False
    USEFOCALLOSS  = False
    FOCALGAMMA    = 2.0
    LABELSMOOTH   = 0.0

    HEADLRMULT    = 5.0
    LLRD          = 0.8
    FREEZEEPOCHS  = 8
    FREEZESTAGES  = 4

    STOCHDEPTH    = 0.3
    MIXUPPROB     = 0.0
    MIXUPALPHA    = 0.4
    ERASEPROB     = 0.10
    CROPSCALE     = (0.85, 1.0)
    SATNOISESTD   = 0.02
    LIDARNOISESTD = 0.05

    STEMNOISESTD  = 1e-4
    RGBDROPPROB   = 0.30
    LIDARDROPPROB = 0.15
    SAT_SCALE     = 10000.0
    IMAGENET_NORM = True

    LIDARNORM     = 'global'
    LIDARCLIP     = 8.0
    EVALTRAINEACHEPOCH = True

# ----- (notebook cell 2) -----
def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setseed(CFG.SEED)
CFG.OUTPUTDIR.mkdir(parents=True, exist_ok=True)
