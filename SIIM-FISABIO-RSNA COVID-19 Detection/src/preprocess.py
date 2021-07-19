import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
import scipy.ndimage
import gdcm

import glob

from skimage import measure 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening, closing
from tqdm import tqdm

from IPython.display import HTML
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from os import listdir, mkdir

BASE_DIR = "../../../../kaggle/"
DATASET_PATH = "../../../../kaggle/input/siim-covid19-detection/"
print(listdir(BASE_DIR))

# Read in metadata
train_study_df = pd.read_csv(f"{DATASET_PATH}/train_study_level.csv")
train_image_df = pd.read_csv(f"{DATASET_PATH}/train_image_level.csv")

print("Train Study Shape: ", train_study_df.shape, "\n" +
      "Train Image Shape: ", train_image_df.shape, "\n" +
      "\n" +
      "Note: There are {} missing values in train_image_df.".\
                              format(train_image_df["boxes"].isna().sum()), "\n" +
      "This happens for labels = 'none' - no checkboxes.", 3*"\n")



