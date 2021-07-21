import os

from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

if __name__ == '__main__':
    BASE_DIR = "../../"
    DATASET_PATH = "../../input/siim-covid19-detection/"
    train_df = pd.read_csv(f'{DATASET_PATH}train_image_level.csv')
    path = f'{DATASET_PATH}train/ae3e63d94c13/288554eb6182/e00f9fe0cce5.dcm'
    dicom = pydicom.read_file(path)
    image_id = []
    dim0 = []
    dim1 = []
    splits = []

    for split in ['test', 'train']:
        save_dir = f'/kaggle/tmp/{split}/'

        os.makedirs(save_dir, exist_ok=True)
        
        for dirname, _, filenames in tqdm(os.walk(f'{DATASET_PATH}{split}')):
            for file in filenames:
                # set keep_ratio=True to have original aspect ratio
                xray = read_xray(os.path.join(dirname, file))
                im = resize(xray, size=256)  
                im.save(os.path.join(save_dir, file.replace('dcm', 'jpg')))

                image_id.append(file.replace('.dcm', ''))
                dim0.append(xray.shape[0])
                dim1.append(xray.shape[1])
                splits.append(split)

    
    #!tar -zcf train.tar.gz -C "/kaggle/tmp/train/" .
    #!tar -zcf test.tar.gz -C "/kaggle/tmp/test/" .


















#python "src/convert_dicom_to_jpg.py"