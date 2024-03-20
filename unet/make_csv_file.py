from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
import torch
import os
import pandas as pd

# path
current_path = Path.cwd()
unet_path = current_path.parent
segmentation_path = unet_path.parent
neural_rep_path = segmentation_path.parent
source_img_path = Path(neural_rep_path / 'leftImg8bit' / 'train')
gt_img_path = Path(neural_rep_path / 'gtFine' / 'train')
source_img_path_val = Path(neural_rep_path / 'leftImg8bit' / 'val')
gt_img_path_val = Path(neural_rep_path / 'gtFine' / 'val')


# create csv file
def create_csv_file(source_path, gt_path):
    matched_files = []

    # access all files
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith('_leftImg8bit.png'):
                identifier = '_'.join(file.split('_')[:3])
                gt_file_name = f"{identifier}_gtFine_labelIds.png"
                gt_color_name = f"{identifier}_gtFine_color.png"
                gt_file_path = Path(gt_path) / Path(root).relative_to(source_path) / gt_file_name
                gt_file_color_path = Path(gt_path) / Path(root).relative_to(source_path) / gt_color_name
                if gt_file_path.exists(): 
                    matched_files.append({
                        'img_name' : str(identifier),
                        'source_img': os.path.join(root, file),
                        'gt_labe_img': str(gt_file_path),
                        'gt_color_map': str(gt_file_color_path)
                    })

    df = pd.DataFrame(matched_files)

    # save csv
    df.to_csv('./cityscape_val.csv', index=False)

create_csv_file(source_img_path_val, gt_img_path_val)
