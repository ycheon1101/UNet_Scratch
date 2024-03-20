import pandas as pd
import os
import shutil
from PIL import Image

def crop_center(image_path, output_path, target_size=(388, 388)):
    with Image.open(image_path) as img:
        width, height = img.size
        left = (width - target_size[0])/2
        top = (height - target_size[1])/2
        right = (width + target_size[0])/2
        bottom = (height + target_size[1])/2

        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)

# path
csv_file_path_train = './cityscape_train.csv'
csv_file_path_val = './cityscape_val.csv'
image_dir_path = './images'
image_train_path = './images/train'
image_val_path = './images/val'

# read csv
df_train = pd.read_csv(csv_file_path_train)
df_val = pd.read_csv(csv_file_path_val)

# check df info
# print(f'df_train: {df_train.info()}')
# print(f'df_val: {df_val.info()}')

# make dir and store images
if not os.path.exists(image_dir_path):
    os.makedirs(image_dir_path)
if not os.path.exists(image_train_path):
    os.makedirs(image_train_path)
if not os.path.exists(image_val_path):
    os.makedirs(image_val_path)

# train
for idx, row in df_val.iterrows():    # train / val
    img_name = row['img_name']
    source_img = row['source_img']
    gt_color_map = row['gt_color_map']

    img_path = os.path.join(image_val_path, img_name)   # train / val

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    
    shutil.copy2(source_img, img_path)
    shutil.copy2(gt_color_map, img_path)
    # crop_center(gt_color_map, f'{image_val_path}/{img_name}/color_cropped_img.png') 

    
