# %% [markdown]
# ## Train YoloR on COTS dataset (PART 1 - TRAINING) - as easy as possible to help people start with YoloR and develop this notebook
# This notebook introduces YOLOR on Kaggle and TensorFlow - Help Protect the Great Barrier Reef competition. It shows how to train custom object detection model (COTS dataset) using YoloR. It could be good starting point for build own custom model based on YoloR detector. Full github repository you can find here - [YOLOR](https://github.com/WongKinYiu/yolor)
# 
# Steps covered in this notebook:
# 
# * Prepare COTS dataset for YoloR training
# * Install YoloR (YoloR, MISH CUDA, pytorch_wavelets)
# * Download Pre-Trained Weights for YoloR HUB
# * Prepare configuration files (YoloR hyperparameters and dataset)
# * Weights and Biases configuration for training logging
# * YoloR training
# * Run YoloR inference on test images
# 
# <div class="alert alert-warning">I found that there is no reference custom model training YoloR notebook on Kaggle. Since we have such an opportunity this is my contribution to this competition. Feel free to use it and enjoy! I really appreciate if you upvote this notebook. Thank you!</div>
# 
# <div class="alert alert-success" role="alert">
# I introduced YoloX in TensorFlow - Help Protect the Great Barrier Reef competition as well. You can find these notebooks here:      
#     <ul>
#         <li> <a href="https://www.kaggle.com/remekkinas/yolox-full-training-pipeline-for-cots-dataset">YoloX full training pipeline for COTS dataset</a></li>
#         <li> <a href="https://www.kaggle.com/remekkinas/yolox-inference-on-kaggle-for-cots-lb-0-507">YoloX detections submission made on COTS dataset</a></li>
#     </ul>
#     
# </div>

# %% [markdown]
# <div align="center"><img width="640" src="https://github.com/WongKinYiu/yolor/raw/main/figure/unifued_network.png"/></div>
# 
# <div align="center"><img width="640" src="https://github.com/WongKinYiu/yolor/raw/main/figure/performance.png"/></div>

# %% [markdown]
# ## 0. IMPORT MODULES

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:28:49.971645Z","iopub.execute_input":"2021-12-28T12:28:49.97231Z","iopub.status.idle":"2021-12-28T12:28:50.021166Z","shell.execute_reply.started":"2021-12-28T12:28:49.972215Z","shell.execute_reply":"2021-12-28T12:28:50.020506Z"}}
import ast
import glob
import os
import yaml

import numpy as np
import pandas as pd


from IPython.display import Image, display
from IPython.core.magic import register_line_cell_magic
from shutil import copyfile
from tqdm import tqdm
tqdm.pandas()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:28:50.024226Z","iopub.execute_input":"2021-12-28T12:28:50.024422Z","iopub.status.idle":"2021-12-28T12:28:50.029828Z","shell.execute_reply.started":"2021-12-28T12:28:50.024398Z","shell.execute_reply":"2021-12-28T12:28:50.029104Z"}}
HOME_DIR = '/media/panda/0CFC3A54FC3A37F2/dataset'
COTS_DATASET_PATH = '/media/panda/0CFC3A54FC3A37F2/dataset/great-barrier-reef/train_images'

# ## 1. PREPARE DATASET
# I just used spllited dataset by @julian3833 - Reef - A CV strategy: subsequences!
# https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences 

df = pd.read_csv("/media/panda/dataset/Hobby/yolor-paper/train-validation-split/train-0.1.csv")
print(df.head(3))

def add_path(row):
    return f"{COTS_DATASET_PATH}/video_{row.video_id}/{row.video_frame}.jpg"

def num_boxes(annotations):
    annotations = ast.literal_eval(annotations)
    return len(annotations)

df['path'] = df.apply(lambda row: add_path(row), axis=1)
df['num_bbox'] = df['annotations'].apply(lambda x: num_boxes(x))
print("New path and annotations preprocessing completed")

df = df[df.num_bbox > 0]

print(f'Dataset images with annotations: {len(df)}')

def add_new_path(row):
    if row.is_train:
        return f"{HOME_DIR}/yolor_dataset/images/train/{row.image_id}.jpg"
    else: 
        return f"{HOME_DIR}/yolor_dataset/images/valid/{row.image_id}.jpg"
    

df['new_path'] = df.apply(lambda row: add_new_path(row), axis=1)
print("New image path for train/valid created")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:28:51.268063Z","iopub.execute_input":"2021-12-28T12:28:51.268746Z","iopub.status.idle":"2021-12-28T12:28:51.283895Z","shell.execute_reply.started":"2021-12-28T12:28:51.268708Z","shell.execute_reply":"2021-12-28T12:28:51.283252Z"}}
df.head(3)

# %% [markdown]
# ## 2. CREATE DATASET FILE STRUCTURE

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:28:51.285076Z","iopub.execute_input":"2021-12-28T12:28:51.285464Z","iopub.status.idle":"2021-12-28T12:28:51.291734Z","shell.execute_reply.started":"2021-12-28T12:28:51.285428Z","shell.execute_reply":"2021-12-28T12:28:51.291033Z"}}
os.makedirs(f"{HOME_DIR}/yolor_dataset/images/train")
os.makedirs(f"{HOME_DIR}/yolor_dataset/images/valid")
os.makedirs(f"{HOME_DIR}/yolor_dataset/labels/train")
os.makedirs(f"{HOME_DIR}/yolor_dataset/labels/valid")
print(f"Directory structure yor YoloR created")

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:28:51.293036Z","iopub.execute_input":"2021-12-28T12:28:51.293509Z","iopub.status.idle":"2021-12-28T12:29:47.415126Z","shell.execute_reply.started":"2021-12-28T12:28:51.293472Z","shell.execute_reply":"2021-12-28T12:29:47.414439Z"}}
def copy_file(row):
  copyfile(row.path, row.new_path)

_ = df.progress_apply(lambda row: copy_file(row), axis=1)

# %% [markdown]
# ## 3. CREATE YoloR ANNOTATIONS

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:29:47.416346Z","iopub.execute_input":"2021-12-28T12:29:47.416774Z","iopub.status.idle":"2021-12-28T12:29:49.044748Z","shell.execute_reply.started":"2021-12-28T12:29:47.416737Z","shell.execute_reply":"2021-12-28T12:29:49.043893Z"}}
IMG_WIDTH, IMG_HEIGHT = 1280, 720

def get_yolo_format_bbox(img_w, img_h, box):
    w = box['width'] 
    h = box['height']
    
    if (bbox['x'] + bbox['width'] > 1280):
        w = 1280 - bbox['x'] 
    if (bbox['y'] + bbox['height'] > 720):
        h = 720 - bbox['y'] 
        
    xc = box['x'] + int(np.round(w/2))
    yc = box['y'] + int(np.round(h/2)) 

    return [xc/img_w, yc/img_h, w/img_w, h/img_h]
    

for index, row in tqdm(df.iterrows()):
    annotations = ast.literal_eval(row.annotations)
    bboxes = []
    for bbox in annotations:
        bbox = get_yolo_format_bbox(IMG_WIDTH, IMG_HEIGHT, bbox)
        bboxes.append(bbox)
        
    if row.is_train:
        file_name = f"{HOME_DIR}/yolor_dataset/labels/train/{row.image_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    else:
        file_name = f"{HOME_DIR}/yolor_dataset/labels/valid/{row.image_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
    with open(file_name, 'w') as f:
        for i, bbox in enumerate(bboxes):
            label = 0
            bbox = [label]+bbox
            bbox = [str(i) for i in bbox]
            bbox = ' '.join(bbox)
            f.write(bbox)
            f.write('\n')
                
print("Annotations in YoloR format for all images created.")

# %% [markdown]
# ## 4. CREATE YoloR DATASET CONFIGURATION FILE

# %% [code] {"execution":{"iopub.status.busy":"2021-12-28T12:29:49.046407Z","iopub.execute_input":"2021-12-28T12:29:49.046682Z","iopub.status.idle":"2021-12-28T12:29:49.055929Z","shell.execute_reply.started":"2021-12-28T12:29:49.046646Z","shell.execute_reply":"2021-12-28T12:29:49.054889Z"}}
data_yaml = dict(
    train = f'{HOME_DIR}/yolor_dataset/images/train',
    val = f'{HOME_DIR}/yolor_dataset/images/valid',
    nc = 1,
    names = ['sf']
)


with open(f'{HOME_DIR}/YoloR-data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)

print(f'Dataset configuration file for YoloR created')

# ## 4. INSTALL YoloR
# 
# ### 4A. CLONE YoloR GIT REPOSITORY 

#!git clone https://github.com/WongKinYiu/yolor

#!pip install torchvision --upgrade -q
#!pip install wandb --upgrade

#%cd yolor
#!pip install -qr requirements.txt

# %% [markdown]
# ### 4B. INSTALL MISH CUDA

#%cd ..
#!git clone https://github.com/JunnYu/mish-cuda
#%cd mish-cuda
#!git reset --hard 6f38976064cbcc4782f4212d7c0c5f6dd5e315a8
#!python setup.py build install
#%cd ..

# %% [markdown]
# ### 4C. INSTALL PYTORCH WAVELETS 

#!git clone https://github.com/fbcotter/pytorch_wavelets
#%cd pytorch_wavelets
#!pip install .
#%cd ..

# %% [markdown]
# ### 4D. DWONLOAD LATEST CHECKPOINT FROM YoloR MODEL HUB 
# 
# In this notebook we take P6 model (because I want to show only how to train YoloR model on Kaggle) but you can experiment with other YoloR models: https://github.com/WongKinYiu/yolor

#%cd yolor
#!bash scripts/get_pretrain.sh


#%%writetemplate /kaggle/working/yolor/data/coco.yaml

#nc: 1
#names: ['starfish',]

#%%writetemplate /kaggle/working/yolor/data/coco.names

#starfish
"""
%%writetemplate /kaggle/working/hyp-yolor.yaml

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.0  # image HSV-Hue augmentation (fraction)
hsv_s: 0.0  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.0  # image HSV-Value augmentation (fraction)
degrees: 90.0  # image rotation (+/- deg)
translate: 0.5  # image translation (+/- fraction)
scale: 0.0  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 0.95  # image mosaic (probability)
mixup: 0.3  # image mixup (probability)
"""

# ## 5. TRAIN YoloR
"""
!python train.py \
 --batch-size 4 \
 --img 1280 768 \
 --data '{HOME_DIR}/YoloR-data.yaml' \
 --cfg './cfg/yolor_p6.cfg' \
 --weights './yolor_p6.pt' \
 --device 0 \
 --name yolor_p6 \
 --hyp '/kaggle/working/hyp-yolor.yaml' \
 --epochs 8
"""
# ## 6. INFERENCE USING YoloR 
"""
INFER_PATH = f"{HOME_DIR}/yolor_dataset/infer"
os.makedirs(INFER_PATH)

df_infer = df.query("~is_train and num_bbox > 4").sample(n = 15)

def copy_file(row):
    new_location = INFER_PATH + '/' + row.image_id + '.jpg'
    copyfile(row.path, new_location)

_ = df_infer.progress_apply(lambda row: copy_file(row), axis=1)

!python detect.py \
    --source {INFER_PATH} \
    --cfg ./cfg/yolor_p6.cfg \
    --weights '/kaggle/working/yolor/runs/train/yolor_p6/weights/best_overall.pt' \
    --conf 0.05 \
    --img-size 1280 \
    --device 0 

for img in glob.glob('/kaggle/working/yolor/inference/output/*.jpg'): 
    display(Image(filename=img))
    print("\n")

"""