import os
import shutil
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import json
import pprint

from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirst,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotate90d,
    Resize,
    Resized,
    EnsureTyped,
    Flipd,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.config import print_config
import torch
import einops
import warnings
from skimage.util import montage
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from monai.utils import set_determinism

train_transforms = Compose(
    [
        # 載入影像和標籤
        LoadImaged(keys=["image", "label"], ensure_channel_first=True), 
        # 對影像進行強度縮放，將像素值映射到指定的範圍
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # 去除影像周圍的空白區域
        CropForegroundd(keys=["image", "label"], source_key="image"), 
        # 將影像和標籤轉換為特定的方向
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 調整影像和標籤的像素間距
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")), # 導致形狀變化
        # 隨機裁剪正樣本和負樣本，生成訓練用的子卷積
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(512, 512, 32), pos=1, neg=1, num_samples=1, image_key="image", image_threshold=0),
        # 隨機沿指定軸翻轉影像和標籤
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10), 
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10), 
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        # 隨機旋轉影像和標籤90度
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),  
        # 隨機改變影像的強度    
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),    
    ]
)

datasets = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset0/0325/dataset_Axial.json"
train_files = load_decathlon_datalist(datasets, True, "training")

# train_ds = CacheDataset(data=train_files, transform=None, cache_num=24, cache_rate=1.0, num_workers=2)
# img1 = train_ds[0]["image"]


train_ds2 = CacheDataset(data=train_files, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=2)
img2 = train_ds2[0]

# 從img2中獲取圖像數據和標籤數據
image_data = img2["image"]
label_data = img2["label"]

# 將圖像數據轉換為numpy數組
image_array = np.squeeze(image_data.numpy())

# 如果您的圖像數據是灰度圖像，請將其作為灰度圖像顯示
plt.imshow(image_array, cmap='gray')
plt.axis('off')  # 隱藏坐標軸
plt.savefig('/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/crop_image/transform_image.png')