import os
import shutil
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import json
import pprint
import torch.nn.functional as F
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
# from monai.networks.nets import SwinUNETR
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

from DiceCELoss_deepSupervision import DiceCELoss_DeepSupervision

from Swin_UNETR_DeepSupervision import SwinUNETR_DeepSupervision, SwinUNETR
from Swin_UNETR_DCA import SwinUNETR_DeepSupervision_DCA

_seed = 0
set_determinism(seed=_seed)

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

num_samples = 1
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
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")), # 導致形狀變化
        # 將影像和標籤轉換為指定的裝置（在這裡是GPU），並取消跟踪元數據
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        # 隨機裁剪正樣本和負樣本，生成訓練用的子卷積
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0),
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

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")), # 導致形狀變化
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"],pixdim=(1.5, 1.5, 2.0),mode=("bilinear")),
        EnsureTyped(keys=["image"], device=device, track_meta=True),
    ]
)

def validation(epoch_iterator_val, with_deepSupervision):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_label = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_label_list = decollate_batch(val_label)
            val_label_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_label_list
            ]
            if with_deepSupervision is True:
                val_outputs_list = decollate_batch(val_outputs[0])
            else:
                val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_label_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 1.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best, model_path, with_deepSupervision):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())

        # print(f"Training Input Shape: {x.shape}")  # 打印輸入形狀

        with torch.cuda.amp.autocast():
            logit_map = model(x)
            # print(f"Training Output Shape: {logit_map[0].shape}")  # 打印輸出形狀
            loss = loss_function(logit_map, y)

        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val, with_deepSupervision)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), model_path)
                print(f"\nModel Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}")
            else:
                print(f"\nModel Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}")
        global_step += 1
    return global_step, dice_val_best, global_step_best

if __name__ == '__main__':
   
    # ----------------------------- Load Data ---------------------------------------
    with_deepSupervision = True    # 有沒有加上deepSupervision

    datasets = f"/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/dataset_Coronal.json"

    train_files = load_decathlon_datalist(datasets, True, "training")
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_num=47, cache_rate=1.0, num_workers=2)
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)

    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=12, cache_rate=1.0, num_workers=2)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False) # 控制是否追踪數據集的metadata(描述數據的數據)

    # ---------------------------- Model -------------------------------

    model = SwinUNETR_DeepSupervision_DCA(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2, # 4改成2(因為分兩類)
            feature_size=48,
            use_checkpoint=False,
        ).to(device)

    # if with_deepSupervision is True:
    #     model = SwinUNETR_DeepSupervision(
    #         img_size=(96, 96, 96),
    #         in_channels=1,
    #         out_channels=2, # 4改成2(因為分兩類)
    #         feature_size=48,
    #         use_checkpoint=False,
    #     ).to(device)
    # else:
    #     model = SwinUNETR(
    #         img_size=(96, 96, 96),
    #         in_channels=1,
    #         out_channels=2, # 4改成2(因為分兩類)
    #         feature_size=48,
    #         use_checkpoint=False,
    #     ).to(device)

    # --------------------------- Pre-training -------------------------
    # weight = torch.load("data/model_swinvit.pt")
    # model.load_from(weights=weight)

    # --------------------------- Training ------------------------------
    torch.backends.cudnn.benchmark = True # 啟用 cuDNN 的自動優化（在某些情況下可能提高性能）
    if with_deepSupervision is True:
        loss_function = DiceCELoss_DeepSupervision(to_onehot_y=True, softmax=True) # 定義損失函數
    else:
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True) # 定義損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # 定義優化器（使用 AdamW，學習率 1e-4，權重衰減 1e-5）
    scaler = torch.cuda.amp.GradScaler() # 使用混合精度訓練的 GradScaler

    # --------------------- 可調的參數 ---------------------
    max_iterations = 30000    # 最大訓練迭代次數
    eval_num = 100         # 每多少次迭代進行一次評估

    model_path = f"/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Coronal)/DCA/Coronal_best_metric_model(seed={_seed}).pth"  # 欲儲存的模型路徑
    result_png_path = f"/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Coronal)/DCA/Coronal_Results_Dice(seed={_seed}).png"
    # ------------------------------------------------------

    post_label = AsDiscrete(to_onehot = 2) # 定義後處理轉換（以確保預測和標籤的格式） # class n
    post_pred = AsDiscrete(argmax=True, to_onehot = 2) # class n
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False) # 定義 Dice 指標（包括背景，使用平均值，不包括 NaN 值）
    global_step = 0 # 全局步數初始化
    dice_val_best = 0.0 # 存儲最佳 Dice 值的變數
    global_step_best = 0 # 存儲達到最佳 Dice 值時的全局步數
    epoch_loss_values = [] # 存儲每個 epoch 的損失值
    metric_values = [] # 存儲每個 epoch 的度量值（這裡是 Dice 值）

    while global_step < max_iterations: # 訓練循環
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best, model_path, with_deepSupervision)
    
    model.load_state_dict(torch.load(model_path)) # 加載達到最佳 Dice 值時的模型權重
    print(f"train completed, best_metric: {dice_val_best:.4f} at iteration: {global_step_best}")

    # ------------------------- Print Results ---------------------------------------
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.savefig(result_png_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")