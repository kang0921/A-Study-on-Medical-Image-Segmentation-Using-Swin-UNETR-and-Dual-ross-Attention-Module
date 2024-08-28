import os
import shutil
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import json
import pprint
from scipy.ndimage import zoom
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
from Swin_UNETR_DeepSupervision import SwinUNETR_DeepSupervision, SwinUNETR
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
from matplotlib.colors import ListedColormap
from monai.networks.nets import UNet
from monai.networks.layers import Norm

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Swin_UNETR_DCA import SwinUNETR_DeepSupervision_DCA
import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

def inference_withLabel():
    case_num = 0

    model.eval()
    with torch.no_grad():
        img = test_ds[case_num]["image"]
        label = test_ds[case_num]["label"]
        test_inputs = torch.unsqueeze(img, 1).cuda()
        test_labels = torch.unsqueeze(label, 1).cuda()
        test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), 4, model, overlap=0.8)

        # 將 CUDA 張量轉換為 CPU 張量，並轉換為 NumPy 陣列
        test_inputs = test_inputs.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        test_outputs = test_outputs.cpu().numpy()

        test_inputs = zoom(test_inputs, (1, 1, 1, 1, 4/151), order=3)
        test_labels = zoom(test_labels, (1, 1, 1, 1, 4/151), order=3)
        test_outputs = zoom(test_outputs, (1, 1, 1, 1, 4/151), order=3)

        dir_path = f'/home/siplab2/sophia/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/newData_6/s002/inference'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 將數據存儲到文件中
        np.save(dir_path + '/test_inputs.npy', test_inputs)
        np.save(dir_path + '/test_labels.npy', test_labels)
        np.save(dir_path + '/test_outputs.npy', test_outputs)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        
def inf(dataset, model_name, slice_num, image_name):
    
    print("model_name:", model_name)
    print("slice_num:", slice_num)
    
    # ------------- 影像存檔的路徑 ------------------------------

    if model_name == "DCA":
        distance_dir_path = f'/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/inference/lung/{image_name}'
        create_directory(distance_dir_path) 
    elif model_name == "3D-UNet":
        distance_dir_path = f'/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/inference/lung/{image_name}'
        create_directory(distance_dir_path)
    elif model_name == "SwinUNETR":
        distance_dir_path = f'/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/inference/lung/{image_name}'
        create_directory(distance_dir_path)

    # ------------ 資料集的路徑 --------------------
    datasets_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/dataset_Coronal_test.json" # 資料集的路徑  
    val_files = load_decathlon_datalist(datasets_path, True, "validation")
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=12, cache_rate=1.0, num_workers=2)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    # ------------ 模型的路徑 --------------------
    if model_name == "DCA":
        model_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Coronal)/DCA/Coronal_best_metric_model(seed=0).pth"
        # model_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Axial)/DCA/Axial_best_metric_model(seed=0).pth"
    elif model_name == "3D-UNet":
        model_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Coronal)/3D U-Net/best_metric_model_Coronal(seed=3).pth"
        # model_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Axial)/3D U-Net/0506_best_metric_model_Axial(seed=0).pth"
    elif model_name == "SwinUNETR":
        model_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Coronal)/Swin UNETR/Coronal_best_metric_model(seed=0).pth"
        # model_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/result (Axial)/Swin UNETR/epoch_20000/Axial_best_metric_model(seed=0).pth"

    # ------------ 模型載入 --------------------
    if dataset == "BTCV":
        if model_name == "DCA":
            model = SwinUNETR_DeepSupervision_DCA(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=14,
                feature_size=48,
                use_checkpoint=False,
            ).to(device)
        elif model_name == "SwinUNETR":
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=14, # 4改成2(因為分兩類)
                feature_size=48,
                use_checkpoint=False,
            ).to(device)
        elif model_name == "3D-UNet":
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=14,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ).to(device)
    elif dataset == "Lung":
        if model_name == "SwinUNETR":
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=2, # 4改成2(因為分兩類)
                feature_size=48,
                use_checkpoint=False,
            ).to(device)
        elif model_name == "DCA":
            model = SwinUNETR_DeepSupervision_DCA(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=False,
            ).to(device)
        elif model_name == "3D-UNet":
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ).to(device)

    # slice_map = {
    #     "img0035.nii.gz": 170,
    #     "img0036.nii.gz": 230,
    #     "img0037.nii.gz": 204,
    #     "img0038.nii.gz": 204,
    #     "img0039.nii.gz": 204,
    #     "img0040.nii.gz": 180,
    # }

    case_num = 0
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).to(device)
        val_labels = torch.unsqueeze(label, 1).to(device)
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
        
        val_inputs = val_inputs.cpu().numpy()
        val_labels = val_labels.cpu().numpy()
        
        if model_name == "DCA":
            val_outputs = val_outputs[0].cpu().numpy()
        else:
            val_outputs = val_outputs.cpu().numpy()           
        
                
        val_inputs = zoom(val_inputs, (1, 1, 1, 1, 4/img.shape[3]), order=3)
        val_labels = zoom(val_labels, (1, 1, 1, 1, 4/img.shape[3]), order=3)
        val_outputs = zoom(val_outputs, (1, 1, 1, 1, 4/img.shape[3]), order=3)
        
        # 將縮放後的數據轉換回 tensor
        val_inputs = torch.tensor(val_inputs, dtype=torch.float32).to(device)
        val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)
        val_outputs = torch.tensor(val_outputs, dtype=torch.float32).to(device)

        # # 第一個子圖：image
        # plt.figure()  # 調整圖像大小和DPI
        # image_slice = val_inputs.cpu().numpy()[0, 0, :, :, slice_num]
        
        # # 將影像旋轉90度
        # image_slice_rotated = np.rot90(image_slice, k=-1)  # 逆時針旋轉90度，即順時針旋轉270度
        
        # plt.imshow(image_slice_rotated, cmap="gray")
        # plt.axis('off')  # 移除坐標軸        
        # plt.savefig(os.path.join(distance_dir_path, "res_image.png"), bbox_inches='tight', pad_inches=0)
        # plt.close()

        # 第二個子圖：label
        # plt.figure()  # 調整圖像大小和DPI
        # image_slice = val_inputs.cpu().numpy()[0, 0, :, :, slice_num]
        # label_slice = val_labels.cpu().numpy()[0, 0, :, :, slice_num]
        
        # # 將影像和標籤旋轉90度
        # image_slice_rotated = np.rot90(image_slice, k=-1)  # 逆時針旋轉90度，即順時針旋轉270度
        # label_slice_rotated = np.rot90(label_slice, k=-1)
        
        # masked_label_slice_rotated = np.ma.masked_where(label_slice_rotated == 0, label_slice_rotated)  # 將背景設置為透明

        # # 創建一個包含透明度的顏色映射
        # colors = [(1, 0, 0, 0.5)]  # 紅色漸變
        # cmap = ListedColormap(colors, name='custom_cmap')
        
        # plt.imshow(image_slice_rotated, cmap="gray")  # 如果需要同時顯示圖像和標籤
        # plt.imshow(masked_label_slice_rotated, cmap=cmap)
        # plt.axis('off')  # 移除坐標軸
        # plt.savefig(os.path.join(distance_dir_path, "Ground_Truth.png"), bbox_inches='tight', pad_inches=0)
        # plt.close()

        # # label的各個器官
        # for organ_id in range(1, 14):  # 假设器官ID从1到13
        #     organ_slice = (label_slice == organ_id).astype(np.uint8)  # 获取当前器官的分割结果
    
        #     # 检查该器官是否在预测中存在
        #     if np.sum(organ_slice) > 0:
        #         plt.figure()  # 調整圖像大小和DPI
        #         plt.axis('off')  # 移除坐標軸
        #         plt.imshow(organ_slice)  # 显示当前器官的分割结果
        #         # plt.title(f"Organ {organ_id}")  # 设置标题，显示器官ID
        #         plt.savefig(os.path.join(distance_dir_path, f"label/res_label_{organ_id}.png"), bbox_inches='tight', pad_inches=0)
        #         plt.close() 
        #     else:
        #         pass

        # ------------------------------------------------------------------------------------------------
        # 第三個子圖：output
        plt.figure()  # 調整圖像大小和DPI
        # plt.title(model_name)
        image_slice = val_inputs.cpu().numpy()[0, 0, :, :, slice_num]
        
        if model_name == "DCA":
            label_slice = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num]
        elif model_name == "3D-UNet":
            label_slice = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num]   
        elif model_name == "SwinUNETR":
            label_slice = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num]   
        masked_label_slice = np.ma.masked_where(label_slice == 0, label_slice)  # 將背景設置為透明
        
        # 將影像和標籤旋轉90度
        image_slice_rotated = np.rot90(image_slice, k=-1)  # 逆時針旋轉90度，即順時針旋轉270度
        masked_label_slice_rotated = np.rot90(masked_label_slice, k=-1)
        
        # 創建一個包含透明度的顏色映射
        colors = [(1, 0, 0, 0.5)]  # 紅色漸變
        cmap = ListedColormap(colors, name='custom_cmap')
        
        plt.imshow(image_slice_rotated, cmap="gray")
        plt.imshow(masked_label_slice_rotated, cmap=cmap) #
        plt.axis('off')  # 移除坐標軸
        plt.savefig(os.path.join(distance_dir_path, f"{model_name}_output.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # # predict的各個器官
        # for organ_id in range(1, 14):  # 假设器官ID从1到13
        #     organ_slice = (label_slice == organ_id).astype(np.uint8)  # 获取当前器官的分割结果
    
        #     # 检查该器官是否在预测中存在
        #     if np.sum(organ_slice) > 0:
        #         plt.figure()  # 調整圖像大小和DPI
        #         plt.axis('off')  # 移除坐標軸
        #         plt.imshow(organ_slice)  # 显示当前器官的分割结果
        #         # plt.title(f"Organ {organ_id}")  # 设置标题，显示器官ID
        #         plt.savefig(os.path.join(distance_dir_path, f"output/res_output_{organ_id}.png"), bbox_inches='tight', pad_inches=0)
        #         plt.close()  
        #     else:
        #         pass
        # for organ_id in range(1, 14):  # 假设器官ID从1到13
        #     organ_slice = (label_slice == organ_id).astype(np.uint8)  # 获取当前器官的分割结果
        #     plt.imshow(organ_slice)  # 显示当前器官的分割结果
        #     plt.title(f"Organ {organ_id}")  # 设置标题，显示器官ID
        #     plt.savefig(distance_dir_path + f"/res_output_{organ_id}.png")  

        plt.close()
if __name__ == '__main__':
    
    # for slice_num in range(100, 260, 5):
    #     inf("BTCV", "3D-UNet", slice_num,  f"img00040-{slice_num}")
    #     inf("BTCV", "SwinUNETR", slice_num,  f"img00040-{slice_num}")
    #     inf("BTCV", "DCA", slice_num,  f"img00040-{slice_num}")
    # slice_num = 1
    for slice_num in range(0, 4):
        inf("Lung", "3D-UNet", slice_num,  f"12_3_s004-{slice_num}")
        inf("Lung", "SwinUNETR", slice_num,  f"12_3_s004-{slice_num}")
        inf("Lung", "DCA", slice_num,  f"12_3_s004-{slice_num}")
    
