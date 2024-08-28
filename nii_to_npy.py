import nibabel as nib
import numpy as np

def convert_nii_to_npy(nii_file_path, npy_file_path):
    # 讀取 .nii 檔案
    nii_image = nib.load(nii_file_path)
    
    # 將 .nii 檔案的數據轉換為 NumPy 數組
    image_data = nii_image.get_fdata()
    
    # 儲存數組到 .npy 檔案
    np.save(npy_file_path, image_data)
    
    print(f"檔案已轉換並儲存為 {npy_file_path}")

# 使用範例
nii_file_path = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset1/augment/Axial/image/6_s002.nii'  # 修改此處為你的 .nii 檔案路徑
npy_file_path = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset1/augment/Axial/image/6_s002.npy'  # 修改此處為你希望儲存的 .npy 檔案路徑

convert_nii_to_npy(nii_file_path, npy_file_path)
