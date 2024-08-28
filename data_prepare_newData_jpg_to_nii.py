import os
import numpy as np
import nibabel as nib
from PIL import Image

# type = "Axial"
type = "Coronal"

folder = f"/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset2/original/{type}"
distance_folder = f"/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset2/xyz=512_512_32/{type}"


contents = os.listdir(folder)   # 獲取資料夾下的所有檔案和資料夾
subfolders = [f for f in contents if os.path.isdir(os.path.join(folder, f))] # 篩選出資料夾名稱


for subfolder in subfolders: # 打印所有資料夾名稱
    
    # 載入所有.jpg檔案並將它們轉換為numpy陣列
    image_folder = f'{folder}/{subfolder}'
    jpg_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

    # 創建一個空的3D數據集，根據您的數據集大小調整shape
    # 這裡假設所有的jpg影像大小相同
    image_shape = (512, 512)  # 將影像調整為512x512
    num_slices = len(jpg_files)
    data = np.zeros((image_shape[0], image_shape[0], num_slices), dtype=np.uint8)

    # 將.jpg檔案轉換為numpy數據並堆疊成3D數據集
    for i, file in enumerate(jpg_files):
        img = Image.open(os.path.join(image_folder, file)).convert('L')  # 轉換為灰度圖像
        img = img.resize(image_shape, Image.LANCZOS)  # 調整影像大小為512x512
        data[:, :, i] = np.array(img)

    # 創建nifti影像對象並保存為.nii檔案
    nifti_img = nib.Nifti1Image(data, np.eye(4))  # 使用單位矩陣創建空間信息
    nii_file_path = f'{distance_folder}/{subfolder}.nii'
    nib.save(nifti_img, nii_file_path)