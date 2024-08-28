import nibabel as nib
import numpy as np

# 加载 .npy 文件
data = np.load('/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset1/test0406/test_inputs_6_s002.nii.npy')

# 将数据转换为 NIfTI 格式的图像对象
img = nib.Nifti1Image(data, np.eye(4))  # 使用单位矩阵作为仿射矩阵

# 保存为 NIfTI 文件
nib.save(img, '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset1/test0406/nii/test_inputs_6_s002.nii.gz')