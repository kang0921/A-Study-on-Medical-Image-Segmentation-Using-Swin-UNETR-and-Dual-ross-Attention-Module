import os
import SimpleITK as sitk
from monai.transforms import Compose, Resize
import nibabel as nib
from scipy.ndimage import zoom

def augment_folder(data_folder, output_folder):

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 列出資料夾中的所有.nii.gz檔案
    file_list = [f for f in os.listdir(data_folder) if f.endswith(".nii")]
    print(file_list)

    # 對每個檔案應用資料擴增
    for file_name in file_list:
        # 讀取.nii.gz檔案
        img_path = os.path.join(data_folder, file_name)
        image = nib.load(img_path).get_fdata()
        
        # 定義擴增的比例
        zoom_factor = (512/image.shape[0], 512/image.shape[1], 301 / image.shape[2])
        
        # 使用 zoom 函數執行線性插值
        image = zoom(image, zoom_factor, order=1)

        print("image.shape: ", image.shape)

        img_nifti = nib.Nifti1Image(image, affine=None)  # 若有影像的仿射矩陣，可填入 affine 參數

        output_file_path = os.path.join(output_folder, file_name)
        nib.save(img_nifti, output_file_path)   
        
        print(f"Saved transformed image to: {output_file_path}\n")

def augment(img_path, output_folder, file_name):
    image = nib.load(img_path).get_fdata()
    
    # 定義擴增的比例
    zoom_factor = (512/image.shape[0], 512/image.shape[1], 301 / image.shape[2])
    print("before image.shape: ", image.shape)
    # 使用 zoom 函數執行線性插值
    image = zoom(image, zoom_factor, order=1)

    print("after image.shape: ", image.shape)

    img_nifti = nib.Nifti1Image(image, affine=None)  # 若有影像的仿射矩陣，可填入 affine 參數

    output_file_path = os.path.join(output_folder, file_name)
    nib.save(img_nifti, output_file_path)   
    
    print(f"Saved transformed image to: {output_file_path}\n")

if __name__ == '__main__':

    data_folder = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/Coronal/label'
    output_folder = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/data_augmentation/Coronal/label'
    augment_folder(data_folder, output_folder)
