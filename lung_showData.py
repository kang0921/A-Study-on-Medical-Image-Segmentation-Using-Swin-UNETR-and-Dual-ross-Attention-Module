import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.util import montage
import os
import glob
import pprint
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom

def show_data(img_path, label_path, output_name, slice):

    # load NIfTI data
    test_image = nib.load(img_path).get_fdata()
    test_label = nib.load(label_path).get_fdata()
    test_image = np.transpose(test_image, (2, 1, 0))
    test_label = np.transpose(test_label, (2, 1, 0))
    
    # test_image = zoom(test_image, (4/301, 1, 1), order=3)
    # test_label = zoom(test_label, (4/301, 1, 1), order=3)
    
    # 將影像和標籤旋轉90度
    test_image = np.rot90(test_image, k=-1)  # 逆時針旋轉90度，即順時針旋轉270度
    test_label = np.rot90(test_label, k=-1)


    test_label = np.ma.masked_where(test_label == 0, test_label)  # 將背景設置為透明
    
    # 創建一個包含透明度的顏色映射
    colors = [(1, 0, 0, 0.5)]  # 紅色漸變
    cmap = ListedColormap(colors, name='custom_cmap')
    
    plt.imshow(test_image[:, slice, :], cmap="gray")  # 如果需要同時顯示圖像和標籤
    plt.imshow(test_label[:, slice, :], cmap=cmap)
    plt.axis('off')  # 移除坐標軸
    plt.savefig(os.path.join("/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/inference/lung/", output_name), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # # 顯示單一切面的影像和標籤
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.imshow(test_image[test_image.shape[0]//2], cmap='gray')
    # ax1.set_title(f'Image shape: {test_image.shape}')
    # ax2.imshow(test_label[test_label.shape[0]//2])
    # ax2.set_title(f'Label shape: {test_label.shape}')
    # plt.savefig('data/output_png/' + output_name + '_slice_*.png')

    # # 以 montage 顯示整個 image 序列
    # fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
    # ax1.imshow(montage(test_image, padding_width=10, fill=1), cmap='gray')
    # plt.axis('off')
    # plt.savefig('data/output_png/' + output_name + '_montage_img_*.png')

    # # 以 montage 顯示整個 label 序列
    # fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
    # ax1.imshow(montage(test_label, padding_width=10, fill=1))
    # plt.axis('off')
    # plt.savefig('data/output_png/' + output_name + '_montage_label_*.png')

def show_data_withoutLabel(img_path, output_name, output_path):

    # load NIfTI data
    test_image = nib.load(img_path).get_fdata()
    test_image = np.transpose(test_image, (2, 1, 0))

    # 顯示單一切面的影像和標籤
    fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
    ax1.imshow(test_image[test_image.shape[0]//2], cmap='gray')
    ax1.set_title(f'Image shape: {test_image.shape}')
    plt.savefig(output_path + output_name + '_slice.png')

    # # 以 montage 顯示整個 image 序列
    fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
    ax1.imshow(montage(test_image, padding_width=10, fill=1), cmap='gray')
    plt.axis('off')
    plt.savefig(output_path + output_name + '_montage_img.png')

if __name__ == '__main__':

    img_path = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/12_3_s004_image.nii'
    label_path = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/12_3_s004_label.nii'
    for slice in range(0, 4):
        output_name = f'12_3_s004__GT_{slice}'
        show_data(img_path, label_path, output_name, slice)