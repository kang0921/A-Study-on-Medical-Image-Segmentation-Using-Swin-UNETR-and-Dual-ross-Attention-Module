import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.metrics import DiceMetric
import torch
import os

def load_data_withLabel_BTCV(slice_num, dir_path):    

    # 加載數據
    val_inputs = np.load(dir_path + 'test_inputs.npy')
    val_labels = np.load(dir_path + 'test_labels.npy')
    num_outputs = 5
    val_outputs = [np.load(dir_path + f'test_output_{i}.npy') for i in range(num_outputs)]

    # 向右旋轉90度
    val_inputs = np.rot90(val_inputs[0, 0, :, :, slice_num], k=-1)
    val_labels = np.rot90(val_labels[0, 0, :, :, slice_num], k=-1)
    # val_outputs = np.rot90(np.argmax(val_outputs, axis=1)[0, :, :, slice_num], k=-1)
    # 創建一個列表來存儲所有旋轉後的預測結果
    rotated_val_outputs = []

    # 對每個預測結果進行旋轉
    for val_output in val_outputs:
        rotated_val_output = np.rot90(np.argmax(val_output, axis=0), k=-1)
        rotated_val_outputs.append(rotated_val_output)

    # 將旋轉後的預測結果存儲在一個列表中
    rotated_val_outputs = np.array(rotated_val_outputs)

    return val_inputs, val_labels, rotated_val_outputs

def load_data_withLabel(slice_num, dir_path):    

    # 加載數據
    val_inputs = np.load(dir_path + 'val_inputs.npy')
    val_labels = np.load(dir_path + 'val_labels.npy')
    val_outputs = np.load(dir_path + 'val_outputs.npy')

    # 向右旋轉90度
    val_inputs = np.rot90(val_inputs[0, 0, :, :, slice_num], k=-1)
    val_labels = np.rot90(val_labels[0, 0, :, :, slice_num], k=-1)
    val_outputs = np.rot90(np.argmax(val_outputs, axis=1)[0, :, :, slice_num], k=-1)

    return val_inputs, val_labels, val_outputs

def load_data_withoutLabel(slice_num, dir_path, file_name):    

    # 加載數據
    test_inputs = np.load(dir_path + f'/test_inputs_{file_name}.npy')
    test_outputs = np.load(dir_path + f'/test_outputs_{file_name}.npy')

    # 向右旋轉90度
    test_inputs = np.rot90(test_inputs[0, 0, :, :, slice_num], k=-1)
    test_outputs = np.rot90(np.argmax(test_outputs, axis=1)[0, :, :, slice_num], k=-1)

    return test_inputs, test_outputs

def show_image_withLabel_BTCV(val_inputs, val_labels, val_outputs, dir_path, slice_num):
    
    # 創建一個包含透明度的顏色映射
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.9)]  # 第一個元組代表透明，第二個元組代表紅色
    cmap = ListedColormap(colors, name='custom_cmap')

    # 現在可以使用這些數據進行相關的分析、處理或視覺化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(val_inputs, cmap="gray")  # 直接從 NumPy 陣列中選取切片
    ax1.set_title('Image')
    ax1.axis('off')  # 移除坐標軸

    ax2.imshow(val_inputs, cmap="gray")
    ax2.imshow(val_labels, cmap=cmap, alpha=0.5)
    ax2.set_title(f'Label')
    ax2.axis('off')  # 移除坐標軸

    ax3.imshow(val_inputs, cmap="gray")
    for rotated_val_output in val_outputs:
        ax3.imshow(rotated_val_output, cmap=cmap, alpha=0.5)
    ax3.set_title(f'Predict')
    ax3.axis('off')  # 移除坐標軸

    plt.savefig(dir_path + f'result_slice{slice_num}.png')
    plt.close()

def show_image_withLabel(val_inputs, val_labels, val_outputs, dir_path, slice_num):
    
    # 創建一個包含透明度的顏色映射
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.9)]  # 第一個元組代表透明，第二個元組代表紅色
    cmap = ListedColormap(colors, name='custom_cmap')

    # 現在可以使用這些數據進行相關的分析、處理或視覺化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(val_inputs, cmap="gray")  # 直接從 NumPy 陣列中選取切片
    ax1.set_title('Image')
    ax1.axis('off')  # 移除坐標軸

    ax2.imshow(val_inputs, cmap="gray")
    ax2.imshow(val_labels, cmap=cmap, alpha=0.5)
    ax2.set_title(f'Label')
    ax2.axis('off')  # 移除坐標軸

    ax3.imshow(val_inputs, cmap="gray")
    ax3.imshow(val_outputs, cmap=cmap, alpha=0.5)
    ax3.set_title(f'Predict')
    ax3.axis('off')  # 移除坐標軸

    plt.savefig(dir_path + f'result_slice{slice_num}.png')
    plt.close()

def show_image_withoutLabel(val_inputs, val_outputs, dir_path, slice_num, file_name):
    
    # 創建一個包含透明度的顏色映射
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.9)]  # 第一個元組代表透明，第二個元組代表紅色
    cmap = ListedColormap(colors, name='custom_cmap')

    # 現在可以使用這些數據進行相關的分析、處理或視覺化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.imshow(val_inputs, cmap="gray", interpolation='bilinear')  # 设置插值方法为双线性插值
    
    # ax1.imshow(val_inputs, cmap="gray")  # 直接從 NumPy 陣列中選取切片
    ax1.set_title('Image')
    ax1.axis('off')  # 移除坐標軸

    ax2.imshow(val_inputs, cmap="gray", interpolation='bilinear')
    ax2.imshow(val_outputs, cmap=cmap, alpha=0.5, interpolation='bilinear')
    # ax2.imshow(val_inputs, cmap="gray")
    # ax2.imshow(val_outputs, cmap=cmap, alpha=0.5)
    ax2.set_title(f'Predict')
    ax2.axis('off')  # 移除坐標軸

    plt.savefig(dir_path + '/output_png/' + f'{file_name}_{slice_num}.png')
    plt.close()

if __name__ == '__main__':

    ''' BTCV '''
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    file_name = "img0035.nii.gz"
    slice_num = slice_map[file_name]

    dir_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/our/case0/"
    

    val_inputs, val_labels, rotated_val_outputs = load_data_withLabel_BTCV(slice_num, dir_path)
    show_image_withLabel_BTCV(val_inputs, val_labels, rotated_val_outputs, dir_path, slice_num)

    # dir_path = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset1/test0406/'
    # filename_path = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset1/augment/Axial/image'

    # # 獲取目錄下的所有文件和目錄名
    # entries = os.listdir(filename_path)

    # # 篩選出所有的文件名
    # file_names = [entry for entry in entries if os.path.isfile(os.path.join(filename_path, entry))]

    # # 打印所有文件名
    # for file_name in file_names:
    #     print(file_name)
    #     for slice_num in range(0, 4):
    #         teset_inputs, teset_outputs = load_data_withoutLabel(slice_num, dir_path, file_name)
    #         show_image_withoutLabel(teset_inputs, teset_outputs, dir_path, slice_num, file_name)
