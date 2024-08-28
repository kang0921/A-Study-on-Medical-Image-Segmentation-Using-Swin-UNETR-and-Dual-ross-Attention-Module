import torch
import SimpleITK as sitk
import glob
import re

def extract_number(filename):
    """
    從文件名中提取數字
    """
    return int(re.findall(r'\d+', filename)[-1])  # 提取文件名中的數字部分

def dice(predict, soft_y):
    """
    get dice scores for each class in predict and soft_y
    """
    num_class = predict.size(1)
    dice_scores = []
    for i in range(1, num_class):
        predict_class = predict[:, i]
        soft_y_class = soft_y[:, i]
        
        y_vol = soft_y_class.sum()
        p_vol = predict_class.sum()
        sop = soft_y_class * predict_class
        intersect = sop.sum()
        dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
        dice_scores.append(dice_score.item())
    
    total_dice = sum(dice_scores) / num_class
    
    return total_dice

if __name__ == "__main__":

    label_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/label"
    infer_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/output"
    
    label_files = sorted(glob.glob(label_path + "/*"), key=extract_number)  # 按數字排序
    infer_files = sorted(glob.glob(infer_path + "/*"), key=extract_number)  # 按數字排序
    
    score_avg = 0
    
    for label_file, infer_file in zip(label_files, infer_files):
        
        lab = sitk.ReadImage(label_file, sitk.sitkFloat32)
        inf = sitk.ReadImage(infer_file, sitk.sitkFloat32)
        
        lab = sitk.GetArrayFromImage(lab)
        inf = sitk.GetArrayFromImage(inf)
        
        lab = (lab - lab.min()) / (lab.max() - lab.min())  # 正規化
        inf = (inf - inf.min()) / (inf.max() - inf.min())  # 正規化
        
        lab, inf = torch.from_numpy(lab), torch.from_numpy(inf)

        score = dice(inf, lab)
        
        print(f"Image {label_file}: Dice Score - {score:.4f}")
        score_avg += score
        
    score_avg /= len(label_files)
    print(f"Average Dice Score: {score_avg:.4f}")


# import torch
# import SimpleITK as sitk
# from glob import glob

# def dice(predict, soft_y):
#     """
#     get dice scores for each class in predict and soft_y
#     """
#     num_class = predict.size(1)
#     soft_y = soft_y.view(-1, num_class)
#     predict = predict.view(-1, num_class)
    
#     y_vol = soft_y.sum(dim=0)
#     p_vol = predict.sum(dim=0)
#     sop = soft_y * predict
#     intersect = sop.sum(dim=0)
#     dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
#     total_dice = dice_score.mean()
    
#     return total_dice.item()

# if __name__ == "__main__":

#     label_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/res_label.png"
#     infer_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/res_output.png"
    
#     infer = sorted(glob(infer_path))
#     label = sorted(glob(label_path))
#     score_avg = 0
    
#     for i in range(len(label)):
#         inf, lab = infer[i], label[i]
#         inf, lab = sitk.ReadImage(inf, sitk.sitkFloat32), sitk.ReadImage(lab, sitk.sitkFloat32)
#         inf, lab = sitk.GetArrayFromImage(inf), sitk.GetArrayFromImage(lab)
        
#         inf = (inf - inf.min()) / (inf.max() - inf.min())  # 正規化
#         lab = (lab - lab.min()) / (lab.max() - lab.min())  # 正規化
        
#         inf, lab = torch.from_numpy(inf), torch.from_numpy(lab)
#         score = dice(inf, lab)
        
#         print(f"Image {i+1}: Dice Score - {score:.4f}")
#         score_avg += score
        
#     score_avg /= len(label)
#     print(f"Average Dice Score: {score_avg:.4f}")


# # import torch
# # from PIL import Image
# # import numpy as np

# # def load_image(image_path):
# #     """
# #     Load an image from disk and convert it to a PyTorch tensor.
# #     """
# #     # 使用PIL庫打開圖像
# #     image = Image.open(image_path)
    
# #     # 將PIL圖像轉換為NumPy數組
# #     image_np = np.array(image)
    
# #     # 將NumPy數組轉換為PyTorch張量
# #     image_tensor = torch.tensor(image_np)
    
# #     return image_tensor

# # def dice_coefficient(predict, target):
# #     """
# #     Calculate the Dice coefficient for two binary segmentation masks.
# #     """
# #     smooth = 1e-5
    
# #     # 將背景像素設置為0
# #     predict = predict.clone()  # 創建一個預測圖像的副本，以避免原始張量被修改
# #     target = target.clone()    # 創建一個標籤圖像的副本，以避免原始張量被修改
# #     predict[predict == 0] = 0
# #     target[target == 0] = 0
    
# #     # 計算交集和聯集
# #     intersection = torch.sum(predict * target)
# #     union = torch.sum(predict) + torch.sum(target)
    
# #     # 計算Dice係數
# #     dice = (2. * intersection + smooth) / (union + smooth)
# #     return dice.item()

# # if __name__ == "__main__":
# #     # 讀取標籤圖像和預測圖像
# #     label_image_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/res_label.png"
# #     predict_image_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/res_output.png"
    
# #     # 讀取圖像並轉換為PyTorch張量
# #     label_image = load_image(label_image_path)
# #     predict_image = load_image(predict_image_path)
    
# #     # 確保圖像具有相同的形狀
# #     assert label_image.shape == predict_image.shape, "Shape mismatch between label and predict images"
    
# #     # 計算Dice係數
# #     dice = dice_coefficient(label_image, predict_image)
# #     print(f"Dice coefficient: {dice:.4f}")

# import torch
# import SimpleITK as sitk
# from glob import glob

# def dice(predict, soft_y):
#     """
#     get dice scores for each class in predict and soft_y
#     """
#     num_class = predict.size(1)
#     soft_y = soft_y.view(-1, num_class)
#     predict = predict.view(-1, num_class)
    
#     y_vol = soft_y.sum(dim=0)
#     p_vol = predict.sum(dim=0)
#     sop = soft_y * predict
#     intersect = sop.sum(dim=0)
#     dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
#     total_dice = dice_score.mean()
    
#     return total_dice.item()

# if __name__ == "__main__":

#     label_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/res_label.png"
#     infer_path = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/BTCV/inference/3D-Unet/img0035_withoutBackground/res_output.png"
    
#     infer = sorted(glob(infer_path))
#     label = sorted(glob(label_path))
#     score_avg = 0
    
#     for i in range(len(label)):
#         inf, lab = infer[i], label[i]
#         inf, lab = sitk.ReadImage(inf, sitk.sitkFloat32), sitk.ReadImage(lab, sitk.sitkFloat32)
#         inf, lab = sitk.GetArrayFromImage(inf), sitk.GetArrayFromImage(lab)
        
#         inf, lab = torch.from_numpy(inf), torch.from_numpy(lab)
#         score = dice(inf, lab)
        
#         print(f"Image {i+1}: Dice Score - {score:.4f}")
#         score_avg += score
        
#     score_avg /= len(label)
#     print(f"Average Dice Score: {score_avg:.4f}")
