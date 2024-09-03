# 結合Swin UNETR與雙重交叉注意力模組於醫學影像分割之研究
## 國立中興大學資訊工程學系碩士學位論文
- 論文題目：結合Swin UNETR與雙重交叉注意力模組於醫學影像分割之研究
- 論文英文題目：A Study on Medical Image Segmentation Using Swin UNETR and Dual Cross-Attention Module
- 指導教授：吳俊霖教授
- 研究生：康智絜
- 口試日期：中華民國113年7月31日

## 摘要
- 醫學影像在病情診斷中至關重要。然而，人工解讀這些影像需要專業的醫學知識，且耗時費力，長時間工作可能會降低診斷的準確性。因此，自動化的影像分割技術可以減輕醫生的工作負擔，提高診斷的精準度。
- 近年來，基於U型架構的深度卷積神經網路在醫學影像分割領域取得了顯著進展。然而，卷積運算的局部性限制了其捕捉全局和長距離特徵的能力。為了解決這一問題，基於自注意力機制的Transformers技術逐漸被引入影像處理領域。Swin UNETR是一種結合Swin Transformer和U型架構的模型，其中Swin Transformer用於提取特徵。然而，這種模型的編碼器和解碼器之間的簡單跳躍連接造成了語意差距，影響了全局多尺度上下文的捕捉效果。
- 因此，本研究提出將Swin UNETR與雙重交叉注意力模組相結合的方法。雙重交叉注意力模組能夠在通道維度和空間維度上建立更精確的特徵之間的關聯，有效縮小語意差距，從而提升模型在不同尺度目標分割任務中的表現。
- 在公開的腹腔器官分割資料集上，本研究提出的方法的Dice coefficient達到85.13%，相比主要比較方法提升了1.84%，同時整體正確率也達到了85.13%。在肺部纖維化資料集中，我們的方法在水平面和冠狀面的分割結果上，Dice coefficient分別提高了2.84%和0.85%，且水平面整體正確率為89.03%，冠狀面整體正確率則為89.57%。這些結果充分證明了我們提出方法的有效性和在醫學影像分割任務中的潛在應用價值。

> 關鍵字：醫學影像、深度學習、器官分割、Swin UNETR、雙重交叉注意力模組

## Datasets
#### [腹腔器官分割資料集(BTCV)](https://www.synapse.org/Synapse:syn3193805/wiki/217789)
- 該資料集包含30位受試者的腹部電腦斷層掃描(CT)數據
- 每位受試者的CT掃描包含80到225層切片，每層512×512像素，切片厚度範圍從1到6毫米不等。
- 資料集由美國范德堡大學醫學院暨醫學中心的臨床放射科醫生監督標記13種不同35的器官，包括脾臟、右腎、左腎、膽囊、食道、肝臟、胃、主動脈、下腔靜脈、門靜脈和脾靜脈、胰腺、右腎上腺和左腎上腺。
- 我們使用其中24位受試者的數據作為訓練資料，剩餘的6位受試者數據則用於測試。

#### 肺部纖維化數據集(由台中榮總傅斌貴主任提供)
- 本研究使用的資料集由臺中榮民總醫院傅彬貴醫師提供
- 包含兩種不同切面方向的肺部電腦斷層掃描影像。影像分別為水平面(Axial)和冠狀面(Coronal)的肺部電腦斷層掃描。
- 水平面方向電腦斷層掃描影像包含188張訓練資料和48張測試資料。
- 冠狀面方向電腦斷層掃描影像則包含108張訓練資料和28張測試資料。

## 模型架構
![本研究之網路架構示意圖](https://github.com/kang0921/A-Study-on-Medical-Image-Segmentation-Using-Swin-UNETR-and-Dual-ross-Attention-Module/blob/main/assets/%E6%9C%AC%E7%A0%94%E7%A9%B6%E4%B9%8B%E7%B6%B2%E8%B7%AF%E6%9E%B6%E6%A7%8B%E7%A4%BA%E6%84%8F%E5%9C%96.png)

## 實驗結果
### 腹腔器官分割資料集(BTCV)
- BTCV的平均評估結果
| Methods | Average Dice Coefficient | 
|---------|--------------------------|
| 3D U-Net |  81.71% |
| Swin UNETR | 82.89% |
| Proposed Method | 85.13% |

- BTCV各個器官的評估結果
| Methods | Liver |  Sto | L-Kid | Aorta | R-Kid | Spleen |	Pan | IVC |	Eso |	Veins | L-Ad | Gall | R-Ad |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 3D U-Net |  96.02% | 81.12% | 94.69%|76.89%|94.97%|91.79%|75.63%|79.41%|67.83%|79.40%|76.80%|75.87%|54.70%|
| Swin UNETR | 96.84% | 88.63% | 94.23%|83.75%|95.17%|94.35%|83.00%|83.40%|73.70%|82.52%|83.62%|78.65%|66.67%|
| Proposed Method | 97.18% | 92.79% | 96.11%|85.48%|96.02%|96.94%|83.69%|86.13%|77.20%|81.34%|80.85%|85.07%|65.46%|

### 肺部纖維化資料集(Axial)
| Methods | Dice Coefficient | 
|---------|--------------------------|
| 3D U-Net |  85.06% |
| Swin UNETR | 86.19% |
| Proposed Method | 89.03% |

### 肺部纖維化資料集(Coronal)
| Methods | Dice Coefficient | 
|---------|--------------------------|
| 3D U-Net |  87.56% |
| Swin UNETR | 88.72% |
| Proposed Method | 89.57% |
## 安裝環境
- environment.yml

## 資料前處理
1. 將最原始沒有做處裡的資料夾放進Available dataset的資料夾中
2. 到資料夾中將該病患各個時期的照片，分別分成3個資料夾s_002、s_003、s_004，每個資料夾放4張選好的纖維化照片
3. 到程式碼中，執行data_prepare_newData_jpg_to_nii.py，將jpg檔案轉換成.nii檔案
4. 將.nii檔案分別放到對應的資料夾
5. 將這個.nii檔案由ITK-SNAP開啟做label
6. 將image和label的檔做處理，執行data_augmentation.py，將資料的維度轉換成(512, 512, 301)

## 設定資料的json檔案
- 執行lung_get_dataset_json.py，參數分別是要做的影像的image和label檔案的資料夾路徑

## 模型訓練
1. 將參數datasets改為欲training、validation的json檔案
2. 設定max_iterations、model_path、result_png_path、set_determinism(seed=?)

## 執行的程式檔案
#### Preprocessing
- data_prepare_newData_jpg_to_nii.py
- data_augmentation.py
- lung_get_dataset_json.py

#### Train:
- lung_main.py

#### Inference:
- lung_inference.py
- lung_show_result_image.py

#### 如果要換模型，換成3D-Unet:
- 3D_UNet/lung_3dUNet.py
