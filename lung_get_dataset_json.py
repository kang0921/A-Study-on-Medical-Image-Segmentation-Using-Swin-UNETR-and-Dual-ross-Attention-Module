import os
import json
import pprint

# dataset_json = {
#     "labels": {
#         "0": "background",
#         "1": "spleen",
#         "2": "rkid",
#         "3": "lkid",
#         "4": "gall",
#         "5": "eso",
#         "6": "liver",
#         "7": "sto",
#         "8": "aorta",
#         "9": "IVC",
#         "10": "veins",
#         "11": "pancreas",
#         "12": "rad",
#         "13": "lad"
#     },
#     "name": "btcv",
#     "numTraining": 66,
#     "numValidation": 16,
#     "tensorImageSize": "3D",
#     "training": [],
#     "validation": [],
#     "test": []
# }
dataset_json = {
    "labels": {
        "0": "Background",
        "1": "Pulmonary Fibrosis"
    },
    "tensorImageSize": "3D",
    "training": [],
    "validation": [],
    "test": []
}

def get_train_dataset_json(img_folder, label_folder, datasets, num):
    
    img_paths = sorted(os.path.join(img_folder, file) for file in os.listdir(img_folder))
    label_paths = sorted(os.path.join(label_folder, file) for file in os.listdir(label_folder))

    for img_path, label_path in zip(img_paths[:-1*num], label_paths[:-1*num]):
        image_filename = os.path.split(img_path)[-1]
        label_filename = os.path.split(label_path)[-1]
        dataset_json["training"].append({
            "image": f'{img_folder}/{image_filename}',
            "label": f'{label_folder}/{label_filename}',
        })

    for img_path, label_path in zip(img_paths[-1*num:], label_paths[-1*num:]):
        image_filename = os.path.split(img_path)[-1]
        label_filename = os.path.split(label_path)[-1]
        dataset_json["validation"].append({
            "image": f'{img_folder}/{image_filename}',
            "label": f'{label_folder}/{label_filename}',
        })
    
    with open(datasets, 'w') as outfile:
        json.dump(dataset_json, outfile)

    print(f"num of training：{len(dataset_json['training'])}")
    print(f"num of validation：{len(dataset_json['validation'])}")

    pprint.pprint(dataset_json)

def get_test_dataset_json(img_folder, datasets):

    img_paths = sorted(os.path.join(img_folder, file) for file in os.listdir(img_folder))
    for img_path in img_paths:
        image_filename = os.path.split(img_path)[-1]
        dataset_json["test"].append({
            "image": f'{img_folder}/{image_filename}',
        })

    with open(datasets, 'w') as outfile:
        json.dump(dataset_json, outfile)

    pprint.pprint(dataset_json)
if __name__ == '__main__':

    img_folder = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/data_augmentation/Coronal/image'
    label_folder = '/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/data_augmentation/Coronal/label'
    datasets = "/home/siplab5/Swin-UNETR-LeonidAlekseev/Swin-UNETR/data/dataset3/data_augmentation/dataset_Coronal.json"
    num = 5
    get_train_dataset_json(img_folder, label_folder, datasets, num)