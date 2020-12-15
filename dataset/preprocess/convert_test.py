import os
from pathlib import Path
import numpy as np
import pickle
import json
from easydict import EasyDict
from scipy.io import loadmat

POSITIVE = 1
NEGATIVE = 0

attr_list = ['Female', 
                    'AgeLess16', 'Age17-45', 'Age46-60', 'Ageover60', 
                    'Front', 'Side', 'Back', 
                    'a-Backpack', 'a-ShoulderBag', 
                    'hs-Hat', 
                    'hs-Glasses', 
                    'ub-ShortSleeve', 'ub-LongSleeve', 
                    'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Coat', 
                    'ub-Black', 'ub-Blue', 'ub-Brown', 'ub-Green', 'ub-Grey', 'ub-Orange', 'ub-Pink', 'ub-Purple', 'ub-Red', 'ub-White', 'ub-Yellow', 
                    'lb-LongTrousers', 'lb-Shorts', 'lb-ShortSkirt', 'lb-Dress', 
                    'lb-Black', 'lb-Blue', 'lb-Brown', 'lb-Green', 'lb-Grey', 'lb-Orange', 'lb-Pink', 'lb-Purple', 'lb-Red', 'lb-White', 'lb-Yellow', 
                    ]

# STD是TEST的子集，skirt类别融合到Dress中。

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = EasyDict()
    dataset.description = 'test'
    dataset.root = save_dir
    dataset.attr_name = attr_list
    dataset.image_name = []

    org_json = "/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/test/testLabel.json"
    with open(org_json, "r") as f:
        org_label_list = json.load(f)
    index_list = [data["img_name"] for data in org_label_list]
    img_list = (Path(save_dir) / "images").glob("*")
    label_list = []
    for id, img in enumerate(img_list):
        dataset.image_name.append(str(img))
        org_label = org_label_list[index_list.index(img.name)]
        label = []
        for index, att in enumerate(attr_list):
            # test有对应标签
            if att in org_label["attribute"]:
                label.append(POSITIVE)
            else:
                label.append(NEGATIVE)
        label_list.append(label)
    dataset.label = np.array(label_list)

    # 拆分数据集
    dataset.partition = EasyDict()
    dataset.partition.test = np.arange(len(label_list))

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/test'

    generate_data_description(save_dir)
