import os
from pathlib import Path
import numpy as np
import random
import pickle
import json
from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)

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

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir, peta_dir, pa100k_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = EasyDict()
    dataset.description = 'combined'
    dataset.root = "/data/pantengteng"
    dataset.attr_name = attr_list

    peta_dataset = pickle.load(open(peta_dir + "/dataset.pkl", 'rb+'))
    pa100k_dataset = pickle.load(open(pa100k_dir + "/dataset.pkl", 'rb+'))
    dataset.image_name = [os.path.join(peta_dataset.root, img_name) for img_name in peta_dataset.image_name] + [os.path.join(pa100k_dataset.root, img_name) for img_name in pa100k_dataset.image_name]
    dataset.label = np.concatenate((peta_dataset.label, pa100k_dataset.label),0)

    # 拆分数据集
    dataset.partition = EasyDict()
    # dataset.partition.train = []
    # dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    pa100k_trainval_indice = pa100k_dataset.partition.trainval + peta_dataset.label.shape[0]
    trainval = np.concatenate((peta_dataset.partition.trainval, pa100k_trainval_indice),0)
    indices = np.random.permutation(trainval)
    dataset.partition.trainval = indices
    pa100k_test_indice = pa100k_dataset.partition.test + peta_dataset.label.shape[0]
    dataset.partition.test = np.concatenate((peta_dataset.partition.test, pa100k_test_indice),0)

    with open(os.path.join(save_dir, 'test_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/combined'
    make_dir(save_dir)
    peta_dir = '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PETA'
    pa100k_dir = '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/PA100k'
    generate_data_description(save_dir, peta_dir, pa100k_dir)
