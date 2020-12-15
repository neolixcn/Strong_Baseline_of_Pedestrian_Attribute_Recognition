import os
from pathlib import Path
import numpy as np
import pickle
import json
from easydict import EasyDict
from scipy.io import loadmat

# note: ref by annotation.md

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


baidu_dict = {
    'Female': {'gender': '女性'},
    # 'Male': {'gender':'男性'},
    'AgeLess16': {'age': ['幼儿', '青少年']},
    'Age17-45': {'age': '青年'},
    'Age46-60': {'age': '中年'},
    'Ageover60': {'age': '老年'},
    'Front': {'orientation': '正面'},
    'Side': {'orientation': ['左侧面', '右侧面']},
    'Back': {'orientation': '背面'},
    'a-Backpack': {'bag': '双肩包'},
    'a-ShoulderBag': {'bag': '单肩包'},
    'hs-Hat': {'headwear': '普通帽'},
    'hs-Glasses': {'glasses': '戴眼镜'},
    'ub-ShortSleeve': {'upper_wear': '短袖'},
    'ub-LongSleeve': {'upper_wear': '长袖'},
    'ub-Shirt': {'upper_wear_fg': '衬衫'},
    'ub-Sweater': {'upper_wear_fg': '毛衣'},
    'ub-Vest': {'upper_wear_fg': '无袖'},
    'ub-TShirt': {'upper_wear_fg': 'T恤'},
    'ub-Cotton': {'upper_wear_fg': '羽绒服'},
    'ub-Jacket': {'upper_wear_fg': ['外套', '夹克']},
    'ub-SuitUp': {'upper_wear_fg': '西装'},
    'ub-Coat': {'upper_wear_fg': '风衣'},  # coat这里仅指风衣
    'ub-Black': {'upper_color': '黑'},
    'ub-Blue': {'upper_color': '蓝'},
    'ub-Brown': {'upper_color': '棕'},
    'ub-Green': {'upper_color': '绿'},
    'ub-Grey': {'upper_color': '灰'},
    'ub-Orange': {'upper_color': '橙'},
    'ub-Pink': {'upper_color': '粉'},
    'ub-Purple': {'upper_color': '紫'},
    'ub-Red': {'upper_color': '红'},
    'ub-White': {'upper_color': '白'},
    'ub-Yellow': {'upper_color': '黄'},
    'lb-LongTrousers': {'lower_wear': '长裤'},
    'lb-Shorts': {'lower_wear': '短裤'},
    'lb-ShortSkirt': {'lower_wear': '短裙'},
    'lb-Dress': {'lower_wear': '长裙'},
    'lb-Black': {'lower_color': '黑'},
    'lb-Blue': {'lower_color': '蓝'},
    'lb-Brown': {'lower_color': '棕'},
    'lb-Green': {'lower_color': '绿'},
    'lb-Grey': {'lower_color': '灰'},
    'lb-Orange': {'lower_color': '橙'},
    'lb-Pink': {'lower_color': '粉'},
    'lb-Purple': {'lower_color': '紫'},
    'lb-Red': {'lower_color': '红'},
    'lb-White': {'lower_color': '白'},
    'lb-Yellow': {'lower_color': '黄'}
}

# 生成baidu的np.array
save_dir = '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/data/test'

baidu_json = "/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/baidu.json"

with open(baidu_json, "r") as f:
    db_label_list = json.load(f)

# 将名字放入index_list中
index_list = [data["img_name"] for data in db_label_list]

img_list = (Path(save_dir) / "images").glob("*")
label_list = []
dataset = EasyDict()
dataset.image_name = []

dataset1 = EasyDict()
dataset1.image_name = []

# 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
for id, img in enumerate(img_list):
    dataset.image_name.append(str(img))
    bd_label = db_label_list[index_list.index(img.name)]
    label = []
    for index, att in enumerate(attr_list):
        # 字典需要获取key
        sub_key = list(baidu_dict[att].keys())[0]
        da_att = bd_label["person_info"][0]["attributes"][sub_key]["name"]
        if isinstance(baidu_dict[att][sub_key], list):
            if da_att in baidu_dict[att][sub_key]:
                label.append(POSITIVE)
            else:
                label.append(NEGATIVE)
        else:
            if da_att == baidu_dict[att][sub_key]:
                label.append(POSITIVE)
            else:
                label.append(NEGATIVE)
    label_list.append(label)
preds_probs = np.array(label_list)
print('aaa')

# 生成真实标签的np.array
org_json = "/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/new_data.json"
with open(org_json, "r") as f:
    org_label_list = json.load(f)
index_list = [data["img_name"] for data in org_label_list]
img_list = (Path(save_dir) / "images").glob("*")
label_list = []
for id, img in enumerate(img_list):
    dataset1.image_name.append(str(img))
    org_label = org_label_list[index_list.index(img.name)]
    label = []
    for index, att in enumerate(attr_list):
        # test有对应标签
        if att in org_label["attribute"]:
            label.append(POSITIVE)
        else:
            label.append(NEGATIVE)
    label_list.append(label)

gt_label = np.array(label_list)
print("bbb")


# 得到指标
def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1),
                      axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0),
                      axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)),
                       axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)),
                       axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    # result.label_pos_recall = label_pos_recall
    # result.label_neg_recall = label_neg_recall
    # result.label_prec = true_pos / (true_pos + false_pos + eps)
    # result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    # result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
    # result.label_prec + result.label_pos_recall + eps)

    # result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum(
        (gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)),
                       axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / \
        (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    # result.error_num, result.fn_num, result.fp_num = false_pos + \
    #    false_neg, false_neg, false_pos

    print(result)


get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5)
