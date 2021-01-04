import os
from pathlib import Path
import numpy as np
import random
import pickle
import json
from easydict import EasyDict

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

# customed 2 peta
convert_dict = {
    'Female': 'personalFeMale',
    # 'Male': 'personalMale',
    'AgeLess16': 'personalLess15',
    'Age17-45': ['personalLess45', 'personalLess30'],
    'Age46-60': 'personalLess60',
    'Ageover60': 'personalLarger60',
    'Front': None,
    'Side': None,
    'Back': None,
    'a-Backpack': 'carryingBackpack',
    'a-ShoulderBag': 'carryingMessengerBag',
    'hs-Hat': 'accessoryHat',
    'hs-Glasses': None, # 'accessorySunglasses'
    'ub-ShortSleeve': 'upperBodyShortSleeve',
    'ub-LongSleeve': 'upperBodyLongSleeve',
    'ub-Shirt': None,
    'ub-Sweater': 'upperBodySweater',
    'ub-Vest': None,
    'ub-TShirt': 'upperBodyTshirt',
    'ub-Cotton': None,
    'ub-Jacket': 'upperBodyJacket',
    'ub-SuitUp': 'upperBodySuit',
    'ub-Coat': None,
    'ub-Black': 'upperBodyBlack',
    'ub-Blue': 'upperBodyBlue',
    'ub-Brown': 'upperBodyBrown',
    'ub-Green': 'upperBodyGreen',
    'ub-Grey': 'upperBodyGrey',
    'ub-Orange': 'upperBodyOrange',
    'ub-Pink': 'upperBodyPink',
    'ub-Purple': 'upperBodyPurple',
    'ub-Red': 'upperBodyRed',
    'ub-White': 'upperBodyWhite',
    'ub-Yellow': 'upperBodyYellow',
    'lb-LongTrousers': 'lowerBodyTrousers',
    'lb-Shorts': 'lowerBodyShorts',
    'lb-ShortSkirt': 'lowerBodyShortSkirt',
    'lb-Dress': 'lowerBodyLongSkirt',
    'lb-Black': 'lowerBodyBlack',
    'lb-Blue': 'lowerBodyBlue',
    'lb-Brown': 'lowerBodyBrown',
    'lb-Green': 'lowerBodyGreen',
    'lb-Grey': 'lowerBodyGrey',
    'lb-Orange': 'lowerBodyOrange',
    'lb-Pink': 'lowerBodyPink',
    'lb-Purple': 'lowerBodyPurple',
    'lb-Red': 'lowerBodyRed',
    'lb-White': 'lowerBodyWhite',
    'lb-Yellow': 'lowerBodyYellow'
}

# customed 2 baidu
baidu_dict ={
    'Female': {'gender':'女性'},
    #'Male': {'gender':'男性'},
    'AgeLess16': {'age':['幼儿', '青少年']},
    'Age17-45': {'age':'青年'},
    'Age46-60': {'age':'中年'},
    'Ageover60': {'age':'老年'},
    'Front': {'orientation':'正面'},
    'Side': {'orientation':['左侧面', '右侧面']},
    'Back': {'orientation':'背面'},
    'a-Backpack': {'bag':'双肩包'},
    'a-ShoulderBag': {'bag':'单肩包'},
    'hs-Hat': {'headwear':'普通帽'},
    'hs-Glasses': {'glasses':'戴眼镜'},
    'ub-ShortSleeve': {'upper_wear':'短袖'},
    'ub-LongSleeve': {'upper_wear':'长袖'},
    'ub-Shirt': {'upper_wear_fg':'衬衫'},
    'ub-Sweater': {'upper_wear_fg':'毛衣'},
    'ub-Vest': {'upper_wear_fg':'无袖'},
    'ub-TShirt': {'upper_wear_fg':'T恤'},
    'ub-Cotton': {'upper_wear_fg':'羽绒服'},
    'ub-Jacket': {'upper_wear_fg':['外套', '夹克']},
    'ub-SuitUp': {'upper_wear_fg':'西装'},
    'ub-Coat': {'upper_wear_fg':'风衣'},
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
    'lb-LongTrousers': {'lower_wear':'长裤'},
    'lb-Shorts': {'lower_wear':'短裤'},
    'lb-ShortSkirt': {'lower_wear':'短裙'},
    'lb-Dress': {'lower_wear':'长裙'},
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
    dataset.description = 'peta'
    dataset.root = save_dir
    dataset.attr_name = attr_list
    dataset.image_name = []

    # sub_dir = Path(save_dir).glob("*")
    sub_dir = [f for f in Path(save_dir).iterdir() if f.is_dir()]
    label_list = []
    for sub in sub_dir:
        sub = sub / "archive"
        # peta 图片格式多，不加后缀，但会把txt文件也加入列表
        img_list =sorted( list(sub.glob("*")))
        # 读取并创建原始label字典
        label_file = sub / "Label.txt"
        with open(label_file, "r") as f:
            org_label_list = f.readlines()
        label_dict = {}
        for line_label in org_label_list:
            label_split = line_label.strip().split(" ")
            label_key = label_split[0]
            label_value = [label_ for label_ in label_split[1:]]
            label_dict[label_key] = label_value
        
        # 读取并创建百度label字典
        baidu_json = sub / "bdLabel.json"
        with open(baidu_json, "r") as f:
            db_label_list = json.load(f)
        index_list = [data["img_name"] for data in db_label_list]
        
        for img in img_list:
            import pdb
            pdb.set_trace()
            if "Label" in str(img.name):
                # img_list.remove(img)
                continue
            dataset.image_name.append(str(img))
            label = []
            peta_label = label_dict[img.name.split("_")[0]]
            db_label = db_label_list[index_list.index(img.name)]
            for index, att in enumerate(attr_list):
                # peta有对应标签
                if convert_dict[att]:
                    if isinstance(convert_dict[att], list):
                        find_flag = False
                        for att_ in convert_dict[att]:
                            if att_ in peta_label:
                                label.append(POSITIVE)
                                find_flag = True
                                break
                        if not find_flag:
                            label.append(NEGATIVE)
                    else:
                        if convert_dict[att] in peta_label:
                                label.append(POSITIVE)
                        else:
                                label.append(NEGATIVE)
                # peta无对应标签
                else:
                    # 字典需要获取key
                    sub_key = list(baidu_dict[att].keys())[0]
                    da_att = db_label["person_info"][0]["attributes"][sub_key]["name"]
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

    dataset.label = np.array(label_list)

    # 拆分数据集
    dataset.partition = EasyDict()
    # dataset.partition.train = []
    # dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    indices = np.random.permutation(len(label_list))
    training_idx, test_idx = indices[:int(0.8*len(label_list))], indices[int(0.8*len(label_list)):]
    dataset.partition.trainval = training_idx
    dataset.partition.test = test_idx
    # 全部作为训练集
    # dataset.partition.train = np.arange(len(label_list))

    with open(os.path.join(save_dir, 'peta_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/data/pantengteng/PETA/data/'

    generate_data_description(save_dir)
