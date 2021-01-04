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

# customed 2 pa100k
convert_dict={
    'Female': 'Female',
    # 'Male': None,
    'AgeLess16': 'AgeLess18',
    'Age17-45': None, # 'Age18-60',
    'Age46-60': None, # 'Age18-60',
    'Ageover60': 'AgeOver60',
    'Front': 'Front',
    'Side': 'Side',
    'Back': 'Back',
    'a-Backpack': 'Backpack',
    'a-ShoulderBag': 'ShoulderBag',
    'hs-Hat': 'Hat',
    'hs-Glasses': 'Glasses',
    'ub-ShortSleeve': 'ShortSleeve',
    'ub-LongSleeve': 'LongSleeve',
    'ub-Shirt': None,
    'ub-Sweater': None,
    'ub-Vest': None,
    'ub-TShirt': None,
    'ub-Cotton': None,
    'ub-Jacket': None,
    'ub-SuitUp': None,
    'ub-Coat': 'LongCoat', # None
    'ub-Black': None,
    'ub-Blue': None,
    'ub-Brown': None,
    'ub-Green': None,
    'ub-Grey': None,
    'ub-Orange': None,
    'ub-Pink': None,
    'ub-Purple': None,
    'ub-Red': None,
    'ub-White': None,
    'ub-Yellow': None,
    'lb-LongTrousers': 'Trousers',
    'lb-Shorts': 'Shorts',
    'lb-ShortSkirt': None, #'Skirt&Dress',
    'lb-Dress': None, #'Skirt&Dress',
    'lb-Black': None,
    'lb-Blue': None,
    'lb-Brown': None,
    'lb-Green': None,
    'lb-Grey': None,
    'lb-Orange': None,
    'lb-Pink': None,
    'lb-Purple': None,
    'lb-Red': None,
    'lb-White': None,
    'lb-Yellow': None
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
    'ub-Jacket': {'upper_wear_fg':['外套', '夹克']}, #
    'ub-SuitUp': {'upper_wear_fg':'西装'},
    'ub-Coat': {'upper_wear_fg':'风衣'}, #coat这里仅指风衣
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
    dataset.description = 'pa100k'
    dataset.root = os.path.join(save_dir, "data/release_data/release_data")
    dataset.attr_name = attr_list
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation/annotation.mat'))
    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name
    org_att_list = [pa100k_data['attributes'][i][0][0] for i in range(26)]
    split_img = {'train':80000, 'val':10000, 'test':10000}

    # 读取并创建百度label字典
    baidu_json = save_dir + "bdLabel.json"
    with open(baidu_json, "r") as f:
        db_label_list = json.load(f)
    index_list = [data["img_name"] for data in db_label_list]

    label_list = []
    for sub, length in split_img.items():
        sub_img_list = [pa100k_data[f'{sub}_images_name'][i][0][0] for i in range(split_img[sub])]
        org_label = pa100k_data[sub + "_label"]

        for id, img in enumerate(sub_img_list):
            label = []
            pa100k_label = org_label[id, :]
            db_label = db_label_list[index_list.index(img)]
            for index, att in enumerate(attr_list):
                # peta有对应标签
                if convert_dict[att]:
                    label.append(pa100k_label[org_att_list.index(convert_dict[att])])
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
    dataset.partition.train = np.arange(0, 80000)
    dataset.partition.val = np.arange(80000, 90000)
    dataset.partition.test = np.arange(90000, 100000)
    dataset.partition.trainval = np.arange(0, 90000)

    with open(os.path.join(save_dir, 'pa100k_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dir = '/data/pantengteng/PA100K/'

    generate_data_description(save_dir)
