# 调用百度api生成伪行人属性标签
# 需修改：filePath，后缀，保存位置
import json
import os
import pdb
import cv2
import base64
from pathlib import Path
from PIL import Image
from aip import AipBodyAnalysis


""" 你的 APPID AK SK """
#APP_ID = '23082193'
#API_KEY = '3S8rXa9H5KwlqRVwxLQrZRtF'
#SECRET_KEY = 'KPnvX7lWc7LSW3SnX9HV8mQUDzlrW0Ka'

APP_ID = '23116978'
API_KEY = 'yHzx0bycHBRAR9SxGzSt0FQX'
SECRET_KEY = '9kEdsFOnkGn6VsmkDRsplFFKH92KiAF3'

client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)


# 按比例缩放图片比例

def scale(img, width=None, height=None):

    if not width and not height:
        # width, height = img.size  # 原图片宽高
        width, height = img.shape[1], img.shape[0]
    if not width or not height:
        # _width, _height = img.size
        _width, _height = img.shape[1], img.shape[0]
        height = width * _height / _width if width else height
        width = height * _width / _height if height else width
    return width, height


""" 读取图片 """

# PETA的十个数据集
list = ['3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'i-LID',
        'MIT', 'PRID', 'SARC3D', 'TownCentre', 'VIPeR']

# for l in list:
# filePath = '/home/leilanxin/reduced-data' #测试集
# filePath = '/data/pantengteng/PA100K/data/release_data/release_data'  # PA100K数据集
filePath = '/data/pantengteng/PETA/data/SARC3D/archive'
# filePath = '/data/pantengteng/PETA/data/'+l+'/archive' #PETA数据集

jsonPath = "SARC3D.json"
json_toPath = "test.json"
suffix = "bmp"

# 获取文件夹中的图片（以数据传入接口）
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 将未在json文件中的图片生成列表，再传入函数中
dataArray = []
data = []
if Path(jsonPath).exists():
    # 注意是否手动添加了[]以及去掉最后一个标签的,
    with open(jsonPath, encoding="utf-8") as f:
        images = json.load(f)
    for image in images:
        data.append(image['img_name'])
    for files in os.listdir(filePath):
        if files not in data:
            # pdb.set_trace()
            if files[-3:] == suffix:
                files_all_name = os.path.join(filePath, files)
                dataArray.append(files_all_name)
else:
    for files in os.listdir(filePath):
        # 修改后缀
        if files[-3:] == 'bmp':
            files_all_name = os.path.join(filePath, files)
            dataArray.append(files_all_name)
print("总共", len(dataArray), "张图片待识别")


# 递归函数调用百度接口


def mFuction(dataArray):
    i = 0
    print('下一轮调用")
    failArray = []
    for data in dataArray:
        try:
            path = Path(data)

            # 判断是否为txt文件
            if data[-3:] == 'txt':
                continue

            # 读取图片
            img = cv2.imread(data)

            # 若图像宽未达到50像素，等比例放大
            if img.shape[1] < 50:
                # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
                w, h = scale(img, width=55)
                img2 = cv2.resize(img, (int(w), int(h)))
                img_str = cv2.imencode(path.suffix, img2)[
                    1].tobytes()  # tostring()
            else:
                img_str = get_file_content(data)

            # 调用人体检测与属性识别
            res = client.bodyAttr(img_str)
            res['img_name'] = path.name

            if 'error_msg' in res:
                if res['error_msg'] == 'image size error':
                    print(img.size)

            if "error_code" in res:
                failArray.append(data)
            else:
                e = json.dumps(res, indent=2, ensure_ascii=False)
                i += 1
                # pdb.set_trace()
                with open(json_toPath, 'a', encoding='utf-8') as w:
                    w.write(e+',')
                    w.write('\n')

        except Exception as e:
            if e == KeyboardInterrupt:
                raise e

            print("exception", e)
            failArray.append(data)

    # 如果有未成功生成标签的图像，再调用函数
    if failArray:
        print(i)
        print(len(failArray))  # failArray)
        mFuction(failArray)
    else:
        print("已完成")
        return


mFuction(dataArray)


'''

# 单张操作（针对'person_num'=0的情况）

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

path = '/home/pantengteng/Programs/Strong_Baseline_of_Pedestrian_Attribute_Recognition/test_images4/person_202008221514:51:982_1.jpg'

img_str = get_file_content(path)
res = client.bodyAttr(img_str)

res['img_name'] = path
print(res)
e = json.dumps(res, indent=2, ensure_ascii=False)
with open('aaaaa.json', 'a', encoding='utf-8') as w:
    w.write(e+',')
    w.write('\n')

'''

# 检查每个json文件是否覆盖了所有的图片
'''
# 对PETA的检查
list = ['3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'i-LID',
        'MIT', 'PRID', 'SARC3D', 'TownCentre', 'VIPeR']

for l in list:
    i = 0
    names = []
    toPath = '/data/pantengteng/PETA/data/'
    toPath = toPath + l + '/archive/'
    for file in os.listdir(toPath):
        names.append(file)
    # print(names)

    with open(toPath+'bdLabel.json', encoding="utf-8") as f:
        labels = json.load(f)
    for label in labels:
        if label["img_name"] in names:
            i += 1
            continue
        else:
            print('未找到该文件' + label)
    print(i)
    print('已完成' + l+'.json文件的检查')
'''


# 对PA100K的检查
'''
i = 0
names = []
toPath = '/data/pantengteng/PA100K/data/release_data/release_data'
for file in os.listdir(toPath):
    names.append(file)
# print(names)

with open('pa100k1.json', encoding="utf-8") as f:
    labels = json.load(f)
for label in labels:
    if label["img_name"] in names:
        i += 1
        continue
    else:
        print('未找到该文件' + label)
print(i)
print('已完成pa100k1.json文件的检查')
'''
