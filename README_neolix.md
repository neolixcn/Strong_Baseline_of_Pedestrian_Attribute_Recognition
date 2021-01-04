## A Strong Baseline of Pedestrian Attribute Recognition

The code for paper [Rethinking of Pedestrian Attribute Recognition: Realistic Datasets with Efficient Method](https://arxiv.org/abs/2005.11909).

## 数据集生成
原始数据集不能满足需求，因此可借助于预训练模型和可用的接口补充伪标签监督训练。

### 接口生成标签
文件  baidu_api.py  
**功能**：调用百度api接口输出给定文件夹中的图片的行人属性标签json文件

**待修改参数：**
- filePath：传入图片的文件夹的完整路径
- jsonPath：打开的json文件名
- suffix：图片后缀
- json_toPath：待保存json文件的文件名

**注意：**

 - 当输入图片未检测到行人时，返回属性为空。可进行手动裁剪图片，再传入“单张操作”中，更改传入图片路径，即可得到这张图片的标签。

 - 建议对生成的json文件，进行复查。

### 辅助标注工具
文件： label_assistant.ipynb  
**功能**：  
实现快速的手动标注图片中的行人属性标签功能，返回人体的静态属性和行为，共支持91种属性，包括：性别、年龄阶段、衣着等。

**使用**：  
对每张图片，属性会按类别单个或成堆出现，单个出现时，0代表不存在，1代表存在；成堆出现时，输入数字代表对应位置的属性存在，-1代表都不存在。输入无效时，会提醒并等待再次输入；输入错误时，可输”again”对该张图片重新标注。全部标注完成后，会打印出错的图片名称。
 
**待修改参数：**  
 - img_dir： 输入图片路径
 - i：从当前文件夹的第i张图片开始标注，用于中断的情况
 - jsonPath：待保存json文件的文件名

### 合并标签
这里结合百度接口生产的标签文件，合并出需要的数据集pkl文件。
转换脚本所在目录：dataset/preprocess/

1. convert peta.py  
修改save_dir为指向数据集PETA/data/的路径。
生产的pkl默认在save_dir目录下。
2. convert_pa100k.py  
修改save_dir为指向PA100K数据集根目录的路径。
生产的pkl默认在save_dir目录下。

### 混合数据集
1. combine_peta_pa100k.py  
   - 修改save_dir为生成的混合数据集pkl的存放目录，建议选择data/combined路径。
    - peta_dir为peta的数据集pkl文件存放路径，建议存放于data/PETA。
    - pa100k_dir为pa100k数据集pkl存放目录，建议存放于data/PA100k。
2. combine_peta_pa100k_add_val.py 
与1类似，但是生产的数据集不做拆分，且支持重采样：
sample_prob参数配置为采样概率（或权重系数）的路径即可。

### neolix数据集
1. convert_neolix.py
   - save_dir为保存目录，建议选择/data/neolix。
   - json_file为neolix的json标注文件
   - sample_prob，如需重采样，可指定采样概率数组路径。

## 训练
可使用train.sh脚本或如下命令：
```
python train.py combined --train_epoch 30 --batchsize 256 --height 256 --width 192 --ft "path/to/pth"
```
可选参数参考config.py文件
关键参数：
 - train_epoch：训练轮数
 - batchsize：批大小
 - ft：预训练模型路径

## 测试
1.仅有图片测试可是化结果
可使用test.sh脚本或如下命令：
```
CUDA_VISIBLE_DEVICES=2 python test_infer.py 'neolix' --att-type 'STD' --attr-num 48 --test-imgs "path/to/testimages" --save-path "path/to/save/test_result/" --pretrained-model "path/to/pth"
```
参数介绍：
 - dataset：指定数据集名称，未指定att-type时会用于对应属性表，若已指定att-type，则仅用于打印信息和保存名称区分
 - att-type：用于指定属性表
 - attr-num：属性个数
 - test-imgs：测试集图片路径
 - save-path：结果保存路径
 - pretrained-model：预训练模型

2.测试集有标签测试指标  
可使用test.sh脚本或如下命令：
```
CUDA_VISIBLE_DEVICES=1 python infer.py 'test' --attr-num 48 --pretrained-model  'path/to/pth'
```
参数介绍：
 - dataset：指定数据集名称
  - attr-num：属性个数
  - pretrained-model：预训练模型

## 实验文档
预训练模型存放于exp_result目录下：
http://wiki.neolix.cn/pages/resumedraft.action?draftId=8327035&draftShareId=88e6b5c8-42e9-450c-b626-476431333534