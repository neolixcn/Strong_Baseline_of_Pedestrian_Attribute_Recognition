import os,cv2
import pickle
import pprint
from collections import OrderedDict, defaultdict
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50

from tools.function import get_model_log_path, get_pedestrian_metrics # get_reload_weight
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, AverageMeter

set_seed(605)

def get_reload_weight(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dicts'])
    return model

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    backbone = resnet50(pretrained=False)
    classifier = BaseClassifier(nattr=args.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    print("reloading pretrained models")

    # exp_dir = os.path.join('exp_result', args.dataset)
    # model_path = os.path.join(exp_dir, args.dataset, 'img_model')
    model_path = args.pretrained_model
    model = get_reload_weight(model_path, model).cuda()

    model.eval()
    preds_probs = []
    img_list = list(Path(args.test_imgs).glob("*"))
    print("total_imgs:", len(img_list))
    with torch.no_grad():
        for step, img_path in enumerate(img_list):
            img = Image.open(img_path)
            # img = img.resize((192, 256))
            img = img.convert("RGB")
            org_img = np.array(img).copy()
            org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
            org_img = cv2.resize(org_img, (192, 256))
            
            img = valid_tsfm(img)
            imgs = img.cuda()
            imgs = img.unsqueeze(0)

            valid_logits = model(imgs)
            # 2
            # valid_probs = torch.sigmoid(valid_logits)
            # probs = valid_probs.flatten()
            # probs = probs.cpu().numpy()
            # idx = np.where(probs>0.5)
            # no_idx  = np.where(probs<0.5)

            # 1
            logits = valid_logits.flatten()
            # print("logits:", logits)
            logits = logits.cpu().numpy()
            idx = np.where(logits>0)
            # print("idx:", idx)
            print("img: {}\n labels:{}".format(img_path.name, args.attr_list[idx]))
            bg_img = np.ones([300,350,3]) * 255
            bg_img[:256, :192, :] = org_img
            h = 0
            for id in idx[0]:
                text_line = str(args.attr_list[id])
                # print(text_line)
                text_size, baseline = cv2.getTextSize(str(text_line), args.fontFace, args.fontScale, args.thickness)
                h += baseline + text_size[1]
                cv2.putText(bg_img, str(text_line), (193, h), args.fontFace, args.fontScale, (0, 255, 0), args.thickness, 8)
            cv2.imwrite(args.save_path + args.dataset + img_path.name , bg_img)



if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument("--pretrained-model", type=str, default="pa100k_ckpt_max.pth")
    parser.add_argument("--test-imgs", type=str, default="./test_images")
    parser.add_argument("--save-path", type=str, default="./test_results/")
    parser.add_argument("--attr-num", type=int, default=26)
    args = parser.parse_args()
    args.fontScale = 0.4
    args.thickness = 1
    args.fontFace = cv2.FONT_HERSHEY_SIMPLEX

    # 'PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'
    # pa100k 26
    if args.dataset == 'PA100k':
        args.attr_list = np.array(['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo'
        , 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts'
        , 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'AgeOver60'
        , 'Age18-60', 'AgeLess18', 'Female', 'Front', 'Side', 'Back'])
    
    #peta 35
    elif args.dataset == 'PETA':
        args.attr_list = np.array(['accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','hairLong'
        ,'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt','upperBodyOther','upperBodyVNeck'
        , 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt','lowerBodyTrousers'
        , 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker'
        , 'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags'
        , 'personalLess30','personalLess45','personalLess60','personalLarger60'
        , 'personalMale'])

    #rapv1
    elif args.dataset == 'RAP':
        args.attr_list = np.array(['hs-BaldHead','hs-LongHair','hs-BlackHair','hs-Hat','hs-Glasses','hs-Muffler'
        , 'ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve'
        , 'lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers'
        , 'shoes-Leather','shoes-Sport','shoes-Boots','shoes-Cloth','shoes-Casual'
        , 'attach-Backpack','attach-SingleShoulderBag','attach-HandBag','attach-Box','attach-PlasticBag','attach-PaperBag','attach-HandTrunk','attach-Other'
        , 'AgeLess16','Age17-30','Age31-45'
        , 'Female', 'BodyFat','BodyNormal','BodyThin'
        , 'Customer','Clerk'
        , 'action-Calling','action-Talking','action-Gathering','action-Holding','action-Pusing','action-Pulling','action-CarrybyArm','action-CarrybyHand'])

    #rapv2
    elif args.dataset == 'RAP2':
        args.attr_list = np.array(['hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses'
        , 'ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve','ub-Others'
        , 'lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers'
        , 'shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'shoes-Other'
        , 'attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk','attachment-Other'
        , 'AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60'
        , 'Female'
        , 'BodyFat','BodyNormal','BodyThin'
        , 'Customer','Employee'
        , 'action-Calling','action-Talking','action-Gathering','action-Holding','action-Pushing','action-Pulling','action-CarryingByArm','action-CarryingByHand','action-Other'])

    main(args)

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')

"""
