import os
import pickle
import pprint
from collections import OrderedDict, defaultdict

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
from models.resnet import resnet50, resnet101

from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, AverageMeter

set_seed(605)

def get_reload_weight(model_path, model):
    # 加载到cpu
    model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_dict["state_dicts"].items():
        name = k[7:]  # 去掉 `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
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

    _, valid_tsfm = get_transform(args)

    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.test_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {args.attr_num}')

    backbone = resnet50()
    classifier = BaseClassifier(nattr=args.attr_num)
    model = FeatClassifier(backbone, classifier)
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()

    print("reloading pretrained models")

    # exp_dir = os.path.join('exp_result', args.dataset)
    # model_path = os.path.join(exp_dir, args.dataset, 'img_model')
    model_path = args.pretrained_model
    model = get_reload_weight(model_path, model)

    model.eval()
    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            # imgs = imgs.cuda()
            # gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits = model(imgs)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    valid_result = get_pedestrian_metrics(gt_label, preds_probs, threshold=0.4)

    print(f'Evaluation on test set, \n',
          'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
              valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
          'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
              valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
              valid_result.instance_f1))

    #


if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument("--pretrained-model", type=str, default="pa100k_ckpt_max.pth")
    parser.add_argument("--attr-num", type=int, default=26)
    args = parser.parse_args()
    args.valid_split = "test"
    args.test_batchsize = 1
    main(args)

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
