import os, pdb
import sys
import argparse
from PIL import Image
import numpy as np
import cv2
import scipy
import scipy.misc
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
import network
from optimizer import restore_snapshot
from datasets import kitti
from config import assert_and_infer_cfg
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--demo-image-folder', type=str, default='/home/ganlu/media/PERL-SSD/data_odometry_color/dataset/sequences/05/image_2/', help='path to demo image')
parser.add_argument('--snapshot', type=str, default='/home/ganlu/semantic-segmentation/kitti_best.pth', help='pre-trained checkpoint')
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--save-dir', type=str, default='/home/ganlu/media/PERL-SSD/data_odometry_color/dataset/sequences/05/semantic_seg/', help='path to save your results')
args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = True
# get net
args.dataset_cls = kitti
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# get data
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
for img_names in sorted(os.listdir(args.demo_image_folder)):
    img = Image.open(os.path.join(args.demo_image_folder, img_names)).convert('RGB')
    img_tensor = img_transform(img)
    # predict
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).cuda()
        prob = net(img)
        print('Inference done.')
        prob = prob.cpu().numpy().squeeze()
        prob = np.transpose(prob, ( 1,2,0))
        pred = np.argmax(prob, axis=2)
    softmax_prob = scipy.special.softmax(prob, axis=2)
    softmax_prob.tofile(os.path.join(args.save_dir, img_names[:-4]+'.bin') )
    colorized = args.dataset_cls.colorize_mask(pred)
    colorized.save(os.path.join(args.save_dir, '{}_color_mask.png'.format(img_names)))
    print('Results saved. {}'.format(img_names))
