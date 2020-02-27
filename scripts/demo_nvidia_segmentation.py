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

traversability = {
  0 : 1, # road
  1 : 1, # sidewalk
  2 : 0, # building
  3 : 0, # wall
  4 : 0, # fence
  5 : 0, # pole
  6 : 0, # traffic light
  7 : 0, # traffic sign
  8 : 0, # vegetation
  9 : 1, # terrain
  10 : 0, # sky
  11 : 0, # person
  12 : 0, # rider
  13 : 0, # car
  14 : 0, # truck
  15 : 0, # bus
  16 : 0, # train
  17 : 0, # motorcycle
  18 : 0, # bicycle
}

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--demo-image-folder', type=str, default='/home/ganlu/media/PERL-SSD/data_odometry_color/dataset/sequences/06/image_2/', help='path to demo image')
parser.add_argument('--snapshot', type=str, default='/home/ganlu/semantic-segmentation/kitti_best.pth', help='pre-trained checkpoint')
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
parser.add_argument('--save-dir', type=str, default='/home/ganlu/media/PERL-SSD/data_odometry_color/dataset/sequences/06/semantic_seg/', help='path to save your results')
parser.add_argument('--save-traversability-dir', type=str, default="/home/ganlu/media/PERL-SSD/data_odometry_color/dataset/sequences/06/traversability_seg/", help='path to save traversability results')
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
        img_cuda = img_tensor.unsqueeze(0).cuda()
        prob = net(img_cuda)
        print('Inference done.')
        prob = prob.cpu().numpy().squeeze()
        prob = np.transpose(prob, ( 1,2,0))
        pred = np.argmax(prob, axis=2)
    #softmax_prob = scipy.special.softmax(prob, axis=2)
    #softmax_prob.tofile(os.path.join(args.save_dir, img_names[:-4]+'.bin') )
    #colorized = args.dataset_cls.colorize_mask(pred)
    #colorized.save(os.path.join(args.save_dir, '{}_color_mask.png'.format(img_names)))
 
    # convert to traversability
    traversability_pred = np.empty_like(pred)
    for i in range(pred.shape[0]):
      for j in range(pred.shape[1]):
        traversability_pred[i][j] = traversability[pred[i][j]]
   
    # output results
    semantic_label = Image.fromarray(np.uint8(pred), 'L')
    semantic_label.save(os.path.join(args.save_dir, '{}'.format(img_names)))
    traversability_label = Image.fromarray(np.uint8(traversability_pred), 'L')
    traversability_label.save(os.path.join(args.save_traversability_dir, '{}'.format(img_names)))

    # visualize results
    #colorized = args.dataset_cls.colorize_mask(traversability_pred)
    #mask = Image.new("L", colorized.size, 128)
    #colorized_img = Image.composite(colorized, img, mask)
    #colorized_img.save(os.path.join(args.save_traversability_dir, '{}_color_mask.png'.format(img_names)))
    
    print('Results saved. {}'.format(img_names))
