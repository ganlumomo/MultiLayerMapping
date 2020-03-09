import os
import numpy as np
from PIL import Image

img_traversability_seg_dir = "/home/ganlu/media/PERL-SSD/data_odometry_color/dataset/sequences/06/traversability_seg/"
img_traversability_proj_dir = "/home/ganlu/media/PERL-SSD/Datasets/KITTI/dataset/sequences/06/traversability_new/"
img_traversability_final_dir = "/home/ganlu/media/PERL-SSD/Datasets/KITTI/dataset/sequences/06/traversability_gt_new/"
img_list = "/home/ganlu/media/PERL-SSD/Datasets/KITTI/dataset/sequences/06/traversability_new/list.txt"
img_list = np.loadtxt(img_list)

for img_id in img_list:
  img_id = "%06i" % img_id
  img_traversability_seg = Image.open(img_traversability_seg_dir + img_id + '.png')
  img_traversability_proj = Image.open(img_traversability_proj_dir + img_id + '.png')

  np_traversability_seg = np.array(img_traversability_seg)
  np_traversability_proj = np.array(img_traversability_proj)
  np_traversability_final = np.empty_like(np_traversability_seg)
  np_traversability_final = np.logical_and(np_traversability_seg, np_traversability_proj)
  
  img_traversability_final = Image.fromarray(np.uint8(np_traversability_final))
  img_traversability_final.save(os.path.join(img_traversability_final_dir, '{}.png'.format(img_id)))
  print('Result saved.'.format(img_id))
