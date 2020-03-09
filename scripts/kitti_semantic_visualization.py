import os
import numpy as np
from PIL import Image

img_semantic_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_15/semantic_reproj/"
img_list = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_15/evaluatioList.txt"
img_list = np.loadtxt(img_list)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

def colorize_mask(mask):
  new_mask = mask.convert('P')
  new_mask.putpalette(palette)
  return new_mask

for img_id in img_list:
  img_id = "%06i" % img_id
  img_semantic = Image.open(img_semantic_dir + img_id + '.png')
  img_semantic_color = colorize_mask(img_semantic)
  img_semantic_color.save(os.path.join(img_semantic_dir, '{}_visual.png'.format(img_id)))
  print('Result saved.'.format(img_id))
