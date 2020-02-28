import os
import numpy as np
from PIL import Image

img_rgb_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_05/rgb_img/"
img_pred_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_05/traversability/"
evaluation_list = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_05/evaluatioList.txt"
evaluation_list = np.loadtxt(evaluation_list)

palette = [255, 0, 0,
           0, 255, 0]

def colorize_mask(mask):
  new_mask = mask.convert('P')
  new_mask.putpalette(palette)
  return new_mask

for img_id in evaluation_list:
  img_id = "%06i" % img_id
  img_rgb = Image.open(img_rgb_dir + img_id + '.png')
  img_pred = Image.open(img_pred_dir + img_id + '.png')

  img_pred_color = colorize_mask(img_pred)
  
  mask = Image.new("L", img_pred_color.size, 128)
  img_visual = Image.composite(img_pred_color, img_rgb, mask)
  img_visual.save(os.path.join(img_pred_dir, '{}_visual.png'.format(img_id)))
  print('Results save'.format(img_id))
