import os
import numpy as np
from PIL import Image

img_rgb_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_15/rgb_img/"
img_traversability_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_15/traversability_reproj/"
img_semantics_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_15/semantic_reproj/"
evaluation_list = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_15/evaluatioList.txt"
evaluation_list = np.loadtxt(evaluation_list)

palette_traversability = [255, 0, 0,
           		  0, 255, 0]

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

def colorize_mask_traversability(mask):
  new_mask = mask.convert('P')
  new_mask.putpalette(palette_traversability)
  return new_mask

def colorize_mask_semantics(mask):
  new_mask = mask.convert('P')
  new_mask.putpalette(palette)
  return new_mask

for img_id in evaluation_list:
  img_id = "%06i" % img_id
  img_rgb = Image.open(img_rgb_dir + img_id + '.png')
  img_traversability = Image.open(img_traversability_dir + img_id + '.png')
  #img = np.array(Image.open(img_traversability_dir + img_id + '.png'))
  #img = img[:,:,1]
  #img[img == 255] = 1
  #img_traversability = Image.fromarray(img)
  img_semantics = Image.open(img_semantics_dir + img_id + '.png')
  img_traversability_color = colorize_mask_traversability(img_traversability)
  img_semantics_color = colorize_mask_semantics(img_semantics)
  
  mask = Image.new("L", img_traversability_color.size, 128)
  img_traversability_visual = Image.composite(img_traversability_color, img_rgb, mask)
  img_semantics_visual = Image.composite(img_semantics_color, img_rgb, mask)
  img_traversability_visual.save(os.path.join(img_traversability_dir, '{}_visual.png'.format(img_id)))
  img_semantics_color.save(os.path.join(img_semantics_dir, '{}_visual.png'.format(img_id)))
  print('Results save'.format(img_id))
