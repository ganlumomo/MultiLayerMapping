import os
import numpy as np
from PIL import Image

img_rgb_dir = "/home/ganlu/media/PERL-SSD/Datasets/KITTI/dataset/sequences/07/image_2/"
img_traversability_dir = "/home/ganlu/media/PERL-SSD/Datasets/KITTI/dataset/sequences/07/traversability_gt_new/"
img_list = "/home/ganlu/media/PERL-SSD/Datasets/KITTI/dataset/sequences/07/traversability_new/list.txt"
img_list = np.loadtxt(img_list)

palette = [255, 0, 0, 0, 255, 0]

def colorize_mask(mask):
  new_mask = mask.convert('P')
  new_mask.putpalette(palette)
  return new_mask

for img_id in img_list:
  img_id = "%06i" % img_id
  img_rgb = Image.open(img_rgb_dir + img_id + '.png')
  img_traversability = Image.open(img_traversability_dir + img_id + '.png')
  img_traversability_color = colorize_mask(img_traversability)

  mask = Image.new("L", img_traversability_color.size, 128)
  #img_rgb = img_rgb.resize((640, 480))
  img_traversability_visual = Image.composite(img_traversability_color, img_rgb, mask)
  img_traversability_visual.save(os.path.join(img_traversability_dir, '{}_visual.png'.format(img_id)))
  print('Result saved.'.format(img_id))
