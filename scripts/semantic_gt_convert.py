import numpy as np
from PIL import Image

# For KITTI odometry dataset
semantic_color_to_traversability = {
    (128,   0,   0) : 0,  # building
    (128, 128, 128) : 0,  # sky
    (128,  64, 128) : 1,  # road
    (128, 128,   0) : 0,  # vegetation
    (  0,   0, 192) : 1,  # sidewalk
    ( 64,   0, 128) : 0,  # car
    ( 64,  64,   0) : 1,  # terrain
    (  0, 128, 192) : 0,  # cyclist
    (192, 128, 128) : 0,  # signate
    ( 64,  64, 128) : 0,  # fence
    (192, 192, 128) : 0,  # pole
    (  0,   0,   0) : 0   # invalid
}

semantic_gt_dir = '/media/ganlu/PERL-SSD/00000IROS2020/gt_label/kitti_15/'
traversability_gt_dir = '/media/ganlu/PERL-SSD/00000IROS2020/gt_label/kitti_15_traversability/'
name_list = np.loadtxt(semantic_gt_dir + 'namelist.txt')

for img_id in name_list:
    img_id = "%06i" % img_id

    # Read images
    semantic_gt_color = np.array(Image.open(semantic_gt_dir + img_id + '.png'))
    traversability_gt_name = traversability_gt_dir + img_id + '.png'

    # Convert rgb to label
    rows, cols, _ = semantic_gt_color.shape
    traversability_gt = Image.new('L', (cols, rows))
    traversability_gt_pixels = traversability_gt.load()
    for i in range(rows):
        for j in range(cols):
            traversability_gt_pixels[j, i] = semantic_color_to_traversability[tuple(semantic_gt_color[i, j, :])]
    traversability_gt.save(traversability_gt_name)
