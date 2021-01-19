import os
import cv2
import numpy as np
from PIL import Image

# For TartanAir dataset
semantic_to_traversability = {
        13 : 0,
        64 : 1,
        96 : 0,
        110 : 0,
        129 : 0,
        137 : 0,
        152 : 1, # not sure
        153 : 0,
        160 : 1, # not sure
        163 : 1, # not sure
        164 : 0,
        167 : 0,
        178 : 0,
        184 : 0,
        196 : 0,
        197 : 1,
        199 : 0,
        200 : 0,
        205 : 1,
        207 : 1,
        220 : 0,
        222 : 1,
        226 : 1, # not sure
        227 : 0,
        230 : 1, # not sure
        244 : 1, # not sure
        245 : 0,
        246 : 0,
        250 : 0,
        252 : 1  # not sure
}

palette = [255, 0, 0, 0, 255, 0]

def colorize_mask(mask):
    new_mask = mask.convert('P')
    new_mask.putpalette(palette)
    return new_mask


seq_dir = '/media/ganlu/Samsung_T5/000_tro2020/tartanair-release1/abandonedfactory/Easy/P008/'
rgb_img_dir = seq_dir + 'image_left/'
semantic_gt_dir = seq_dir + 'seg_left/'
traversability_gt_dir = seq_dir + 'traversability_new/'
traversability_gt_final_dir = seq_dir + 'traversability_gt/'

for file_name in os.listdir(traversability_gt_dir):
    print(file_name[:-4])

    # Load semantic labels
    semantic_gt = np.load(semantic_gt_dir + file_name[:-4] + '_left_seg.npy')
    cv2.imwrite(semantic_gt_dir + file_name[:-4] + '_left.png', semantic_gt)

    # Read traversability images
    traversability_gt = np.array(Image.open(traversability_gt_dir + file_name))
    traversability_gt_final_name = traversability_gt_final_dir + file_name[:-4] + '_left.png'

    # Infer traversability from semantics
    rows, cols = semantic_gt.shape
    for i in range(0, rows):
        for j in range(0, cols):
            semantic_gt[i, j] = semantic_to_traversability[semantic_gt[i, j]]
    traversability_gt_final = np.uint8(np.logical_and(traversability_gt, semantic_gt))
    cv2.imwrite(traversability_gt_final_dir + file_name[:-4] + '_left.png', traversability_gt_final)

    # Visualization
    rgb_img = Image.open(rgb_img_dir + file_name[:-4] + '_left.png')
    traversability_gt_final = Image.fromarray(traversability_gt_final)
    traversability_gt_final_color = colorize_mask(traversability_gt_final)
    mask = Image.new("L", traversability_gt_final.size, 128)
    traversability_gt_final_visual = Image.composite(traversability_gt_final_color, rgb_img, mask)
    traversability_gt_final_visual.save(traversability_gt_final_dir + file_name[:-4] + '_visual.png')
