import os
import cv2
import numpy as np

# For TartanAir dataset
depth_scale = 2000
uint16_max = 65535

seq_dir = '/media/ganlu/Samsung_T5/000_tro2020/tartanair-release1/abandonedfactory/Easy/P001/'
depth_img_dir = seq_dir + 'depth_left/'

for file_name in os.listdir(depth_img_dir):
    print(file_name[:-4])
    depth_left = np.load(depth_img_dir + file_name)
    depth_left = depth_left * depth_scale
    depth_left[depth_left > uint16_max] = 0
    cv2.imwrite(depth_img_dir + file_name[:-4] + '.png', depth_left.astype(np.uint16))
