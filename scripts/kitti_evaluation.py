import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image

img_gt_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_05/traversability_gt/"
img_pred_dir = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_05/traversability_reproj/"
evaluation_list = "/home/ganlu/media/PERL-SSD/00000IROS2020/kitti/data_kitti_05/evaluationList.txt"
evaluation_list = np.loadtxt(evaluation_list)

img_gt_all = []
img_pred_all = []

for img_id in evaluation_list:
  img_id = "%06i" % img_id
  img_gt = np.array(Image.open(img_gt_dir + img_id + '.png'))
  img_pred = np.array(Image.open(img_pred_dir + img_id + '.png'))

  img_gt_all.append(img_gt)
  img_pred_all.append(img_pred)

img_gt_all = np.array(img_gt_all).flatten()
img_pred_all = np.array(img_pred_all).flatten()

img_gt_all = img_gt_all[img_pred_all != 200]
img_pred_all = img_pred_all[img_pred_all != 200]

print( np.unique(np.concatenate((img_gt_all, img_pred_all), axis=0)) )
print( jaccard_score(img_gt_all, img_pred_all, average=None) )
