import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image

img_gt_dir = "/media/ganlu/Samsung_T5/000_tro2020/tartanair-release1/neighborhood/Easy/P005/seg_left/"
img_test_dir = "/media/ganlu/Samsung_T5/000_tro2020/tartanair-release1/neighborhood/Easy/P005/semantic_pred/"
img_pred_dir = "/media/ganlu/Samsung_T5/000_tro2020/tartanair-release1/neighborhood/Easy/P005/semantic_reproj/"
#img_gt_dir = "/home/ganlu/media/Samsung_T5/000_tro2020/tartanair-release1/abandonedfactory/Easy/P001/seg_left/"
#img_test_dir = "/home/ganlu/media/Samsung_T5/000_tro2020/tartanair-release1/abandonedfactory/Easy/P001/semantic_pred/"
#img_pred_dir = "/home/ganlu/media/Samsung_T5/000_tro2020/tartanair-release1/abandonedfactory/Easy/P001/semantic_reproj/"
evaluation_list = "/media/ganlu/Samsung_T5/000_tro2020/tartanair-release1/neighborhood/Easy/P005/evaluation_list.txt"
evaluation_list = np.loadtxt(evaluation_list)

img_gt_all = []
img_test_all = []
img_pred_all = []

for img_id in evaluation_list:
  img_id = "%06i" % img_id
  img_gt = np.array(Image.open(img_gt_dir + img_id + '_left_trainid.png'))
  img_test = np.array(Image.open(img_test_dir + img_id + '_left.png'))
  img_pred = np.array(Image.open(img_pred_dir + img_id + '.png'))

  img_gt_all.append(img_gt)
  img_test_all.append(img_test)
  img_pred_all.append(img_pred)

img_gt_all = np.array(img_gt_all).flatten()
img_test_all = np.array(img_test_all).flatten()
img_pred_all = np.array(img_pred_all).flatten()

img_gt_all = img_gt_all[img_pred_all != 255]
img_test_all = img_test_all[img_pred_all != 255]
img_pred_all = img_pred_all[img_pred_all != 255]

print( np.unique(np.concatenate((img_gt_all, img_test_all), axis=0)) )
print( jaccard_score(img_gt_all, img_test_all, average='macro') )
#print( np.mean(jaccard_score(img_gt_all, img_test_all, average=None)) )
print( np.unique(np.concatenate((img_gt_all, img_pred_all), axis=0)) )
print( jaccard_score(img_gt_all, img_pred_all-1, average='macro') )
#print( np.mean(jaccard_score(img_gt_all, img_pred_all, average=None)) )
