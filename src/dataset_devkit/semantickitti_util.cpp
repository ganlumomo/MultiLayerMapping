#include "semantickitti_util.h"

int main() {
  SemanticKITTIData semantic_kitti_data;
  std::string input_data_dir = "/media/ganlu/PERL-SSD/ICRA2020/semantic_kitti/kitti/dataset/sequences/05/velodyne/";
  std::string input_label_dir = "/media/ganlu/PERL-SSD/ICRA2020/semantic_kitti/kitti/dataset/sequences/05/labels/";
  semantic_kitti_data.project_scans(input_data_dir, input_label_dir, 1);
}
