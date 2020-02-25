#include "semantickitti_util.h"

int main() {
  SemanticKITTIData semantic_kitti_data;
  std::string input_data_dir = "/media/ganlu/PERL-SSD/ICRA2020/semantic_kitti/kitti/dataset/sequences/05/velodyne/";
  std::string input_label_dir = "/media/ganlu/PERL-SSD/ICRA2020/semantic_kitti/kitti/dataset/sequences/05/labels/";
  std::string output_label_dir = "/media/ganlu/PERL-SSD/ICRA2020/semantic_kitti/kitti/dataset/sequences/05/semantic_gt/";
  semantic_kitti_data.project_scans(input_data_dir, input_label_dir, output_label_dir, 1226, 370, 2761);
}
