#pragma once

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/highgui/highgui.hpp>

#include "markerarray_pub.h"

class SemanticKITTIData {

  public:
    bool project_scans(std::string input_data_dir, std::string input_label_dir, std::string output_label_dir,
                       int image_width, int image_height, int scan_num) {
      for (int scan_id = 0; scan_id < scan_num; ++scan_id) {
        char scan_id_c[256];
        sprintf(scan_id_c, "%06d", scan_id);
        std::string scan_name = input_data_dir + std::string(scan_id_c) + ".bin";
        std::string label_name = input_label_dir + std::string(scan_id_c) + ".label";
        std::string proj_name = output_label_dir + std::string(scan_id_c) + ".png";
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = kitti2pcl(scan_name, label_name);

        // 04-10: 2011_09_30_drive
        Eigen::Matrix<double, 3, 4> Tr;
        Tr << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
              -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
              9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01;

        Eigen::Matrix3d P0;
        P0 << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02,
              0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02,
              0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;
        
        cv::Mat projectImage = cv::Mat(cv::Size(image_width, image_height), CV_32FC1, cv::Scalar(0));
        //cv::Mat projectImageRGB = cv::Mat(cv::Size(image_width, image_height), CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < cloud->points.size(); ++i) {
          Eigen::Vector4d lidar_point;
          lidar_point << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1;
          Eigen::Vector3d image_point = P0 * Tr * lidar_point;
          if (image_point[2] < 0)
            continue;
          int x = image_point[0] / image_point[2];
          int y = image_point[1] / image_point[2];
          if (x < image_width &&  x > 0 && y < image_height && y > 0) {
            projectImage.at<float>(y, x) = cloud->points[i].label;
            //projectImageRGB.at<cv::Vec3b>(y, x)[0] = la3dm::SemanticKITTISemanticMapColor(cloud->points[i].label).r * 255;
            //projectImageRGB.at<cv::Vec3b>(y, x)[1] = la3dm::SemanticKITTISemanticMapColor(cloud->points[i].label).g * 255;
            //projectImageRGB.at<cv::Vec3b>(y, x)[2] = la3dm::SemanticKITTISemanticMapColor(cloud->points[i].label).b * 255;
          }
        }
        cv::imwrite(proj_name, projectImage);
      }
    }

  private:
    pcl::PointCloud<pcl::PointXYZL>::Ptr kitti2pcl(std::string fn, std::string fn_label) {
      FILE* fp_label = std::fopen(fn_label.c_str(), "r");
      if (!fp_label)
        std::perror("File opening failed");
      std::fseek(fp_label, 0L, SEEK_END);
      std::rewind(fp_label);
      FILE* fp = std::fopen(fn.c_str(), "r");
      if (!fp)
        std::perror("File opening failed");
      std::fseek(fp, 0L, SEEK_END);
      size_t sz = std::ftell(fp);
      std::rewind(fp);
      int n_hits = sz / (sizeof(float) * 4);
      pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
      for (int i = 0; i < n_hits; i++) {
        pcl::PointXYZL point;
        float intensity;
        if (fread(&point.x, sizeof(float), 1, fp) == 0) break;
        if (fread(&point.y, sizeof(float), 1, fp) == 0) break;
        if (fread(&point.z, sizeof(float), 1, fp) == 0) break;
        if (fread(&intensity, sizeof(float), 1, fp) == 0) break;
        if (fread(&point.label, sizeof(float), 1, fp_label) == 0) break;
        pc->push_back(point);
      }
      std::fclose(fp);
      std::fclose(fp_label);
      return pc;
    }
};
