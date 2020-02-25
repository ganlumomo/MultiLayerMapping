#pragma once

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

class SemanticKITTIData {
  public:
    
    bool project_scans(std::string input_data_dir, std::string input_label_dir, int scan_num) {
      for (int scan_id = 0; scan_id < 60; ++scan_id) {
        char scan_id_c[256];
        sprintf(scan_id_c, "%06d", scan_id);
        std::string scan_name = input_data_dir + std::string(scan_id_c) + ".bin";
        std::string label_name = input_label_dir + std::string(scan_id_c) + ".label";
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = kitti2pcl(scan_name, label_name);

        // 04-10: 2011_09_30_drive
        Eigen::Matrix4d calibration;
        calibration <<  -0.001857739385241, -0.999965951350955, -0.008039975204516, -0.004784029760483,
                        -0.006481465826011,  0.008051860151134, -0.999946608177406, -0.073374294642306,
                         0.999977309828677, -0.001805528627661, -0.006496203536139, -0.333996806443304,
       	                 0                ,  0                ,  0                ,  1.000000000000000;

        Eigen::Matrix<double, 3, 4> calibration3;
        calibration3 <<  -0.001857739385241, -0.999965951350955, -0.008039975204516, -0.004784029760483,
                        -0.006481465826011,  0.008051860151134, -0.999946608177406, -0.073374294642306,
                         0.999977309828677, -0.001805528627661, -0.006496203536139, -0.333996806443304;


        //Eigen::Matrix4d calibration_inverse = calibration.inverse();
        pcl::transformPointCloud(*cloud, *cloud, calibration);
        pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);

        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
            707.0912, 0, 601.8873,
            0, 707.0912, 183.1104,
            0, 0, 1);


        Eigen::Matrix<double, 3, 3> K;
        K <<  707.0912, 0, 601.8873,
            0, 707.0912, 183.1104,
            0, 0, 1;


        std::vector<cv::Point3d> objectPoints;
        for (int i = 0; i < cloud->points.size(); ++i) {
          cv::Point3d pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          objectPoints.push_back(pt);
        }

        cv::Mat projectImage(cv::Size(1226, 370), CV_32FC1);
        for (int i = 0; i < cloud->points.size(); ++i) {
          Eigen::Vector4d lidar_point;
          lidar_point << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1;
          Eigen::Vector3d image_point = K * calibration3 * lidar_point;
          double x = image_point[0]/image_point[2];
          double y = image_point[1]/image_point[2];
          if (y < 370 && y > 0 &&  x < 1226 &&  x > 0)
            projectImage.at<float>(int(y), int(x)) = cloud->points[i].label;
 
        }

        //std::vector<cv::Point2d> imagePoints;
        //cv::projectPoints(objectPoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3,1,CV_64F), cameraMatrix, cv::noArray(), imagePoints);
        //for (int i = 0; i < imagePoints.size(); ++i) {
          //std::cout << imagePoints[i].x << " " << imagePoints[i].y << std::endl;
          //if (imagePoints[i].y < 370 && imagePoints[i].y > 0 &&  imagePoints[i].x < 1226 &&  imagePoints[i].x > 0 && cloud->points[i].label != 255)
            //projectImage.at<float>(imagePoints[i].y, imagePoints[i].x) = cloud->points[i].label;
        //}
        cv::imwrite("project.png", projectImage);
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
