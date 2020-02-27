#pragma once

#include <opencv/cv.hpp>
#include <pcl/common/transforms.h>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

class KittiData {
  public:
    KittiData(ros::NodeHandle& nh, std::string smap_topic, std::string tmap_topic,
        float fx, float fy, float cx, float cy, float depth_scaling,
        double resolution, double block_depth, int num_class,
        double sf2, double ell, double free_thresh, double occupied_thresh, double var_thresh,
        double prior_A, double prior_B, double prior,
        double ds_resolution, double free_resolution, double max_range)
      : nh_(nh)
      , fx_(fx)
      , fy_(fy)
      , cx_(cx)
      , cy_(cy)
      , depth_scaling_(depth_scaling)
      , num_class_(num_class)
      , ds_resolution_(ds_resolution)
      , free_resolution_(free_resolution)
      , max_range_(max_range) {
        map_ = new la3dm::BGKOctoMap(resolution, block_depth, num_class_,
                                     sf2, ell, free_thresh, occupied_thresh, var_thresh,
                                     prior_A, prior_B, prior);
        sm_pub_ = new la3dm::MarkerArrayPub(nh_, smap_topic, resolution);
        tm_pub_ = new la3dm::MarkerArrayPub(nh_, tmap_topic, resolution);
      }

    void process_scans(int scan_num, std::string depth_img_dir, std::string label_img_dir) {
      for (int scan_id = 0; scan_id <= scan_num; ++scan_id) {
        char scan_id_c[256];
        sprintf(scan_id_c, "%06d", scan_id);
        std::string scan_id_s(scan_id_c);
        std::string depth_img_name(depth_img_dir + scan_id_s + ".png");
        std::string label_img_name(label_img_dir + scan_id_s + ".png");

        cv::Mat depth_img = cv::imread(depth_img_name, CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat label_img = cv::imread(label_img_name, CV_LOAD_IMAGE_UNCHANGED);
        Eigen::Matrix4f transform = get_current_pose(scan_id);

        la3dm::PCLPointCloudwithLabel cloudwlabel;
        la3dm::point3f origin;
        process_scan(depth_img, label_img, transform, cloudwlabel, origin);
        map_->insert_semantics(cloudwlabel, origin, ds_resolution_, free_resolution_, max_range_, num_class_);
        publish_semantic_map();
      }
    }

    void process_scan(const cv::Mat& depth_img, const cv::Mat& label_img, const Eigen::Matrix4f& transform,
                      la3dm::PCLPointCloudwithLabel& cloudwlabel, la3dm::point3f& origin) {
      int width = depth_img.cols;
      int height = depth_img.rows;
      for (int32_t i = 0; i < width * height; ++i) {
        int ux = i % width;
        int uy = i / width;
        
        int pix_label = label_img.at<uint8_t>(uy, ux);
        if (pix_label == 10)  // ignore sky label
          continue;
        
        float pix_depth = (float) depth_img.at<uint16_t>(uy, ux) / depth_scaling_;
        if (pix_depth > 0.1) {
          la3dm::PCLPointwithLabel ptl;
          ptl.x = (ux - cx_) * (1.0 / fx_) * pix_depth;
          ptl.y = (uy - cy_) * (1.0 / fy_) * pix_depth;
          ptl.z = pix_depth;
          ptl.label = pix_label + 1;
          cloudwlabel.points.push_back(ptl);
        }
      }
      pcl::transformPointCloud(cloudwlabel, cloudwlabel, transform);
      origin.x() = transform(0, 3);
      origin.y() = transform(1, 3);
      origin.z() = transform(2, 3);
    }
    
    bool read_camera_poses(const std::string camera_pose_name) {
      if (std::ifstream(camera_pose_name)) {
        std::vector<std::vector<float>> camera_poses_v;
        std::ifstream fPoses;
        fPoses.open(camera_pose_name.c_str());
        int counter = 0;
        while (!fPoses.eof()) {
          std::vector<float> camera_pose_v;
          std::string s;
          std::getline(fPoses, s);
          if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            float t;
            for (int i = 0; i < 12; ++i) {
              ss >> t;
              camera_pose_v.push_back(t);
            }
            camera_poses_v.push_back(camera_pose_v);
            counter++;
          }
        }
        fPoses.close();
        camera_poses_.resize(counter, 12);
        for (int c = 0; c < counter; ++c) {
          for (int i = 0; i < 12; ++i)
            camera_poses_(c, i) = camera_poses_v[c][i];
        }
        return true;
     } else {
       ROS_ERROR_STREAM("Cannot open camera pose file " << camera_pose_name);
       return false;
     }
    }

    void publish_semantic_map() {
      sm_pub_->clear();
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          int semantics = it.get_node().get_semantics();
          sm_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), semantics, it.get_size());
        }
      }
      sm_pub_->publish();
    }
   
  private:
    ros::NodeHandle nh_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    float depth_scaling_;
    int num_class_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;

    Eigen::MatrixXf camera_poses_;

    la3dm::BGKOctoMap* map_;
    la3dm::MarkerArrayPub* sm_pub_;
    la3dm::MarkerArrayPub* tm_pub_;

    Eigen::Matrix4f get_current_pose(const int scan_id) {
      Eigen::VectorXf curr_pose_v = camera_poses_.row(scan_id);
      Eigen::MatrixXf curr_pose = Eigen::Map<MatrixXf_row>(curr_pose_v.data(), 3, 4);
      Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
      transform.block(0, 0, 3, 4) = curr_pose;
      //Eigen::Matrix4f new_transform = init_trans_to_ground_ * transform;
      return transform;
    }
};
