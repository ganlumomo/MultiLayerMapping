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

    void process_scans(int scan_num, std::string depth_img_dir, 
                       std::string semantic_img_dir, std::string traversability_img_dir, std::string reproj_img_dir) {
      for (int scan_id = 0; scan_id <= scan_num; ++scan_id) {
        char scan_id_c[256];
        sprintf(scan_id_c, "%06d", scan_id);
        std::string scan_id_s(scan_id_c);
        std::string depth_img_name(depth_img_dir + scan_id_s + ".png");
        std::string semantic_img_name(semantic_img_dir + scan_id_s + ".png");
        std::string traversability_img_name(traversability_img_dir + scan_id_s + ".png");

        cv::Mat depth_img = cv::imread(depth_img_name, CV_LOAD_IMAGE_ANYDEPTH);
        // save depth img if reproject current scan
        int reproj_id = check_element_in_vector(scan_id, evaluation_list_);
        if (reproj_id >= 0)
          depth_imgs_.push_back(depth_img);

        cv::Mat semantic_img = cv::imread(semantic_img_name, CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat traversability_img = cv::imread(traversability_img_name, CV_LOAD_IMAGE_UNCHANGED);
        Eigen::Matrix4f transform = get_scan_pose(scan_id);

        la3dm::PCLPointCloudwithLabel cloudwlabel;
        la3dm::point3f origin;
        process_scan(depth_img, semantic_img, transform, cloudwlabel, origin);
        map_->insert_semantics(cloudwlabel, origin, ds_resolution_, free_resolution_, max_range_, num_class_);
        publish_semantic_map();
       
        process_scan(depth_img, traversability_img, transform, cloudwlabel, origin);
        map_->insert_traversability(cloudwlabel, origin, ds_resolution_, free_resolution_, max_range_);
        publish_traversability_map();
        //cloudwlabel.width = cloudwlabel.points.size();
        //cloudwlabel.height = 1;
        //pcl::io::savePCDFileASCII ("test_pcd.pcd", cloudwlabel);
        
        // reprojection
        reproject_imgs(scan_id, reproj_img_dir);
      }
    }

    void process_scan(const cv::Mat& depth_img, const cv::Mat& label_img, const Eigen::Matrix4f& transform,
                      la3dm::PCLPointCloudwithLabel& cloudwlabel, la3dm::point3f& origin) {
      
      int width = depth_img.cols;
      int height = depth_img.rows;
      
      cloudwlabel.points.clear();
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
          ptl.label = pix_label;
          cloudwlabel.points.push_back(ptl);
        }
      }
      pcl::transformPointCloud(cloudwlabel, cloudwlabel, transform);
      origin.x() = transform(0, 3);
      origin.y() = transform(1, 3);
      origin.z() = transform(2, 3);
    }

    void reproject_imgs(const int current_scan_id, std::string reproj_img_dir) {
      if (check_element_in_vector(current_scan_id, evaluation_list_) < 0)
        return;
      for (int reproj_id = 0; reproj_id < evaluation_list_.rows(); ++reproj_id) {
        int scan_id = evaluation_list_[reproj_id];
        if (scan_id <= current_scan_id) {
          cv::Mat reproj_img;
          reproject_img(scan_id, reproj_id, reproj_img, reproj_img_dir);
        }
      }
    }

    void reproject_img(const int scan_id, const int reproj_id, cv::Mat& reproj_img, std::string reproj_img_dir) {
      std::cout << scan_id << " " << reproj_id << std::endl;
      int width = depth_imgs_[reproj_id].cols;
      int height = depth_imgs_[reproj_id].rows;
      reproj_img = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
      Eigen::Matrix4f transform = get_scan_pose(scan_id);
      
      la3dm::PCLPointCloud cloud;
      for (int32_t i = 0; i < width * height; ++i) {
        int ux = i % width;
        int uy = i / width;
        
        float pix_depth = (float) depth_imgs_[reproj_id].at<uint16_t>(uy, ux) / depth_scaling_;

        if (pix_depth > 0.1) {
          la3dm::PCLPointType pt;
          pt.x = (ux - cx_) * (1.0 / fx_) * pix_depth;
          pt.y = (uy - cy_) * (1.0 / fy_) * pix_depth;
          pt.z = pix_depth;

          transform_pt_to_global(transform, pt);
          cloud.push_back(pt);
          la3dm::OcTreeNode node = map_->search(pt.x, pt.y, pt.z);
          if (node.get_state() == la3dm::State::OCCUPIED) {
            int semantics = node.get_semantics();
            std::cout << semantics << std::endl;
            reproj_img.at<uint8_t>(uy, ux) = (uint8_t) semantics - 1;
          }
        }
      }
        cloud.width = cloud.points.size();
        cloud.height = 1;
        pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
 
          char scan_id_c[256];
          sprintf(scan_id_c, "%06d", scan_id);
          std::string scan_id_s(scan_id_c);
          std::string reproj_img_name(reproj_img_dir + scan_id_s + ".png");
          cv::imwrite(reproj_img_name, reproj_img);

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

    bool read_evaluation_list(const std::string evaluation_list_name) {
      if (std::ifstream(evaluation_list_name)) {
        std::vector<int> evaluation_list_v;
        std::ifstream fImgs;
        fImgs.open(evaluation_list_name.c_str());
        int counter = 0;
        while (!fImgs.eof()) {
          std::string s;
          std::getline(fImgs, s);
          if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            int t;
            ss >> t;
            evaluation_list_v.push_back(t);
            counter++;
          }
        }
        fImgs.close();
        evaluation_list_.resize(counter);
        for (int c = 0; c < counter; ++c)
          evaluation_list_(c) = evaluation_list_v[c];
        return true;
      } else {
        ROS_ERROR_STREAM("Cannot open evaluation list file " << evaluation_list_name);
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

    void publish_traversability_map() {
      tm_pub_->clear();
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          float traversability = it.get_node().get_prob_traversability();
          tm_pub_->insert_point3d_traversability(p.x(), p.y(), p.z(), traversability, it.get_size());
        }
      }
      tm_pub_->publish();
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
    Eigen::VectorXi evaluation_list_;
    std::vector<cv::Mat> depth_imgs_;

    la3dm::BGKOctoMap* map_;
    la3dm::MarkerArrayPub* sm_pub_;
    la3dm::MarkerArrayPub* tm_pub_;

    int check_element_in_vector(const int element, const Eigen::VectorXi& vec_check) {
      for (int i = 0; i < vec_check.rows(); ++i)
        if (element == vec_check(i) )
          return i;
      return -1;
    }

    Eigen::Matrix4f get_scan_pose(const int scan_id) {
      Eigen::VectorXf curr_pose_v = camera_poses_.row(scan_id);
      Eigen::MatrixXf curr_pose = Eigen::Map<MatrixXf_row>(curr_pose_v.data(), 3, 4);
      Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
      transform.block(0, 0, 3, 4) = curr_pose;
      return transform;
    }

    void transform_pt_to_global(const Eigen::Matrix4f& transform, la3dm::PCLPointType& pt) {
      Eigen::Vector4f global_pt_4 = transform * Eigen::Vector4f(pt.x, pt.y, pt.z, 1);
      Eigen::Vector3f global_pt_3 = global_pt_4.head(3) / global_pt_4(3);
      pt.x = global_pt_3(0);
      pt.y = global_pt_3(1);
      pt.z = global_pt_3(2);
    }
};
