#pragma once

#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/common/transforms.h>

float max_semantic_var = std::numeric_limits<float>::min();
float min_semantic_var = std::numeric_limits<float>::max();
float max_traversability_var = std::numeric_limits<float>::min();
float min_traversability_var = std::numeric_limits<float>::max();

class CassieData {
  public:
    CassieData(ros::NodeHandle& nh, std::string static_frame, std::string smap_topic, std::string tmap_topic,
        double resolution, double block_depth, int num_class,
        double sf2, double ell, double free_thresh, double occupied_thresh, double var_thresh,
        double prior_A, double prior_B, double prior,
        double ds_resolution, double free_resolution, double max_range)
    : nh_(nh)
    , static_frame_(static_frame)
    , num_class_(num_class)
    , ds_resolution_(ds_resolution)
    , free_resolution_(free_resolution)
    , max_range_(max_range) {
      map_ = new la3dm::BGKOctoMap(resolution, block_depth, num_class_,
                                   sf2, ell, free_thresh, occupied_thresh, var_thresh,
                                   prior_A, prior_B, prior);
      sm_pub_ = new la3dm::MarkerArrayPub(nh_, smap_topic, resolution);
      tm_pub_ = new la3dm::MarkerArrayPub(nh_, tmap_topic, resolution);
      sv_pub_ = new la3dm::MarkerArrayPub(nh_, "semantic_variance_map", resolution);
      tv_pub_ = new la3dm::MarkerArrayPub(nh_, "traversability_variance_map", resolution);
    }

    void SemanticPointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg) {
      la3dm::PCLPointCloudwithLabel cloudwlabel;
      la3dm::point3f origin;

      // Read point cloud
      for (int i = 0; i < cloud_msg->points.size(); ++i) {
        la3dm::PCLPointwithLabel ptl;
        ptl.x = cloud_msg->points[i].x;
        ptl.y = cloud_msg->points[i].y;
        ptl.z = cloud_msg->points[i].z;
        ptl.label = cloud_msg->channels[0].values[i];

        if (std::isnan(ptl.x) || std::isnan(ptl.y) || std::isnan(ptl.z))
          continue;
        if (ptl.label == 0 || ptl.label == 13 || ptl.label == 8 || 
            ptl.label == 1 || ptl.label == 7 || ptl.label == 9 || 
            ptl.label == 11 || ptl.label == 12)  // Note: don't project background and sky
          continue;
        cloudwlabel.push_back(ptl);
      }

      // Fetch tf transform
      tf::StampedTransform transform;
      try {
        listener_.lookupTransform(static_frame_,
                                  cloud_msg->header.frame_id,
                                  cloud_msg->header.stamp,
                                  transform);
      } catch (tf::TransformException ex) {
        ROS_ERROR("%s",ex.what());
        return;
      }
      Eigen::Affine3d tf_eigen;
      tf::transformTFToEigen(transform, tf_eigen);

      // Transform point cloud
      pcl::transformPointCloud(cloudwlabel, cloudwlabel, tf_eigen);
      origin.x() = tf_eigen.matrix()(0, 3);
      origin.y() = tf_eigen.matrix()(1, 3);
      origin.z() = tf_eigen.matrix()(2, 3);
      map_->insert_semantics(cloudwlabel, origin, ds_resolution_, free_resolution_, max_range_, num_class_);
      la3dm::PCLPointCloudwithLabel new_cloudwlabel;
      map_->get_training_data_semantic_traversability(cloudwlabel, new_cloudwlabel);
      map_->insert_traversability(new_cloudwlabel, origin, ds_resolution_, free_resolution_, max_range_);

      // Visualize maps
      publish_semantic_map();
      //publish_semantic_variance_map();

    }

    void TraversabilityPointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg) {
      la3dm::PCLPointCloudwithLabel cloudwlabel;
      la3dm::point3f origin;

      // Read point cloud
      for (int i = 0; i < cloud_msg->points.size(); ++i) {
        la3dm::PCLPointwithLabel ptl;
        ptl.x = cloud_msg->points[i].x;
        ptl.y = cloud_msg->points[i].y;
        ptl.z = cloud_msg->points[i].z;
        ptl.label = cloud_msg->channels[0].values[i];

        if (std::isnan(ptl.x) || std::isnan(ptl.y) || std::isnan(ptl.z))
          continue;
        cloudwlabel.push_back(ptl);
      }

      // Fetch tf transform
      tf::StampedTransform transform;
      try {
        listener_.lookupTransform(static_frame_,
                                  cloud_msg->header.frame_id,
                                  cloud_msg->header.stamp,
                                  transform);
      } catch (tf::TransformException ex) {
        ROS_ERROR("%s",ex.what());
        return;
      }
      Eigen::Affine3d tf_eigen;
      tf::transformTFToEigen(transform, tf_eigen);

      // Transform point cloud
      pcl::transformPointCloud(cloudwlabel, cloudwlabel, tf_eigen);
      origin.x() = tf_eigen.matrix()(0, 3);
      origin.y() = tf_eigen.matrix()(1, 3);
      origin.z() = tf_eigen.matrix()(2, 3);
      map_->insert_traversability(cloudwlabel, origin, ds_resolution_, free_resolution_, max_range_);

      // Visualize maps
      publish_traversability_map();
      //publish_traversability_variance_map();
    }

    void publish_semantic_map() {
      sm_pub_->clear();
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          int semantics = it.get_node().get_semantics();
          sm_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), semantics, it.get_size());
	  /*
	  // get variance
	  std::vector<float> vars(num_class_);
	  it.get_node().get_vars(vars);
	  if (vars[semantics] > max_semantic_var)
	    max_semantic_var = vars[semantics];
          if (vars[semantics] < min_semantic_var)
            min_semantic_var = vars[semantics];
	  */
        }
      }
      sm_pub_->publish();
      //std::cout << "max_semantic_var: " << max_semantic_var << std::endl;
      //std::cout << "min_semantic_var: " << min_semantic_var << std::endl;
    }

    void publish_semantic_variance_map() {
      sv_pub_->clear();
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          int semantics = it.get_node().get_semantics();
          std::vector<float> vars(num_class_);
          it.get_node().get_vars(vars);
          sv_pub_->insert_point3d_variance(p.x(), p.y(), p.z(), 0, max_semantic_var, it.get_size(), vars[semantics]);
        }
      }
      sv_pub_->publish();
    }

    void publish_traversability_map() {
      tm_pub_->clear();
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          float traversability = it.get_node().get_prob_traversability();
          tm_pub_->insert_point3d_traversability(p.x(), p.y(), p.z(), traversability, it.get_size());
	  /*
	  // get variance
	  float var = it.get_node().get_var_traversability();
	  if (var > max_traversability_var)
            max_traversability_var = var;
          if (var < min_traversability_var)
            min_traversability_var = var;
	  */
        }
      }
      tm_pub_->publish();
      //std::cout << "max_traversability_var: " << max_traversability_var << std::endl;
      //std::cout << "min_traversability_var: " << min_traversability_var << std::endl;
    }

    void publish_traversability_variance_map() {
      tv_pub_->clear();
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          float var = it.get_node().get_var_traversability();
          tv_pub_->insert_point3d_variance(p.x(), p.y(), p.z(), 0, max_traversability_var, it.get_size(), var);
        }
      }
      tv_pub_->publish();
    }

  private:
    ros::NodeHandle nh_;
    std::string static_frame_;
    tf::TransformListener listener_;
    int num_class_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;

    la3dm::BGKOctoMap* map_;
    la3dm::MarkerArrayPub* sm_pub_;
    la3dm::MarkerArrayPub* tm_pub_;
    la3dm::MarkerArrayPub* sv_pub_;
    la3dm::MarkerArrayPub* tv_pub_;

    void process_pcd(const la3dm::PCLPointCloudwithLabel &cloudwlabel, la3dm::PCLPointCloud &cloud) {
      for (auto it = cloudwlabel.begin(); it != cloudwlabel.end(); ++it) {
        la3dm::PCLPointType p;
        p.x = it->x;
        p.y = it->y;
        p.z = it->z;
        cloud.push_back(p);
      }
    }
};

