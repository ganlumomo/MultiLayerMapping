#pragma once

#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/common/transforms.h>

class CassieData {
  public:
    CassieData(ros::NodeHandle& nh, std::string static_frame, std::string smap_topic,
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
        if (ptl.label == 0)
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
      ROS_INFO_STREAM("Scan " << cloud_msg->header.stamp.toNSec() << " done.");

      // Visualize maps
      publish_semantic_map();

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
    std::string static_frame_;
    tf::TransformListener listener_;
    int num_class_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;

    la3dm::BGKOctoMap* map_;
    la3dm::MarkerArrayPub* sm_pub_;
};

