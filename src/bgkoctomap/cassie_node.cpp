#include <string>
#include <iostream>
#include <ros/ros.h>
#include "bgkoctomap.h"
#include "markerarray_pub.h"
#include "cassie_util.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "cassie_node");
    ros::NodeHandle nh("~");

    int num_class = 2;
    std::string static_frame("/odom");
    std::string dir;
    std::string prefix;
    int scan_num = 0;
    std::string map_topic("/semantic_map");
    std::string traversability_map_topic("/traversability_map");
    double max_range = -1;
    double resolution = 0.1;
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    double free_resolution = 0.5;
    double ds_resolution = 0.1;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    double min_z = 0;
    double max_z = 0;
    bool original_size = false;
    float var_thresh = 1.0f;
    float prior_A = 1.0f;
    float prior_B = 1.0f;
    float prior = 1.0f;

    nh.param<int>("num_class", num_class, num_class);
    nh.param<std::string>("static_frame", static_frame, static_frame);
    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("prefix", prefix, prefix);
    nh.param<std::string>("topic", map_topic, map_topic);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);
    nh.param<bool>("original_size", original_size, original_size);
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<float>("prior_A", prior_A, prior_A);
    nh.param<float>("prior_B", prior_B, prior_B);
    nh.param<float>("prior", prior, prior);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
            "num_class: " << num_class << std::endl <<
            "static_frame: " << static_frame << std::endl <<
            "dir: " << dir << std::endl <<
            "prefix: " << prefix << std::endl <<
            "topic: " << map_topic << std::endl <<
            "scan_sum: " << scan_num << std::endl <<
            "max_range: " << max_range << std::endl <<
            "resolution: " << resolution << std::endl <<
            "block_depth: " << block_depth << std::endl <<
            "sf2: " << sf2 << std::endl <<
            "ell: " << ell << std::endl <<
            "free_resolution: " << free_resolution << std::endl <<
            "ds_resolution: " << ds_resolution << std::endl <<
            "free_thresh: " << free_thresh << std::endl <<
            "occupied_thresh: " << occupied_thresh << std::endl <<
            "min_z: " << min_z << std::endl <<
            "max_z: " << max_z << std::endl <<
            "original_size: " << original_size << std::endl <<
            "var_thresh: " << var_thresh << std::endl <<
            "prior_A: " << prior_A << std::endl <<
            "prior_B: " << prior_B << std::endl <<
            "prior: " << prior
            );

    CassieData cassie_data(nh, static_frame, map_topic, 
        resolution, block_depth, num_class,
        sf2, ell, free_thresh, occupied_thresh, var_thresh,
        prior_A, prior_B, prior,
        ds_resolution, free_resolution, max_range);

    ros::Subscriber sub = nh.subscribe("/labeled_pointcloud", 1, &CassieData::SemanticPointCloudCallback, &cassie_data);

    ros::spin();

    return 0;
}
