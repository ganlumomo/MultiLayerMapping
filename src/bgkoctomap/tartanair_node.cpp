#include <string>
#include <iostream>
#include <ros/ros.h>
#include "bgkoctomap.h"
#include "markerarray_pub.h"
#include "tartanair_util.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "tartanair_node");
    ros::NodeHandle nh("~");

    int num_class = 2;
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

    // KITTI 05
    float fx = 707.0912;
    float fy = 707.0912;
    float cx = 601.8873;
    float cy = 183.1104;
    float depth_scaling = 1000;
    std::string camera_pose_file = "pose_left.txt";
    std::string depth_img_folder = "depth_left/";
    std::string semantic_img_folder = "seg_left/";
    std::string traversability_img_folder = "traversability_gt/";
    std::string evaluation_list_file = "evaluatioList.txt";
    std::string reproj_traversability_folder = "traversability_reproj/";
    std::string reproj_semantics_folder = "semantic_reproj/";

    nh.param<int>("num_class", num_class, num_class);
    nh.param<float>("fx", fx, fx);
    nh.param<float>("fy", fy, fy);
    nh.param<float>("cx", cx, cx);
    nh.param<float>("cy", cy, cy);
    nh.param<float>("depth_scaling", depth_scaling, depth_scaling);
    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("traversability_img_folder", traversability_img_folder, traversability_img_folder);    
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
            "fx: " << fx << std::endl <<
            "fy: " << fy << std::endl <<
            "cx: " << cx << std::endl <<
            "cy: " << cy << std::endl <<
            "depht_scaling: " << depth_scaling << std::endl <<
            "dir: " << dir << std::endl <<
            "traversability_img_folder: " << traversability_img_folder << std::endl << 
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

    TartanAirData tartanair_data(nh, map_topic, traversability_map_topic,
        fx, fy, cx, cy, depth_scaling,
        resolution, block_depth, num_class,
        sf2, ell, free_thresh, occupied_thresh, var_thresh,
        prior_A, prior_B, prior,
        ds_resolution, free_resolution, max_range);

    // preprocessing data
    std::string camera_pose_name(dir + camera_pose_file);
    tartanair_data.read_camera_poses(camera_pose_name);
    std::string evaluation_list_name(dir + evaluation_list_file);
    //tartanair_data.read_evaluation_list(evaluation_list_name);

    // process scans
    std::string depth_img_dir(dir + depth_img_folder);
    std::string semantic_img_dir(dir + semantic_img_folder);
    std::string traversability_img_dir(dir + traversability_img_folder);
    std::string reproj_traversability_dir(dir + reproj_traversability_folder);
    std::string reproj_semantics_dir(dir + reproj_semantics_folder);
    std::string rgb_img_dir(dir + "image_left/");
    tartanair_data.process_scans(scan_num, depth_img_dir, semantic_img_dir, traversability_img_dir, reproj_traversability_dir, reproj_semantics_dir, rgb_img_dir);

    ros::spin();

    return 0;
}
