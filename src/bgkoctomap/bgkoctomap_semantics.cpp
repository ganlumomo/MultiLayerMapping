#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include "bgkoctomap.h"
#include "bgkinference.h"

using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace la3dm {

    BGKOctoMap::BGKOctoMap() : BGKOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        2, // num_class
                                        1.0, // sf2
                                        1.0, // ell
                                        0.3f, // free_thresh
                                        0.7f, // occupied_thresh
                                        1.0f, // var_thresh
                                        1.0f, // prior_A
                                        1.0f, // prior_B
                                        1.0f // prior
                                    ) { }

    BGKOctoMap::BGKOctoMap(float resolution,
                        unsigned short block_depth,
                        int num_class,
                        float sf2,
                        float ell,
                        float free_thresh,
                        float occupied_thresh,
                        float var_thresh,
                        float prior_A,
                        float prior_B,
                        float prior)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);

        // Note: Bug fixed
        Block::cell_num = static_cast<unsigned short>(round(Block::size / Block::resolution));

        OcTree::max_depth = block_depth;

        OcTreeNode::num_class = num_class;
        OcTreeNode::sf2 = sf2;
        OcTreeNode::ell = ell;
        OcTreeNode::free_thresh = free_thresh;
        OcTreeNode::occupied_thresh = occupied_thresh;
        OcTreeNode::var_thresh = var_thresh;
        OcTreeNode::prior_A = prior_A;
        OcTreeNode::prior_B = prior_B;
        OcTreeNode::prior = prior;
    }

    void BGKOctoMap::insert_semantics(const PCLPointCloudwithLabel &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range, int num_class) {

#ifdef DEBUG
        Debug_Msg("Insert semantics: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data_semantics(cloud, origin, ds_resolution, free_res, max_range, xy);
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, BGK3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            }
            BGK3f *bgk = new BGK3f(OcTreeNode::sf2, OcTreeNode::ell, num_class);
            bgk->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }
#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk == bgk_arr.end())
                    continue;

                vector<vector<float>> ybars;
                bgk->second->predict(xs, ybars);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    auto node_loc = block->get_loc(leaf_it);
                    //if (node_loc.x() == 7.45 && node_loc.y() == 10.15 && node_loc.z() == 1.15) {
                    //    std::cout << "updating the node " << ybar[j] << " " << kbar[j] << std::endl;
                    //}

                    // Only need to update if kernel density total kernel density est > 0
                    //if (kbar[j] > 0.0)
                        node.update(ybars[j]);
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif
        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        /////////////////////////////////////////////////
// #ifdef OPENMP
// #pragma omp parallel for
// #endif
//         for (int i = 0; i < test_blocks.size(); ++i) {
//             BlockHashKey key = test_blocks[i];
//             auto block = block_arr.find(key);
//             if (block == block_arr.end())
//                 continue;
//             block->second->prune();
//         }
// #ifdef DEBUG
//         Debug_Msg("Pruning done");
// #endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void BGKOctoMap::get_training_data_semantics(const PCLPointCloudwithLabel &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {
	PCLPointCloudwithLabel sampled_hits;
	downsample(cloud, sampled_hits, ds_resolution);

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        xy.clear();
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);
            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            xy.emplace_back(p, it->label);  // Note: label 0 is for free class

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);

            frees.push_back(PCLPointType(origin.x(), origin.y(), origin.z()));
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
                frees.push_back(PCLPointType(p->x(), p->y(), p->z()));
                frees.width++;
            }
        }

        PCLPointCloud sampled_frees;
        downsample(frees, sampled_frees, ds_resolution);

        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.emplace_back(point3f(it->x, it->y, it->z), 0.0f);
        }
    }

}
