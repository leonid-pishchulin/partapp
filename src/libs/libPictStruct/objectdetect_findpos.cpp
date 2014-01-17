/** 
    This file is part of the implementation of the human pose estimation model as described in the paper:
    
    Leonid Pishchulin, Micha Andriluka, Peter Gehler and Bernt Schiele
    Strong Appearance and Expressive Spatial Models for Human Pose Estimation
    IEEE International Conference on Computer Vision (ICCV'13), Sydney, Australia, December 2013

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  
*/

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <libAnnotation/annotationlist.h>

#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>


#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_transform.hpp>
#include <libMultiArray/multi_array_filter.hpp>

#include <libPartApp/partapp_aux.hpp>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartDetect/partdetect.h>

#include "objectdetect.h"
#include "objectdetect_aux.hpp"

//#include "objectdetect_aux.hpp"

using boost_math::double_vector;
using boost_math::double_matrix;
using boost_math::double_zero_matrix;

using boost::multi_array_types::index_range;

using namespace boost::lambda;
using namespace std;

namespace object_detect {

  /**
     could be implemeted as template (i.e. to work for arrays and views at the same time)

     C:  covariance matrix of the position of child w.r.t parent
     offset: child_position - parent_position (i.e. offset_p in the joint structure)

     note: at the marginalization step we use unnormalized gaussian filter - this corresponds to implicitly
     mapping all positions to the training scale
   */
  void computePosJointMarginal(FloatGrid2 &log_prob_child, FloatGrid2 &log_prob_parent, 
                               double_vector offset, double_matrix C, double scale, bool bIsSparse)
  {
    //cout << "computePosJointMarginal" << endl;
    assert(scale > 0);

    /** rescale the spatial model */
    offset *= scale;
    C *= square(scale);
    
    /** convert to prob for marginalization */
    multi_array_op::computeExpGrid(log_prob_child);

    assert(log_prob_child.shape()[0] == log_prob_parent.shape()[0]);
    assert(log_prob_child.shape()[1] == log_prob_parent.shape()[1]);

    assert(offset.size() == 2);
    assert(C.size1() == 2 && C.size2() == 2);

    bool bNormalize = false;
    multi_array_op::gaussFilter2dOffset(log_prob_child, log_prob_parent, C, offset, bNormalize, bIsSparse);

    /** convert back to log prob */
    multi_array_op::computeLogGrid(log_prob_parent);    
    multi_array_op::computeLogGrid(log_prob_child);
  }
  
  /** remove the rotation dimension by replacing it with sum of values along the rotations dimension */
  void mergeRotationsSum(const vector<FloatGrid2> &rot_grid, FloatGrid2 &result) 
  {
    int nRotations = rot_grid.size();
    assert(nRotations > 0);

    int grid_width = rot_grid[0].shape()[1];
    int grid_height = rot_grid[0].shape()[0];

    result.resize(boost::extents[grid_height][grid_width]);

    for (int ix = 0; ix < grid_width; ++ix) {
      for (int iy = 0; iy < grid_height; ++iy) {
        long double sum_prob = 0;

        for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
          sum_prob += exp(rot_grid[rotidx][iy][ix]);
        }

        result[iy][ix] = (float)log(sum_prob);
      }
    }
  }
  
  /** 
      note: elements of part_score_grid_roation are assumed to be log probabilities
   */
  void mergeRotations(const PartApp &part_app, const PartConfig &part_conf, 
                      const vector<vector<vector<FloatGrid2> > > &part_score_grid_rotation, 
                      vector<vector<FloatGrid2> > &log_part_detections)
  {
    cout << "mergeRotations..." << endl;
    cout << "part_score_grid_rotation.size() " << part_score_grid_rotation.size() << endl;

    int nParts = part_app.m_part_conf.part_size();
    cout << "nParts: " << nParts << endl;

    int nScales = part_app.m_exp_param.num_scale_steps();

    int img_width = -1;
    int img_height = -1;

    assert((int)part_score_grid_rotation.size() == nParts);

    for (int pidx = 0; pidx < nParts; ++pidx) 
      if (part_app.m_part_conf.part(pidx).is_detect()) {
        cout << "pidx " << pidx << endl;

        assert(part_score_grid_rotation[pidx].size() > 0);
        img_width = part_score_grid_rotation[pidx][0][0].shape()[1];
        img_height = part_score_grid_rotation[pidx][0][0].shape()[0];
      }

    assert(img_width > 0 && img_height > 0);

    cout << "img_width: " << img_width << ", img_height: " << img_height << endl;

    for (int pidx = 0; pidx < nParts; ++pidx) {
      if (!part_conf.part(pidx).is_detect()) {
        cout << "skip part (is_detect = false): " << pidx << endl;

        // only for debug purposes (to avoid assert in mat_save_multi_array_vec2)
        for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
          log_part_detections[pidx].push_back(FloatGrid2(boost::extents[img_height][img_width]));

        continue;
      }

      for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {
        FloatGrid2 scoregrid;
        mergeRotationsSum(part_score_grid_rotation[pidx][scaleidx], scoregrid);

        log_part_detections[pidx].push_back(scoregrid);
      }
      
      cout << "log_part_detections, pidx: " << pidx << ", size: " << log_part_detections[pidx].size() << endl;
    }// parts

    cout << "done." << endl;
  }

  void computeRootPosterior(const PartApp part_app, const vector<vector<FloatGrid2> > &log_part_detections, 
                            FloatGrid3 &root_part_posterior, int rootpart_idx, vector<Joint> joints, bool flip, 
                            QString qsDebugDir, bool bIsSparse)
  {
    bool bDebugMessageOutput = false;
    bool bDebugSaveChildMessages = false;
    bool bDebugOutput = false;

    if (bDebugOutput || bDebugSaveChildMessages) 
      filesys::create_dir(qsDebugDir);

    int nParts = part_app.m_part_conf.part_size();

    //int nParts = log_part_detections.size();
    cout << "nParts: " << nParts << endl;

    int nScales = part_app.m_exp_param.num_scale_steps();
    //int nScales = log_part_detections[0].size();
    
    int img_width = -1;
    int img_height = -1;

    assert((int)log_part_detections.size() == nParts);

    for (int pidx = 0; pidx < nParts; ++pidx) 
      if (part_app.m_part_conf.part(pidx).is_detect()) {
        cout << "pidx " << pidx << endl;

        assert(log_part_detections[pidx].size() > 0);
        img_width = log_part_detections[pidx][0].shape()[1];
        img_height = log_part_detections[pidx][0].shape()[0];
      }

    assert(img_width > 0 && img_height > 0);

    FloatGrid4 log_part_posterior_all(boost::extents[nParts][nScales][img_height][img_width]);

    /** compute object probability for each scale */
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {
      double scale = scale_from_index(part_app.m_exp_param, scaleidx);

      if (bDebugMessageOutput)
        cout << "processing scale " << scaleidx << "(" << scale << ")" << endl;

      /* after upstream pass: message from downstream combined with appearance model */
      /* after downstream pass: part posterior                                       */
      vector<FloatGrid2> log_part_posterior(nParts, 
                                            FloatGrid2(boost::extents[img_height][img_width]));


      /** keep track of messages from branches with all parts having is_detect == false */

      vector<bool> is_uniform_message(nParts, true);

      vector<bool> vComputedPosteriors(nParts, false);
      vector<int> compute_stack;
      compute_stack.push_back(rootpart_idx);

      while (!compute_stack.empty()) {
        bool bCanCompute = true;

        int curidx = compute_stack.back();
        compute_stack.pop_back();

        if (bDebugMessageOutput) 
          cout << "curidx: " << curidx << endl;

        for (uint jidx = 0; jidx < joints.size(); ++jidx) {
          if (joints[jidx].parent_idx == curidx) {
            if (vComputedPosteriors[joints[jidx].child_idx] == false) {
              bCanCompute = false;
              compute_stack.push_back(curidx);
              compute_stack.push_back(joints[jidx].child_idx);

              if (bDebugMessageOutput)
                cout << "push: " << curidx << ", " << joints[jidx].child_idx << endl;
              break;
            }          
          }
        }// joints


        if (bCanCompute) {
          if (bDebugMessageOutput)
            cout << "computing posterior for " << curidx << endl;

          if (curidx == rootpart_idx) 
            assert(compute_stack.empty());

          vector<int> all_children;
          vector<int> incoming_joints;

          /* list of incoming joints and children                        */
          /* this is needed to compute messages from root to child notes */
          for (uint jidx = 0; jidx < joints.size(); ++jidx) 
            if (joints[jidx].parent_idx == curidx) {
              all_children.push_back(joints[jidx].child_idx);
              incoming_joints.push_back(jidx);
            }

          for (int i = 0; i < (int)all_children.size(); ++i) {

            int child_idx = all_children[i];
            int jidx = incoming_joints[i];

            if (!is_uniform_message[child_idx]) {

              if (bDebugMessageOutput)
                cout << "\tcomputing component from " << child_idx << endl;

              assert(vComputedPosteriors[child_idx]);

              FloatGrid2 log_part_posterior_from_child(boost::extents[img_height][img_width]);
              computePosJointMarginal(log_part_posterior[child_idx], log_part_posterior_from_child, 
                                      joints[jidx].offset_p, joints[jidx].C, scale, bIsSparse);

              if (bDebugSaveChildMessages) {
                QString qsFilename = qsDebugDir + "/log_part_posterior_from_child_scaleidx" + QString::number(scaleidx) + 
                  "_childidx" + QString::number(child_idx) + 
                  "_parentidx" + QString::number(curidx) + 
                  + "_flip" + QString::number(flip) + ".mat";

                if (bDebugMessageOutput)
                  cout << "saving " << qsFilename.toStdString() << endl;

                matlab_io::mat_save_multi_array(qsFilename, "log_part_posterior_from_child", log_part_posterior_from_child);
              }

              multi_array_op::addGrid2(log_part_posterior[curidx], log_part_posterior_from_child);
              is_uniform_message[curidx] = false;

            }// is_uniform
            else {
              cout << "skip child " << child_idx << ", message is uniform" << endl;
            }

          }// children

          if (part_app.m_part_conf.part(curidx).is_detect()) {
            multi_array_op::addGrid2(log_part_posterior[curidx], log_part_detections[curidx][scaleidx]);
            is_uniform_message[curidx] = false;
          }

          vComputedPosteriors[curidx] = true;
        }// if can compute
      }// stack
      
      for (int pidx = 0; pidx < nParts; ++pidx) 
        log_part_posterior_all[boost::indices[pidx][scaleidx][index_range()][index_range()]] = log_part_posterior[pidx];

    }// scales

    if (bDebugOutput) {
      QString qsFilename = qsDebugDir + "/log_part_posterior_flip" + QString::number(flip) + ".mat";
      cout << "saving " << qsFilename.toStdString() << endl;
      MATFile *f = matlab_io::mat_open(qsFilename, "wz");
      matlab_io::mat_save_multi_array(f, "log_part_posterior", log_part_posterior_all);
      matlab_io::mat_close(f);
    }

    root_part_posterior.resize(boost::extents[nScales][img_height][img_width]);
    root_part_posterior = log_part_posterior_all[rootpart_idx];
  }

  void findObjectImagePosJoints(const PartApp &part_app, int imgidx, bool flip, HypothesisList &hypothesis_list, int scoreProbMapType)
  {
    cout << "findObjectImagePosJoints" << endl;

    const PartConfig &part_conf = part_app.m_part_conf;
    bool bDebugOutput = false;
    bool bIsSparse = true; 

    /** load joints */
    int nJoints = part_conf.joint_size();
    cout << "nJoints: " << nJoints << endl;

    int nParts = part_conf.part_size();
    cout << "nParts: " << nParts << endl;

    vector<Joint> joints(nJoints);
    loadJoints(part_app, joints, flip);

    /** load classifier scores */

    vector<vector<vector<FloatGrid2> > > part_score_grid_rotation(nParts, vector<vector<FloatGrid2> >());

    int rootpart_idx = -1;
    for (int pidx = 0; pidx < nParts; ++pidx) {

      if (part_conf.part(pidx).is_root()) {
        assert(rootpart_idx == -1);
        rootpart_idx = pidx;
      }

      if (part_conf.part(pidx).is_detect()) {
        int nScales = part_app.m_exp_param.num_scale_steps();
        int nRotations = part_app.m_exp_param.num_rotation_steps();

        bool bInterpolate = false;


        if (scoreProbMapType == SPMT_NONE) {
          //part_app.loadScoreGrid(part_score_grid_rotation[pidx], imgidx, pidx, flip, bLoadLogP, lp_version, bInterpolate);
	  assert(part_app.m_exp_param.has_scoregrid_dir());
	  QString qsScoreGridDir = part_app.m_exp_param.scoregrid_dir().c_str(); 
	  
          part_app.loadScoreGrid(part_score_grid_rotation[pidx], imgidx, pidx, flip, bInterpolate, qsScoreGridDir, part_app.m_test_annolist[imgidx].imageName().c_str());

          for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
            for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
              //multi_array_op::clip_scores(part_score_grid_rotation[pidx][scaleidx][rotidx], 0.0);
              object_detect::clip_scores_fill(part_score_grid_rotation[pidx][scaleidx][rotidx]);
              multi_array_op::computeLogGrid(part_score_grid_rotation[pidx][scaleidx][rotidx]);
            }
          
        }
        else {
          assert(false && "unknown score to prob mapping");
        }
      }// if is_detect

    }// parts

    cout << "root part: " << rootpart_idx << endl;

    vector<vector<FloatGrid2> > log_part_detections(nParts, vector<FloatGrid2>());

    mergeRotations(part_app, part_conf,
                   part_score_grid_rotation,
                   log_part_detections);

    if (bDebugOutput) {
      QString qsFilename = "./debug/merge_rot_max_flip" + QString::number(flip) + ".mat";
      cout << "saving " << qsFilename.toStdString() << endl;
      MATFile *f = matlab_io::mat_open(qsFilename, "wz");
      matlab_io::mat_save_multi_array_vec2(f, "merge_max", log_part_detections);
      matlab_io::mat_close(f);
    }

    QString qsDebugDir = "./debug/flip" + QString::number(flip) + "-maptype" + QString::number(scoreProbMapType) + "-pos";

    FloatGrid3 root_part_posterior;
    computeRootPosterior(part_app, log_part_detections, root_part_posterior, rootpart_idx, joints, flip, qsDebugDir, bIsSparse);
    
    int max_hypothesis_number = 1000;
    findLocalMax(part_app.m_exp_param, root_part_posterior, hypothesis_list, max_hypothesis_number);

    for (int hypidx = 0; hypidx < hypothesis_list.hyp_size(); ++hypidx) {
      hypothesis_list.mutable_hyp(hypidx)->set_flip(flip);
    }

    for (int i = 0; i < min(10, hypothesis_list.hyp_size()); ++i) {
      cout << "hypothesis " << i << 
        ", x: " << hypothesis_list.hyp(i).x() << 
        ", y: " << hypothesis_list.hyp(i).y() << 
        ", scaleidx: " << index_from_scale(part_app.m_exp_param, hypothesis_list.hyp(i).scale()) << 
        ", score: " << hypothesis_list.hyp(i).score() << endl;
    }



    /**
       LEGACY: needed to run CVPR'10 code 

       save root marginal if needed
     */
    if (part_app.m_exp_param.save_root_marginal()) {
      QString qsRootMarginalsDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/root_part_posterior").c_str();
      
      if (!filesys::check_dir(qsRootMarginalsDir))
        filesys::create_dir(qsRootMarginalsDir);

      QString qsFilename = qsRootMarginalsDir + "/root_part_posterior" + 
        "_imgidx" + QString::number(imgidx) +
        "_o" + QString::number((int)flip) + ".mat";
      
      cout << "saving " << qsFilename.toStdString() << endl;
      matlab_io::mat_save_multi_array(qsFilename, "root_part_posterior", root_part_posterior);
    }//if save root

  }

}// namespace 
