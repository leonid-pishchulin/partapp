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

#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_transform.hpp>
#include <libMultiArray/multi_array_filter.hpp>

#include <libBoostMath/homogeneous_coord.h>

#include <libPartApp/partapp_aux.hpp>
#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartDetect/partdetect.h>
#include <libPartDetect/partdef.h>

#include <libDiscPS/disc_ps.h>
#include <libPartEval/parteval.h>

#include <libDiscPS/factors.h>

#include "objectdetect_aux.hpp"
#include "objectdetect.h"
#include "Timer.h"

using boost_math::double_vector;
using boost_math::double_matrix;
using boost_math::double_zero_matrix;

using boost::multi_array_types::index_range;

using namespace boost::lambda;
using namespace std;

namespace object_detect {
  
  double get_runtime()
  {
    clock_t start;
    start = clock();
    return(((double)start*100.0/(double)CLOCKS_PER_SEC));
  }
  
  void get_incoming_joints(const vector<Joint> &joints, int curidx, vector<int> &all_children, vector<int> &all_joints)
  {
    all_children.clear();
    all_joints.clear();

    for (uint jidx = 0; jidx < joints.size(); ++jidx) {
      if (joints[jidx].parent_idx == curidx) {
        all_children.push_back(joints[jidx].child_idx);
        all_joints.push_back(jidx);
      }
    }
    
  }

  void getMaxStates(const PartApp &part_app, vector<vector<FloatGrid3> > &log_part_detections, std::vector<std::vector<PartHyp> > &best_part_hyp){
    
    cout << "\ngetMaxStates()\n" << endl;
    int nParts = part_app.m_part_conf.part_size();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    int scaleidx = 0;
    double scale = scale_from_index(part_app.m_exp_param, scaleidx);
    
    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];
    
    if (best_part_hyp.size() != 0)
      best_part_hyp.clear();
    best_part_hyp.resize(nParts, std::vector<PartHyp>());
    
    for (int pidx = 0; pidx < nParts; ++pidx) {
      // float bestval = -1e6;
      float bestval = -DBL_MAX;
      int bestidx = -1;
      
      for (int idx = 0; idx < (int)log_part_detections[pidx][scaleidx].num_elements(); ++idx) {
	if (log_part_detections[pidx][scaleidx].data()[idx] > bestval) {
	  bestval = log_part_detections[pidx][scaleidx].data()[idx];
	  bestidx = idx;
	}
      }

      assert(bestidx >= 0);
      int best_rotidx, best_x, best_y;
      disc_ps::index_from_flat3(nRotations, img_height, img_width, bestidx, best_rotidx, best_y, best_x);
      best_part_hyp[pidx].push_back(PartHyp(part_app.m_exp_param, scaleidx, best_rotidx, best_x, best_y, bestval));

      /** find and add local maxima */
      vector<PartHyp> local_max;
      findLocalMax(part_app.m_exp_param, log_part_detections[pidx][scaleidx], local_max, part_app.m_exp_param.roi_save_num_samples());
      best_part_hyp[pidx].insert(best_part_hyp[pidx].end(), local_max.begin(), local_max.end());
    }
  }
  
  /**
     compute part marginals (i.e. pass the messages from root downstream)

     assumptions:
     log_part_posterior contains product of messages from downstream and part appearance
     log_from_root product of messages received by root form all children excluding one particular child

     on completion:
     log_part_posterior contains marginal distribution of each part
     log_from_root contains upstream messages received by each part
   */

  void computePartMarginals(const PartApp &part_app, 
			    vector<Joint> joints,
			    int rootpart_idx, 
			    int scaleidx,
			    bool flip,
			    int imgidx, 
			    vector<vector<FloatGrid3> > &log_part_detections, 
			    vector<FloatGrid3> &log_part_posterior,
			    vector<FloatGrid3> &log_from_root, 
			    std::vector<std::vector<PartHyp> > &best_part_hyp, 
			    bool bSaveMarginals)
 {
   QString qsPartMarginalsDir = (part_app.m_exp_param.log_dir() + "/" + 
				 part_app.m_exp_param.log_subdir() + "/part_marginals").c_str();


   if (!filesys::check_dir(qsPartMarginalsDir))
     filesys::create_dir(qsPartMarginalsDir);

   int nParts = part_app.m_part_conf.part_size();

   int nRotations = part_app.m_exp_param.num_rotation_steps();
    double scale = scale_from_index(part_app.m_exp_param, scaleidx);

    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];

    assert(joints.size() > 0);
    assert((int)log_part_posterior.size() == nParts);
    assert((int)log_from_root.size() == nParts);

    vector<bool> vComputedRootMessages(nParts, false);
    vector<int> compute_marginals_stack;

    /** finish computation of messages from root to children */

    cout << "finalizing computation of upstream messages" << endl;
    vector<int> all_children;
    vector<int> incoming_joints;
    get_incoming_joints(joints, rootpart_idx, all_children, incoming_joints);
            
    for (int i = 0; i < (int)all_children.size(); ++i) {
      int child_idx = all_children[i];
      int jidx = incoming_joints[i];
      cout << "\tadding appearance component to message to " << child_idx << endl;
      multi_array_op::addGrid2(log_from_root[child_idx], log_part_detections[rootpart_idx][scaleidx]);
              
      FloatGrid3 tmpgrid2(boost::extents[nRotations][img_height][img_width]);
      
      /** backward pass */
      computeRotJointMarginal(part_app.m_exp_param, 
                              log_from_root[child_idx], tmpgrid2, 
                              joints[jidx].offset_p, joints[jidx].offset_c, 
                              joints[jidx].C,
                              -joints[jidx].rot_mean, joints[jidx].rot_sigma, 
                              scale, false /* not sparse */ );

      log_from_root[child_idx] = tmpgrid2;
          
      vComputedRootMessages[child_idx] = true;
      compute_marginals_stack.push_back(child_idx);

    }// joints from root

    cout << "done." << endl;

    /* send downstream messages */

    while(!compute_marginals_stack.empty()) {
      int curidx = compute_marginals_stack.back();

      compute_marginals_stack.pop_back();
      cout << "computing marginal for part " << curidx << endl;

      assert(vComputedRootMessages[curidx] == true);

      /** this is part posterior (product of downstream messages, upstream message and part appearance ) */
      multi_array_op::addGrid2(log_part_posterior[curidx], log_from_root[curidx]);
                
      vector<int> all_children;
      vector<int> all_joints;
      get_incoming_joints(joints, curidx, all_children, all_joints);

      /** support only simple trees for now, otherwise we must make sure that children do not receive any messages which 
          they have send to the parent 
      */
      assert(all_children.size() <= 1); 

      /** compute messages to child nodes */
      if (all_children.size() == 1) {

        //for (int i = 0; i < (int)all_children.size(); ++i) {
        int i = 0;

        int child_idx = all_children[i];
        int jidx = all_joints[i];
          
        FloatGrid3 tmpgrid = log_part_detections[curidx][scaleidx];
        multi_array_op::addGrid2(tmpgrid, log_from_root[curidx]);

        computeRotJointMarginal(part_app.m_exp_param, 
                                tmpgrid, log_from_root[child_idx], 
                                joints[jidx].offset_p, joints[jidx].offset_c, 
                                joints[jidx].C, 
                                -joints[jidx].rot_mean, joints[jidx].rot_sigma, 
                                scale, false /* not sparse */); 

        vComputedRootMessages[child_idx] = true;
        compute_marginals_stack.push_back(child_idx);
      }// children


    }// children stack 

    /** save part posterior */
    if (bSaveMarginals) {
      //assert(imgidx > 0);

      for (int pidx = 0; pidx < nParts; ++pidx) {
	QString qsFilename = qsPartMarginalsDir + "/log_part_posterior_final" + 
	  "_imgidx" + QString::number(imgidx) +
	  "_scaleidx" + QString::number(scaleidx) +
	  "_o" + QString::number((int)flip) + 
	  "_pidx" + QString::number(pidx) + ".mat";

	cout << "saving " << qsFilename.toStdString() << endl;

	matlab_io::mat_save_multi_array(qsFilename, "log_prob_grid", log_part_posterior[pidx]);
      }
    }

    /** find maxima */
    // assert(best_part_hyp.size() == 0);
    if (best_part_hyp.size() != 0)
      best_part_hyp.clear();
    best_part_hyp.resize(nParts, std::vector<PartHyp>());
        
    for (int pidx = 0; pidx < nParts; ++pidx) {
      // float bestval = -1e6;
      float bestval = -DBL_MAX;
      int bestidx = -1;
      
      for (int idx = 0; idx < (int)log_part_posterior[pidx].num_elements(); ++idx) {
	if (log_part_posterior[pidx].data()[idx] > bestval) {
	  bestval = log_part_posterior[pidx].data()[idx];
	  bestidx = idx;
	}
      }

      assert(bestidx >= 0);

      int best_rotidx, best_x, best_y;
      disc_ps::index_from_flat3(nRotations, img_height, img_width, bestidx, best_rotidx, best_y, best_x);
      best_part_hyp[pidx].push_back(PartHyp(part_app.m_exp_param, scaleidx, best_rotidx, best_x, best_y, bestval));

      /** find and add local maxima */
      vector<PartHyp> local_max;
      findLocalMax(part_app.m_exp_param, log_part_posterior[pidx], local_max, part_app.m_exp_param.roi_save_num_samples());

      best_part_hyp[pidx].insert(best_part_hyp[pidx].end(), local_max.begin(), local_max.end());

    } // parts 
 }

  /**
     rot_mean, rot_sigma - mean and sigma of gaussian which describes range of possible rotations (both are in radians!!!)

   */
  void computeRotJointMarginal(const ExpParam &exp_param, 
                               FloatGrid3 &log_prob_child, FloatGrid3 &log_prob_parent, 
                               const double_vector &_offset_c_10, const double_vector &_offset_p_01, 
                               const double_matrix &C, 
                               double rot_mean, double rot_sigma,
                               double scale, bool bIsSparse)
  {
    cout << "computeRotJointMarginal, scale " << scale << endl;

    FloatGrid3::element in_m, in_M;
    multi_array_op::getMinMax(log_prob_child, in_m, in_M);

    // MA: 2010-12-21 commented this out since now we want to support parts that model foreshortening (those have zero relative rotation)
    //assert(rot_sigma > 0);

    assert(_offset_p_01.size() == 2 && _offset_c_10.size() == 2);
    double_vector offset_c_10 = hc::get_vector(_offset_c_10(0), _offset_c_10(1));
    double_vector offset_p_01 = hc::get_vector(_offset_p_01(0), _offset_p_01(1));
   
    assert(log_prob_child.shape()[0] == log_prob_parent.shape()[0]);
    assert(log_prob_child.shape()[1] == log_prob_parent.shape()[1]);
    assert(log_prob_child.shape()[2] == log_prob_parent.shape()[2]);

    int nRotations = log_prob_child.shape()[0];
    int grid_height = log_prob_child.shape()[1];
    int grid_width = log_prob_child.shape()[2];

    double rot_step_size = (exp_param.max_part_rotation() - exp_param.min_part_rotation())/exp_param.num_rotation_steps();
    rot_step_size *= M_PI / 180.0;
    assert(rot_step_size > 0);
    double rot_sigma_idx = rot_sigma / rot_step_size;

    /** rot_mean is a mean rotation from parent to child in world cs                                    */
    /** since we are propagating information from child to parent the mean rotation is multiplied by -1 */
    int rot_mean_idx = boost_math::round(-rot_mean / rot_step_size);

    cout << "-rot_mean: " << -rot_mean << ", rot_mean_idx: " << rot_mean_idx << endl;
    cout << "rot_sigma: " << rot_sigma << ", rot_sigma_idx: " << rot_sigma_idx << endl;
    
    /* transform from child part center to position of joint between child and parent parts */
    /*                                                                                      */
    /* position of joint depends on part scale and rotation                                 */
    /* joint 1-0: joint of child to parent                                                  */
    /* joint 0-1: joint of parent to child                                                  */
    /* ideally both have the same position in world coordinates                             */
    /* here we allow some offset which is penalized by euclidian distance                   */ 

    FloatGrid3 log_joint_01(boost::extents[nRotations][grid_height][grid_width]);
    FloatGrid3 log_joint_10(boost::extents[nRotations][grid_height][grid_width]);

    multi_array_op::setGrid(log_joint_10, LOG_ZERO);
    multi_array_op::setGrid(log_joint_01, LOG_ZERO);
    
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      int rotidx_out = rotidx + rot_mean_idx;
      if (rotidx_out >= 0 && rotidx_out < nRotations) {
        float alpha = rot_from_index(exp_param, rotidx)*M_PI/180.0;
        double_matrix Tgc = prod(hc::get_rotation_matrix(alpha), hc::get_scaling_matrix(scale));
        double_vector offset_g_10 = prod(Tgc, offset_c_10);
 
	double_matrix Tjoint = hc::get_translation_matrix(offset_g_10(0), offset_g_10(1));
	
        FloatGrid3View2 view_out = log_joint_10[boost::indices[rotidx_out][index_range()][index_range()]];

        multi_array_op::transform_grid_fixed_size(log_prob_child[rotidx], view_out, 
                                                  Tjoint, LOG_ZERO, TM_NEAREST);
      }
    }// rotations
      
    /** divide by the max value to prevent underflow (and multiply with it later)*/
    multi_array_op::addGrid1(log_joint_10, -in_M);

    /* obtain probabilities */
    multi_array_op::computeExpGrid(log_joint_10); 

    /** compute values as if we would be rescaling every image to the training scale -> all normalization constants are 
        the same -> ignore them */

    bool bNormalizeRotFilter = false; 
    bool bNormalizeSpatialFilter = false;

    FloatGrid3 rot_filter_result(boost::extents[nRotations][grid_height][grid_width]);

    /* rotation dimension */
    
    if (rot_sigma > 0) {
      double_vector f_rot;
      boost_math::get_gaussian_filter(f_rot, rot_sigma_idx, bNormalizeRotFilter);

      int firstidx = 0;
      int f_rot_size = (int)f_rot.size();
      
      // if kernel size > rotation grid size => clip kernel tails
      if (f_rot_size >= nRotations && true){
	//cout << f_rot_size << endl;
	int crot = (int)f_rot_size/2;
	f_rot_size = (nRotations % 2 == 1) ? nRotations - 2 : nRotations - 1;
	firstidx = crot - (int)f_rot_size/2;
	/*
	cout << crot << endl;
	cout << firstidx << endl;
	cout << lastidx << endl;
	cout << f_rot_size << endl;
	getchar();
	*/
      }

      int lastidx = firstidx + f_rot_size;
      
      /** MA: convert to plain array representation (needed for blas convolution) */
      float f_rot_float[1000];
      for (int idx = firstidx; idx < lastidx; ++idx)
	f_rot_float[idx - firstidx] = (float)f_rot(idx);
      
      for (int x = 0; x < grid_width; ++x)
	for (int y = 0; y < grid_height; ++y) {
	  
	  FloatGrid3View1 view_in = log_joint_10[boost::indices[index_range()][y][x]];
	  FloatGrid3View1 view_out = rot_filter_result[boost::indices[index_range()][y][x]];
	  multi_array_op::grid_filter_1d_blas_wraparound(view_in, view_out, f_rot_float, f_rot_size);
 	  //multi_array_op::grid_filter_1d_blas(view_in, view_out, f_rot_float, f_rot.size());
	  //multi_array_op::grid_filter_1d(view_in, view_out, f_rot);
	}
      
      //cout << "------------ " << endl;
    }
    else if (rot_sigma == 0){
      rot_filter_result = log_joint_10;
    }

    /* x/y dimensions */
    double_matrix scaleC = square(scale)*C;
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      FloatGrid3View2 view_in = rot_filter_result[boost::indices[rotidx][index_range()][index_range()]];
      FloatGrid3View2 view_out = log_joint_01[boost::indices[rotidx][index_range()][index_range()]];

      multi_array_op::gaussFilter2d(view_in, view_out, scaleC, bNormalizeSpatialFilter, bIsSparse);
    }

    /* go back to log prob */
    multi_array_op::computeLogGrid(log_joint_01);

    /** multiply back by the maximum value */
    multi_array_op::addGrid1(log_joint_01, in_M);

    /* transform joint 0-1 obtain position of parent */
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      float alpha = rot_from_index(exp_param, rotidx)*M_PI/180.0;
      double_matrix Tgo = prod(hc::get_rotation_matrix(alpha), hc::get_scaling_matrix(scale));
      double_vector offset_g_01 = prod(Tgo, offset_p_01);

      double_matrix Tobject = hc::get_translation_matrix(-offset_g_01(0), -offset_g_01(1));
      FloatGrid3View2 view_out = log_prob_parent[boost::indices[rotidx][index_range()][index_range()]];

      multi_array_op::transform_grid_fixed_size(log_joint_01[rotidx], view_out, Tobject, 
                                                LOG_ZERO, TM_NEAREST);
    }

    FloatGrid3::element out_m, out_M;
    multi_array_op::getMinMax(log_prob_parent, out_m, out_M);
    cout << "\t log_prob_child, minval: " << in_m << ", maxval: " << in_M << endl;
    cout << "\t log_prob_parent, minval: " << out_m << ", maxval: " << out_M << endl;

    cout << "done." << endl;
  }
  
  void computeRootPosteriorRot(const PartApp part_app, 
                               std::vector<std::vector<FloatGrid3> > &log_part_detections, 
                               FloatGrid3 &root_part_posterior, int rootpart_idx, std::vector<Joint> joints, bool flip, 
                               bool bIsSparse, int imgidx)
  {
    std::vector<std::vector<PartHyp> > best_part_hyp;
    bool bSaveMarginals = false;
    
    computeRootPosteriorRot(part_app, log_part_detections, root_part_posterior, rootpart_idx, joints, flip, 
			    bIsSparse, imgidx, best_part_hyp, bSaveMarginals);
  }
  
  void computeRootPosteriorRot(const PartApp part_app, 
                               vector<vector<FloatGrid3> > &log_part_detections, 
                               FloatGrid3 &root_part_posterior, int rootpart_idx, vector<Joint> joints, bool flip, 
                               bool bIsSparse, int imgidx, std::vector<std::vector<PartHyp> >& best_part_hyp, bool bSaveMarginals)
  {
    cout << "computeRootPosteriorRot" << endl;

    bool bDebugMessageOutput = false;

    /**
       number of samples to be drawn from posterior
     */
    int num_samples = part_app.m_exp_param.num_pose_samples();

    cout << "\t num_samples: " << num_samples << endl;

    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();

    assert((int)log_part_detections.size() == nParts);
    assert((int)log_part_detections[0].size() == nScales);
    assert((int)log_part_detections[0][0].shape()[0] == nRotations);

    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];
    assert(img_width > 0 && img_height > 0);

    /** grid where we store results */
    FloatGrid4 root_part_posterior_full(boost::extents[nScales][nRotations][img_height][img_width]);
    multi_array_op::computeLogGrid(root_part_posterior_full);

    /** compute object probability for each scale */
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {

      /* 
         enforce upright orientation if necessary 
      */

      for (int pidx = 0; pidx < nParts; ++pidx) {
        if (part_app.m_part_conf.part(pidx).is_upright()) {

          for (int ridx = 0; ridx < nRotations; ++ridx) {
            double cur_rot = rot_from_index(part_app.m_exp_param, ridx);

            if (!(abs(cur_rot) < 15.0)) {
              for (int iy = 0; iy < img_height; ++iy)
                for (int ix = 0; ix < img_width; ++ix) {
                  log_part_detections[pidx][scaleidx][ridx][iy][ix] = LOG_ZERO;                  
                }
            }
          }
        }
      } // parts
      
      /* 
         strip border detections 
      */
      if (part_app.m_exp_param.strip_border_detections() > 0) {
        assert(part_app.m_exp_param.strip_border_detections() < 0.5);

        int strip_width = (int)(part_app.m_exp_param.strip_border_detections() * img_width);

        cout << "strip all detections of root part (" << 
          rootpart_idx << ") inside border region, strip_width: " << 
          strip_width << endl;

        for (int scaleidx = 0; scaleidx < nScales; ++scaleidx)
          for (int ridx = 0; ridx < nRotations; ++ridx) {
            for (int iy = 0; iy < img_height; ++iy) {

              for (int ix = 0; ix < strip_width; ++ix)
                log_part_detections[rootpart_idx][scaleidx][ridx][iy][ix] = LOG_ZERO;

              for (int ix = (img_width - strip_width); ix < img_width; ++ix) 
                log_part_detections[rootpart_idx][scaleidx][ridx][iy][ix] = LOG_ZERO;

            }// iy
              
          }// rotations

      }// strip border
      
      double scale = scale_from_index(part_app.m_exp_param, scaleidx);

      if (bDebugMessageOutput)
        cout << "processing scale " << scaleidx << "(" << scale << ")" << endl;

      /* after upstream pass: message from downstream combined with appearance model */
      /* after downstream pass: part posterior                                       */
      vector<FloatGrid3> log_part_posterior(nParts, 
                                            FloatGrid3(boost::extents[nRotations][img_height][img_width]));

      vector<bool> vComputedPosteriors(nParts, false);
      vector<int> compute_stack;
      compute_stack.push_back(rootpart_idx);

      /** these are only needed if we also want to compute part marginals */
      vector<FloatGrid3> log_from_root;
      
      log_from_root.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));
            
      /** these are only needed if we sample from posterior */
      vector<FloatGrid3> log_part_posterior_sample; 
      vector<FloatGrid3> log_from_root_sample;
      if (num_samples > 0) {
        log_part_posterior_sample.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));
        log_from_root_sample.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));
      }

      /**  pass messages from children to the root  */
      
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
          get_incoming_joints(joints, curidx, all_children, incoming_joints);
	  
          for (int i = 0; i < (int)all_children.size(); ++i) {

            int child_idx = all_children[i];
            int jidx = incoming_joints[i];

	    cout << "child_idx: " << child_idx << endl;
	    
            if (bDebugMessageOutput)
              cout << "\tcomputing component from " << child_idx << endl;
            
            assert(vComputedPosteriors[child_idx]);

            FloatGrid3 log_part_posterior_from_child(boost::extents[nRotations][img_height][img_width]);
	    
	    computeRotJointMarginal(part_app.m_exp_param,
				    log_part_posterior[child_idx], log_part_posterior_from_child, 
				    joints[jidx].offset_c, joints[jidx].offset_p, 
				    joints[jidx].C, 
				    joints[jidx].rot_mean, joints[jidx].rot_sigma,
				    scale, bIsSparse);
	    
	    multi_array_op::addGrid2(log_part_posterior[curidx], log_part_posterior_from_child);
	    
            /* start computing messages to direct children of root node     */
            /* do not include message that was received from child          */
            if (curidx == rootpart_idx) {
              for (int i2 = 0; i2 < (int)all_children.size(); ++i2) {
                if (i2 != i) {
                  int child_idx2 = all_children[i2];
                  cout << "\tadding posterior from " << child_idx << " to root message to " << child_idx2 << endl;
                  multi_array_op::addGrid2(log_from_root[child_idx2], log_part_posterior_from_child);
                }
              }
            }
          }// children
	  
          if (part_app.m_part_conf.part(curidx).is_detect()) {
            multi_array_op::addGrid2(log_part_posterior[curidx], log_part_detections[curidx][scaleidx]);
	  }

          vComputedPosteriors[curidx] = true;
        }// if can compute
      }// stack

      /* store log_from_root before adding appearance of root part                                           */
      /* store precomputed products of appearance and messages from downstream in log_part_posterior_sample */
      if (num_samples > 0) {
        log_from_root_sample = log_from_root;
        log_part_posterior_sample = log_part_posterior;
      }
            
      computePartMarginals(part_app, joints, 
			   rootpart_idx, scaleidx, flip, imgidx,
			   log_part_detections, 
			   log_part_posterior, 
			   log_from_root, 
			   best_part_hyp, 
			   bSaveMarginals);
      
      /** store root posterior (used for object detection) */
      root_part_posterior_full[scaleidx] = log_part_posterior[rootpart_idx];
      
      /** sample from posterior */
      
      if (num_samples > 0) {
        assert(false && "sampling from posterior not supported yet");
      }
      
    }// scales
    
    
    /** 
        since hypothesis with multiple orienations are not supported at the moment
        here we marginalize over valid orientations 

        in the end hypothesis is assumed to be upright 
     */

    vector<int> valid_root_rotations;

    if (part_app.m_part_conf.part(rootpart_idx).is_upright()) {
      int keep_idx1 = index_from_rot(part_app.m_exp_param, -1e-6);
      int keep_idx2 = index_from_rot(part_app.m_exp_param, 1e-6);
      cout << "keep only upright orientations of the root part: " << keep_idx1 << ", " << keep_idx2 << endl;
      
      valid_root_rotations.push_back(keep_idx1);
      if (keep_idx2 != keep_idx1)
        valid_root_rotations.push_back(keep_idx2);
    }
    else {
      cout << "keep all orientations of the root part" << endl;
      
      for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
        valid_root_rotations.push_back(rotidx);
      }
      cout << endl;
    }
    
    root_part_posterior.resize(boost::extents[nScales][img_height][img_width]);
    root_part_posterior = root_part_posterior_full[boost::indices[index_range()][valid_root_rotations[0]][index_range()][index_range()]];

    multi_array_op::computeExpGrid(root_part_posterior);
      
    for (uint idx = 1; idx < valid_root_rotations.size(); ++idx) {
      FloatGrid3 tmp_grid = root_part_posterior_full[boost::indices[index_range()][valid_root_rotations[idx]][index_range()][index_range()]];
      
      multi_array_op::computeExpGrid(tmp_grid);
      multi_array_op::addGrid2(root_part_posterior, tmp_grid);
    }
    
    multi_array_op::computeLogGrid(root_part_posterior);
  }
  
  void findObjectImageRotJoints(const PartApp &part_app, int imgidx, bool flip, 
				HypothesisList &hypothesis_list, 
				int scoreProbMapType, 
				QString qsPartMarginalsDir,
				QString qsScoreGridDir, QString qsImgName, 
				std::vector<std::vector<FloatGrid3> > &log_part_detections,
				bool bLoadScoreGrid)
  {
    cout << "findObjectImageRotJoints" << endl;
    
    const PartConfig &part_conf = part_app.m_part_conf;
    bool bIsSparse = true; 
        
    /** load joints */
    int nJoints = part_conf.joint_size();
    int nParts = part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();

    //cout << "nParts: " << nParts << endl;
    //cout << "nJoints: " << nJoints << endl;

    // yes load the image again :) just to find its width
    int img_width, img_height;
    {
      kma::ImageContent *kmaimg = kma::load_convert_gray_image(qsImgName.toStdString().c_str());
      
      assert(kmaimg != 0);
      img_width = kmaimg->x();
      img_height = kmaimg->y();
      delete kmaimg;
    }

    vector<Joint> joints(nJoints);
    loadJoints(part_app, joints, flip, imgidx);

    for (int jidx = 0; jidx < nJoints; ++jidx) {
      assert(joints[jidx].type == Joint::ROT_GAUSSIAN);

      cout << "covariance matrix of the joint: " << endl;
      boost_math::print_matrix(joints[jidx].C);
      // test end
    }

    /** load classifier scores */
    cout << "nParts: " << nParts << endl;
    cout << "nScales: " << nScales << endl;
    cout << "nRotations: " << nRotations << endl;
    cout << "img_height: " << img_height << endl;
    cout << "img_width: " << img_width << endl;
        
                                                                                                    
    assert(scoreProbMapType == SPMT_NONE);

    int rootpart_idx = -1;
    for (int pidx = 0; pidx < nParts; ++pidx)
      if (part_conf.part(pidx).is_detect() && part_conf.part(pidx).is_root()) {
	  assert(rootpart_idx == -1);
	  rootpart_idx = pidx;
      }
    
    if (bLoadScoreGrid){
      
      double mb_count = (4.0 * nParts * nScales * nRotations * img_height * img_width) / (square(1024));
      cout << "allocating: " << mb_count << " MB " << endl;
      log_part_detections.resize(nParts, vector<FloatGrid3>(nScales, FloatGrid3(boost::extents[nRotations][img_height][img_width])));
      cout << "done." << endl;
      
      cout << "loading scores ... " << endl;
      for (int pidx = 0; pidx < nParts; ++pidx) {
	if (part_conf.part(pidx).is_detect()) {
	  
	  bool bInterpolate = false;
	  if (part_app.m_exp_param.has_interpolate())
	    bInterpolate = part_app.m_exp_param.interpolate();
	  
	  cout << "bInterpolate: " << bInterpolate << endl;
	  
	  vector<vector<FloatGrid2> > cur_part_detections;
	  
	  part_app.loadScoreGrid(cur_part_detections, imgidx, pidx, flip, bInterpolate, qsScoreGridDir, qsImgName);
	  
	  assert((int)cur_part_detections.size() == nScales);
	  assert((int)cur_part_detections[0].size() == nRotations);
	  
	  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx){ 
	    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
	      log_part_detections[pidx][scaleidx][rotidx] = cur_part_detections[scaleidx][rotidx];
	    }
	  }
	}// if is_detect
      }
      cout << "done." << endl;
    }
    else{
      assert(log_part_detections.size() == nParts);
      assert(log_part_detections[0].size() == nScales);
      assert(log_part_detections[0][0].shape()[0] == nRotations);
      assert(log_part_detections[0][0].shape()[1] == img_height);
      assert(log_part_detections[0][0].shape()[2] == img_width);
    }
    
    assert(rootpart_idx >= 0 && "root part not found");
    cout << endl << "rootpart_idx: " << rootpart_idx << endl << endl;

    for (int pidx = 0; pidx < nParts; ++pidx) 
      if (part_app.m_part_conf.part(pidx).is_detect()) {
        for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {

	  if (scoreProbMapType == SPMT_NONE)
	    object_detect::clip_scores_fill(log_part_detections[pidx][scaleidx]);
	  else 
	    assert(false);
	  
	  multi_array_op::computeLogGrid(log_part_detections[pidx][scaleidx]);
	}
      }

    vector<vector<FloatGrid3> > log_part_detections_orig = log_part_detections;
    
    std::vector<FloatGrid2> dpmTorsoGrid;    
    /**************************** load DPM torso score  ************************/    
    if (part_app.m_exp_param.use_dpm_torso()){
      QString qsUnaryDPMDir = part_app.m_exp_param.test_dpm_torso_dir().c_str();
      loadDPMScoreGrid(qsUnaryDPMDir, imgidx, dpmTorsoGrid, false);
    }
    /**************************** load DPM torso score  ************************/    
        
    boost_math::double_matrix rot_params;
    /**************************** predict rotation params ************************/    
    if (part_app.m_exp_param.has_pred_unary_rot() && 
	part_app.m_exp_param.pred_unary_rot()){
      predictFactors(part_app, 1, imgidx);
      getRotParams(part_app, imgidx, rot_params, true);
      //getRotParamsMulti(part_app, imgidx, rot_params, true);
    }
    /**************************** predict rotation params ************************/    
    
    boost_math::double_matrix pos_params;
    /**************************** predict position params ************************/    
    if (part_app.m_exp_param.pred_unary_pos()){
      predictFactors(part_app, 2, imgidx);
      getPosParams(part_app, imgidx, pos_params, rootpart_idx, true);
      //getPosParamsMulti(part_app, imgidx, pos_params, rootpart_idx, true);
    }
    /**************************** predict position params ************************/    
    
    boost_math::double_matrix pos_prior_params;
    /**************************** load torso position prior ************************/    
    if (part_app.m_exp_param.use_torso_pos_prior())
      getTorsoPosPriorParams(part_app, img_height, img_width, pos_prior_params);
    /**************************** load torso position prior ************************/    
    
    /**************************** add DPM torso score ************************/    
    if (!dpmTorsoGrid.empty()){
      int scaleidx = 0;
      for(int ridx = 0; ridx < dpmTorsoGrid.size();++ridx)
	multi_array_op::computeLogGrid(dpmTorsoGrid[ridx]);
      addDPMScore(part_app, log_part_detections, dpmTorsoGrid, rootpart_idx, 
		  part_app.m_exp_param.dpm_torso_weight());
    }
    /**************************** add DPM torso score ************************/    
    
    /**************************** add DPM head score ************************/    
    if (part_app.m_exp_param.use_dpm_head()){
      int headpart_idx = 5;
      if (part_app.m_part_conf.part_size() == 22)
	headpart_idx = 11;
      else if (part_app.m_part_conf.part_size() == 12)
	headpart_idx = 1;
      
      QString qsUnaryDPMDir = (part_app.m_exp_param.test_dpm_unary_dir() + "/head").c_str();
      int nRotationsDPM = 1;
      addLoadDPMScore(part_app, log_part_detections, imgidx, part_app.m_exp_param.dpm_head_weight(), nRotationsDPM,
		      qsUnaryDPMDir, false, headpart_idx);
    }
    /**************************** add DPM head score ************************/    
    
    /**************************** add DPM unary score ************************/    
    if (part_app.m_exp_param.has_use_dpm_unary() && 
	part_app.m_exp_param.use_dpm_unary()){
      assert(part_app.m_exp_param.has_test_dpm_unary_dir());
      int nRotationsDPM = (part_app.m_exp_param.do_dpm_rot() ? nRotations : 1); 
      addLoadDPMScore(part_app, log_part_detections, imgidx, part_app.m_exp_param.dpm_unary_weight(), nRotationsDPM,
		      QString(part_app.m_exp_param.test_dpm_unary_dir().c_str()), true);
    }
    /**************************** add DPM unary score ************************/    
    
    vector<vector<FloatGrid3> > 
      log_rot_scores(nParts, vector<FloatGrid3>(nScales, 
						FloatGrid3(boost::extents[nRotations][img_height][img_width])));
    
    /**************************** add rotation score ************************/
    if (part_app.m_exp_param.has_pred_unary_rot() && 
	part_app.m_exp_param.pred_unary_rot()){
      getRotScoreGrid(part_app, log_rot_scores, rot_params);
      //getRotScoreGridMulti(part_app, log_rot_scores, rot_params);
      addExtraUnary(part_app, log_part_detections, log_rot_scores, part_app.m_exp_param.pred_unary_rot_weight());
    }
    /**************************** add rotation score ************************/
    
    vector<vector<FloatGrid3> > 
      log_pos_scores(nParts, vector<FloatGrid3>(nScales, 
						FloatGrid3(boost::extents[nRotations][img_height][img_width])));
    boost_math::double_vector rootpos_det;
    /**************************** add position score ************************/    
    if (part_app.m_exp_param.has_pred_unary_pos() && 
	part_app.m_exp_param.pred_unary_pos()){
      getRootPosDet(part_app, imgidx, rootpart_idx, rootpos_det, true);
      getPosScoreGrid(part_app, log_pos_scores, imgidx, pos_params, rootpart_idx, rootpos_det);
      //getPosScoreGridMulti(part_app, log_pos_scores, imgidx, pos_params, rootpart_idx, rootpos_det);
      addExtraUnary(part_app, log_part_detections, log_pos_scores, part_app.m_exp_param.pred_unary_pos_weight());
    }
    /**************************** add position score ************************/    
    
    /**************************** add torso position prior score ************************/
    if (part_app.m_exp_param.has_use_torso_pos_prior() && 
	part_app.m_exp_param.use_torso_pos_prior()){
      setTorsoPosPrior(part_app, log_part_detections, pos_prior_params, part_app.m_rootpart_idx);
    }
    /**************************** add torso position prior score ************************/

    /****************** integrate rough position information (cvpr'14) *****************/
    int posX = part_app.m_test_annolist[imgidx][0].getObjPosX();
    int posY = part_app.m_test_annolist[imgidx][0].getObjPosY();
    if (posX > 0 && posY > 0){
      cout << "\nsetTorsoPosition()" << endl;
      cout << "posX: " << posX << "; posY: " << posY << endl;
      int delta = 40;
      cout << "delta: " << delta << endl;
      cout << endl;
      int scaleidx = 0;
      int firstix = max(0, posX - delta);
      int firstiy = max(0, posY - delta);
      int lastix = min(posX + delta, img_width);
      int lastiy = min(posY + delta, img_height);
      for (int ridx = 0; ridx < nRotations; ++ridx) 
	for (int iy = 0; iy < img_height; ++iy) 
	  for (int ix = 0; ix < img_width; ++ix)
	    if (iy < firstiy || iy > lastiy || 
		ix < firstix || ix > lastix)
	      log_part_detections[part_app.m_rootpart_idx][scaleidx][ridx][iy][ix] = LOG_ZERO;                  
    }
    /**************************** integrate rough position information ************************/

    vector<vector<PartHyp> >  best_part_hyp;
    
    FloatGrid3 root_part_posterior;
    bool bSaveMarginals = part_app.m_exp_param.save_part_marginals();

    /**************************** Find part configuration ************************/    
    //double rt1 = get_runtime();
    
    if (!part_app.m_exp_param.use_pairwise())
      getMaxStates(part_app, log_part_detections, best_part_hyp);
    else
      computeRootPosteriorRot(part_app, log_part_detections, root_part_posterior, rootpart_idx, joints, flip, 
			      bIsSparse, imgidx, best_part_hyp, bSaveMarginals);
    
    //cout << "*******************************" << endl;
    //double rt = MAX(get_runtime()-rt1,0)/100;
    //printf("Inference runtime in cpu-seconds: %.2f\n", rt);fflush(stdout);
    //cout << "*******************************" << endl;
    /**************************** Find part configuration ************************/    
    
    log_part_detections = log_part_detections_orig;
    assert((int)best_part_hyp.size() == nParts);
    
    /**************************** Save part configuration ************************/    
    /* MA: vector of best configurations, needed  to call getStructSample */
    vector<PartHyp> _best_part_hyp(nParts);
    for (int pidx = 0; pidx < nParts; ++pidx) {
      assert(best_part_hyp[pidx].size() > 0);
      _best_part_hyp[pidx] = best_part_hyp[pidx][0];
    }
          
    FloatGrid1 v = best_part_hyp[0][0].toVect();
    FloatGrid2 best_conf(boost::extents[nParts][v.shape()[0]]);
    
    for (int pidx = 0; pidx < nParts; ++pidx)
      best_conf[boost::indices[pidx][index_range()]] = best_part_hyp[pidx][0].toVect();
    
    QString qsOutFilename = qsPartMarginalsDir + "/pose_est_imgidx" + padZeros(QString::number(imgidx), 4) + ".mat";
    matlab_io::mat_save_multi_array(qsOutFilename, "best_conf", best_conf);
    /**************************** Save part configuration ************************/    
            
    int max_hypothesis_number = 1000;
    findLocalMax(part_app.m_exp_param, root_part_posterior, hypothesis_list, max_hypothesis_number);

    for (int hypidx = 0; hypidx < hypothesis_list.hyp_size(); ++hypidx) {
      hypothesis_list.mutable_hyp(hypidx)->set_flip(flip);

      /** these are hypothesis for root part, convert them to hypothesis for object bounding box */
      int bbox_x = (int)(hypothesis_list.hyp(hypidx).x() + part_app.m_window_param.bbox_offset_x());
      int bbox_y = (int)(hypothesis_list.hyp(hypidx).y() + part_app.m_window_param.bbox_offset_y());
      hypothesis_list.mutable_hyp(hypidx)->set_x(bbox_x);
      hypothesis_list.mutable_hyp(hypidx)->set_y(bbox_y);
    }

    for (int i = 0; i < min(10, hypothesis_list.hyp_size()); ++i) {
      cout << "hypothesis " << i << 
        ", x: " << hypothesis_list.hyp(i).x() << 
        ", y: " << hypothesis_list.hyp(i).y() << 
        ", scaleidx: " << index_from_scale(part_app.m_exp_param, hypothesis_list.hyp(i).scale()) << 
        ", score: " << hypothesis_list.hyp(i).score() << endl;
    }
  }



}// namespace
