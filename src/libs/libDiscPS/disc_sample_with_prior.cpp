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

/**
   sampling from posterior conditioned on the position of the bounding box (needed if we want samples to be distributed accross multiple people)

   note: currently we perform the inference independently for each bounding box, 
   however the sampling can be done more efficiently by storing the upstream messages 

   NEEDS REFACTORING: the code here is essentially the same as in objectdetect_findrot.cpp
 */

#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <libMultiArray/multi_array_def.h>
#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_filter.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libMatlabIO/matlab_io.hpp>

#include <libPartApp/partapp.h>
#include <libPartApp/partapp_aux.hpp>

#include <libKMA2/kmaimagecontent.h>

#include <libPictStruct/objectdetect.h>
#include <libPictStruct/objectdetect_aux.hpp>

#include "unique_vect.h"

#include "disc_sample.hpp"

using namespace std;
using namespace boost::lambda;
using namespace boost_math;
using boost::multi_array_types::index_range;

namespace disc_ps {

  using object_detect::Joint;

  /** 
      copy/paste from computeRootPosteriorRot objectdetect_findrot.cpp
   */
  void partSampleWithPriorHelper(const PartApp part_app, 
				 vector<vector<FloatGrid3> > log_part_detections, 
				 int scaleidx, int rootpart_idx, vector<Joint> joints, bool bIsSparse, vector<boost_math::int_matrix> &idxmat_allparts)
  {

    cout << "partSampleWithPriorHelper" << endl;

    bool bDebugMessageOutput = false;

    /** 
        always compute part marginals (since we want to sample from them)
     */
    bool bComputePartMarginals = true;

    cout << "\t bComputePartMarginals: " << bComputePartMarginals << endl;

    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();

    assert((int)log_part_detections.size() == nParts);
    assert((int)log_part_detections[0].size() == nScales);
    assert((int)log_part_detections[0][0].shape()[0] == nRotations);

    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];
    assert(img_width > 0 && img_height > 0);

//     vector<FloatGrid4> log_part_posterior_allscales(nParts, 
// 						    FloatGrid4(boost::extents[nScales][nRotations][img_height][img_width]));

    /** compute object probability for each scale */
    //for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {

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
      if (bComputePartMarginals) 
        log_from_root.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));


      /** these are only needed if we sample from posterior */
      //vector<FloatGrid3> log_part_posterior_sample; 
      //vector<FloatGrid3> log_from_root_sample;

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

            if (bDebugMessageOutput)
              cout << "\tcomputing component from " << child_idx << endl;
            
            assert(vComputedPosteriors[child_idx]);

            FloatGrid3 log_part_posterior_from_child(boost::extents[nRotations][img_height][img_width]);

	    object_detect::computeRotJointMarginal(part_app.m_exp_param, 
                                    log_part_posterior[child_idx], log_part_posterior_from_child, 
                                    joints[jidx].offset_c, joints[jidx].offset_p, 
                                    joints[jidx].C, 
                                    joints[jidx].rot_mean, joints[jidx].rot_sigma,
                                    scale, bIsSparse);

            multi_array_op::addGrid2(log_part_posterior[curidx], log_part_posterior_from_child);

            /* start computing messages to direct children of root node     */
            /* do not include message that was received from child          */
            if (curidx == rootpart_idx && bComputePartMarginals) {
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

      /** pass messages from root to children */
      if (bComputePartMarginals) { 

	//         computePartMarginals(part_app, joints, 
	//                             rootpart_idx, scaleidx, flip, imgidx,
	//                             log_part_detections, 
	//                             log_part_posterior, 
	//                             log_from_root);

	/** 
	    here goes copy/paste of computePartMarginals
	*/

	//     QString qsPartMarginalsDir = (part_app.m_exp_param.log_dir() + "/" + 
	//                                   part_app.m_exp_param.log_subdir() + "/part_marginals").c_str();

	//     if (!filesys::check_dir(qsPartMarginalsDir))
	//       filesys::create_dir(qsPartMarginalsDir);

	int nParts = part_app.m_part_conf.part_size();
// 	int nRotations = part_app.m_exp_param.num_rotation_steps();
// 	double scale = scale_from_index(part_app.m_exp_param, scaleidx);

// 	int img_width = log_part_detections[0][0].shape()[2];
// 	int img_height = log_part_detections[0][0].shape()[1];

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
	  cout << "\tadding apprearance component to message to " << child_idx << endl;
	  multi_array_op::addGrid2(log_from_root[child_idx], log_part_detections[rootpart_idx][scaleidx]);
              
	  FloatGrid3 tmpgrid2(boost::extents[nRotations][img_height][img_width]);

	  /** backward pass */
	  object_detect::computeRotJointMarginal(part_app.m_exp_param, 
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

	    object_detect::computeRotJointMarginal(part_app.m_exp_param, 
				    tmpgrid, log_from_root[child_idx], 
				    joints[jidx].offset_p, joints[jidx].offset_c, 
				    joints[jidx].C, 
				    -joints[jidx].rot_mean, joints[jidx].rot_sigma, 
				    scale, false /* not sparse */); 

	    vComputedRootMessages[child_idx] = true;
	    compute_marginals_stack.push_back(child_idx);
	  }// children


	}// children stack 


	//     /** save part posterior */
	//     for (int pidx = 0; pidx < nParts; ++pidx) {
	//       QString qsFilename = qsPartMarginalsDir + "/log_part_posterior_final" + 
	//         "_imgidx" + QString::number(imgidx) +
	//         "_scaleidx" + QString::number(scaleidx) +
	//         "_o" + QString::number((int)flip) + 
	//         "_pidx" + QString::number(pidx) + ".mat";

	//       cout << "saving " << qsFilename.toStdString() << endl;

	//       matlab_io::mat_save_multi_array(qsFilename, "log_prob_grid", log_part_posterior[pidx]);
	//     }



      } // if bComputePartMarginals

//       for (uint pidx = 0; pidx < (uint)nParts; ++pidx) {
// 	log_part_posterior_allscales[pidx][scaleidx] = log_part_posterior[pidx];
//       }

//     }// scales

    /** sample from posterior */

    for (uint pidx = 0; pidx < (uint)nParts; ++pidx) {
      int rnd_seed = (int)1e5;

      //multi_array_op::computeExpGrid(log_part_posterior_allscales[pidx]);
      //disc_ps::discrete_sample(log_part_posterior_allscales[pidx], part_app.m_exp_param.dai_num_samples(), idxmat_allparts[pidx], rnd_seed);

      multi_array_op::computeExpGrid(log_part_posterior[pidx]);
      disc_ps::discrete_sample(log_part_posterior[pidx], part_app.m_exp_param.dai_num_samples(), idxmat_allparts[pidx], rnd_seed);
    }


  }
                               
  void partSampleWithPrior(const PartApp &part_app, int firstidx, int lastidx)
  {
    /** copy/paste from partSample in disc_ps.cpp */

    cout << "disc_ps::partSampleWithPrior " << endl << endl;

    /** make sure we don't overwrite samples from other experiment */
    assert(!part_app.m_bExternalSamplesDir && 
	   "using external samples dir, new samples should be generated using original experiment file");

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {

      QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
	padZeros(QString::number(imgidx), 4);
      cout << "saving samples to " << qsPrecomputedSamplesDir.toStdString() << endl;

      if (!filesys::check_dir(qsPrecomputedSamplesDir)) 
	assert(filesys::create_dir(qsPrecomputedSamplesDir));

      bool flip = false;
      const PartConfig &part_conf = part_app.m_part_conf;

      vector<boost_math::int_matrix> idxmat_allparts(part_conf.part_size());

      /** 
	  copy/paste from findObjectImageRotJoints in objectdetect_findrot.cpp
      */

      // yes load the image again :) just to find its width
      int img_width, img_height;
      {
	kma::ImageContent *kmaimg = kma::load_convert_gray_image(part_app.m_test_annolist[imgidx].imageName().c_str());
	assert(kmaimg != 0);
	img_width = kmaimg->x();
	img_height = kmaimg->y();
	delete kmaimg;
      }

      /** load classifier scores */
      int nJoints = part_conf.joint_size();

      vector<Joint> joints(nJoints);
      loadJoints(part_app, joints, flip);

      for (int jidx = 0; jidx < nJoints; ++jidx) {
	assert(joints[jidx].type == Joint::ROT_GAUSSIAN);

	cout << "covariance matrix of the joint: " << endl;
	boost_math::print_matrix(joints[jidx].C);
	// test end

      }

      int nParts = part_conf.part_size();
      int nScales = part_app.m_exp_param.num_scale_steps();
      int nRotations = part_app.m_exp_param.num_rotation_steps();

      cout << "nJoints: " << nJoints << endl;
      cout << "nParts: " << nParts << endl;
      cout << "nScales: " << nScales << endl;
      cout << "nRotations: " << nRotations << endl;
      cout << "img_height: " << img_height << endl;
      cout << "img_width: " << img_width << endl;

      double mb_count = (4.0 * nParts * nScales * nRotations * img_height * img_width) / (square(1024));
      cout << "allocating: " << mb_count << " MB " << endl;
      vector<vector<FloatGrid3> > log_part_detections(nParts, 
						      vector<FloatGrid3>(nScales, 
									 FloatGrid3(boost::extents[nRotations][img_height][img_width])));
      cout << "done." << endl;
                                                                                                
      cout << "loading scores ... " << endl;

      int rootpart_idx = -1;
      for (int pidx = 0; pidx < nParts; ++pidx) {
	if (part_conf.part(pidx).is_detect()) {

	  bool bInterpolate = false;

	  vector<vector<FloatGrid2> > cur_part_detections;

	  part_app.loadScoreGrid(cur_part_detections, imgidx, pidx, flip, bInterpolate, "test_scoregrid", 
				 part_app.m_test_annolist[imgidx].imageName().c_str());

	  assert((int)cur_part_detections.size() == nScales);
	  assert((int)cur_part_detections[0].size() == nRotations);

	  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
	      log_part_detections[pidx][scaleidx][rotidx] = cur_part_detections[scaleidx][rotidx];
	    }

	}// if is_detect

	if (part_conf.part(pidx).is_root()) {
	  assert(rootpart_idx == -1);
	  rootpart_idx = pidx;
	}
      }

      cout << "done." << endl;

      assert(rootpart_idx >= 0 && "root part not found");
      cout << endl << "rootpart_idx: " << rootpart_idx << endl << endl;;

      for (int pidx = 0; pidx < nParts; ++pidx) 
	if (part_app.m_part_conf.part(pidx).is_detect()) {
	  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {
	    object_detect::clip_scores_fill(log_part_detections[pidx][scaleidx]);
	  
	    multi_array_op::computeLogGrid(log_part_detections[pidx][scaleidx]);
	  }
	}

      /**
	 end of copy/paste from findObjectImageRotJoints
      */

      bool bIsSparse = true; 


      int num_detections = part_app.m_test_annolist[imgidx].m_vRects.size();
      
      vector<vector<boost_math::int_matrix> > det_idxmat_allparts(num_detections);

      vector<int> count_samples(nParts, 0);
      vector<int> det_scaleidx(num_detections, -1);

      vector<FloatGrid3> original_root_detections = log_part_detections[rootpart_idx];


      for (int didx = 0; didx < num_detections; ++didx) {
	int x1 = part_app.m_test_annolist[imgidx].m_vRects[didx].m_x1;
	int x2 = part_app.m_test_annolist[imgidx].m_vRects[didx].m_x2;

	int y1 = part_app.m_test_annolist[imgidx].m_vRects[didx].m_y1;
	int y2 = part_app.m_test_annolist[imgidx].m_vRects[didx].m_y2;

	int cur_det_height = 0.95*(y2 - y1); // the scale of objects trained on LSP is a little smaller than 200 px.
	double cur_det_scale = cur_det_height / part_app.m_window_param.train_object_height();

	for (int pidx = 0; pidx < nParts; ++pidx) {
	  det_idxmat_allparts[didx].push_back(boost_math::int_matrix());
	}
	

	/**
	   begin: add the bounding box prior
	*/

	int_vector root_pos(2);

	if (!part_app.m_exp_param.dai_bbox_prior_annopoints()) {
	
	  double_vector bbox_pos(2);
	  bbox_pos(0) = 0.5*(part_app.m_test_annolist[imgidx].m_vRects[didx].m_x1 + part_app.m_test_annolist[imgidx].m_vRects[didx].m_x2);
	  bbox_pos(1) = 0.5*(part_app.m_test_annolist[imgidx].m_vRects[didx].m_y1 + part_app.m_test_annolist[imgidx].m_vRects[didx].m_y2);

	  double_vector root_offset(2);

	  /** 
	      HARDCODED correction factor which is needed because of mismatch in poses between LSP and dancing data
	  */

	  double correction_factor = 1.5;

	  if (part_app.m_part_conf.part_size() == 10) {
	    correction_factor = 1.0;
	  }

	  cout << "********************************************************************************" << endl;
	  cout << "correction_factor: " << correction_factor << endl;
	  cout << "********************************************************************************" << endl;
	
	  root_offset(0) = part_app.m_window_param.bbox_offset_x();
	  root_offset(1) = correction_factor*part_app.m_window_param.bbox_offset_y();

	  root_pos = bbox_pos - cur_det_scale*root_offset;
	}
	else {
	  // root part position comes from annopoints 
	  
	  assert(part_app.m_test_annolist[imgidx].m_vRects[didx].m_vAnnoPoints.size() == 2 && "assume hand-annotated root part -> 2 annopoints");

	  root_pos(0) = 0.5*(part_app.m_test_annolist[imgidx].m_vRects[didx].m_vAnnoPoints[0].x + 
			     part_app.m_test_annolist[imgidx].m_vRects[didx].m_vAnnoPoints[1].x);
	  
	  root_pos(1) = 0.5*(part_app.m_test_annolist[imgidx].m_vRects[didx].m_vAnnoPoints[0].y + 
			     part_app.m_test_annolist[imgidx].m_vRects[didx].m_vAnnoPoints[1].y);

	}

	cout << "didx: " << didx << ", root position: (" << root_pos(0) << ", " << root_pos(1) << ")" << endl;

	FloatGrid2 _root_prior(boost::extents[img_height][img_width]);

	for (uint ix = 0; ix < (uint)img_width; ++ix) 
	  for (uint iy = 0; iy < (uint)img_height; ++iy) 
	    _root_prior[iy][ix] = 0;

	_root_prior[root_pos(1)][root_pos(0)] = 1.0;

	double root_prior_sigma = 5.0;
	double_matrix C(2, 2);
	C(0, 0) = square(root_prior_sigma);
	C(0, 1) = 0;
	C(1, 0) = 0;
	C(1, 1) = C(0,0);
	
	FloatGrid2 root_prior(boost::extents[img_height][img_width]);

	bool bNormalize = false;
	multi_array_op::gaussFilterDiag2d(_root_prior, root_prior, C, bNormalize);
	
	multi_array_op::computeLogGrid(root_prior);

	int count_inside = 0;

	for (int sidx = 0; sidx < nScales; ++sidx)
	  for (int ridx = 0; ridx < nRotations; ++ridx)
	    for (int yidx = 0; yidx < img_height; ++yidx)
	      for (int xidx = 0; xidx < img_width; ++xidx) {
	      
		if(!(xidx >= x1 && xidx <= x2 && yidx >= y1 && yidx < y2)) {
		  log_part_detections[rootpart_idx][sidx][ridx][yidx][xidx] = LOG_ZERO;
		}
		else {
		  log_part_detections[rootpart_idx][sidx][ridx][yidx][xidx] += root_prior[yidx][xidx];

		  if (exp(log_part_detections[rootpart_idx][sidx][ridx][yidx][xidx]) > 0.0001) 
		    ++count_inside;

		}

	      }
      
	

	cout << "count_inside: " << count_inside << endl;
	assert(count_inside > 0);

	/**
	   end: add the bounding box prior
	*/



	cout << "start helper" << endl;

	det_scaleidx[didx] = index_from_scale_clip(part_app.m_exp_param, cur_det_scale);
	cout << "\t didx: " << didx << ", scale: " << det_scaleidx[didx] << endl;

	partSampleWithPriorHelper(part_app, log_part_detections, det_scaleidx[didx], rootpart_idx, joints, bIsSparse, det_idxmat_allparts[didx]);
	cout << "end helper" << endl;

	assert(det_idxmat_allparts[didx].size() == (uint)nParts);

	for (int pidx = 0; pidx < nParts; ++pidx) {
	  count_samples[pidx] += det_idxmat_allparts[didx][pidx].size1();
	}

	/** restore root detections for the next bounding box */
	log_part_detections[rootpart_idx] = original_root_detections;

      }// detections

      /** 
	  accumulate samples from all detections 
      */

      for (int pidx = 0; pidx < nParts; ++pidx) {
	/** now we also want to store identity of the the subject */
	idxmat_allparts[pidx].resize(count_samples[pidx], 5);

	//idxmat_allparts[pidx].resize(count_samples[pidx], 4);
	//idxmat_allparts[pidx].resize(count_samples[pidx], det_idxmat_allparts[0][pidx].size2());
      }

      for (int pidx = 0; pidx < nParts; ++pidx) {
	uint idx = 0;

	/**
	   didx - subject/detection idx
	   sidx - sample idx

	   pidx - part idx
	 */

	cout << "concatenating samples begin" << endl;
	for (uint didx = 0; didx < det_idxmat_allparts.size(); ++didx) 
	  for (uint sidx = 0; sidx < det_idxmat_allparts[didx][pidx].size1(); ++sidx) {

	    assert(idxmat_allparts[pidx].size1() > idx);

	    /** we should add an extra dimension for the scale because  we now sample from FloatGrid3 */

	    idxmat_allparts[pidx](idx, 0) = det_scaleidx[didx];
	    idxmat_allparts[pidx](idx, 4) = didx;

	    for (uint idx2 = 0; idx2 < det_idxmat_allparts[didx][pidx].size2(); ++idx2) {
	      //cout << pidx << " " << count_samples[pidx] << " " << idxmat_allparts[pidx].size1()<< " " << det_idxmat_allparts[didx][pidx].size1() << " " << idx << endl;
	      assert(idx2+1 < idxmat_allparts[pidx].size2());

	      idxmat_allparts[pidx](idx, idx2+1) = det_idxmat_allparts[didx][pidx](sidx, idx2);
	    }

	    //cout << "idxmat_allparts[pidx](idx, 4): " << idxmat_allparts[pidx](idx, 4) << endl;

	    idx++;
	  }
	cout << "concatenating samples end" << endl;

	assert(idx == (uint)count_samples[pidx]);

	/** BEGIN filter repeating samples */
	vector<double_vector> V;

	assert(idxmat_allparts[pidx].size1() == (uint)count_samples[pidx]);

	cout << "pidx " << pidx << ", count_samples: " << count_samples[pidx] << endl;

	for (int samp_idx = 0; samp_idx < (int)idxmat_allparts[pidx].size1(); ++samp_idx) {
	  //assert(idxmat_allparts[pidx].size2() == 4);
	  assert(idxmat_allparts[pidx].size2() == 5);

	  //double_vector v(4);
	  double_vector v(5);
 	  v(0) = idxmat_allparts[pidx](samp_idx, 0);
 	  v(1) = idxmat_allparts[pidx](samp_idx, 1);
 	  v(2) = idxmat_allparts[pidx](samp_idx, 2);
 	  v(3) = idxmat_allparts[pidx](samp_idx, 3);
 	  v(4) = idxmat_allparts[pidx](samp_idx, 4);
	    
	  V.push_back(v);
	}

	vector<double_vector> V2;
	cout << "get_unique_elements - begin" << endl;
	get_unique_elements(V, V2);
	cout << "get_unique_elements - end" << endl;

	//idxmat_allparts[pidx].resize(V2.size(), idxmat_allparts[pidx].size2());

	vector<int> vect_scale_idx(V2.size());
	vector<int> vect_rot_idx(V2.size());
	vector<int> vect_iy(V2.size());
	vector<int> vect_ix(V2.size());
	vector<int> vect_didx(V2.size());

	for (uint samp_idx = 0; samp_idx < V2.size(); ++samp_idx) {
	    vect_scale_idx[samp_idx] = (int)V2[samp_idx](0);
	    vect_rot_idx[samp_idx] = (int)V2[samp_idx](1);
	    vect_iy[samp_idx] = (int)V2[samp_idx](2);
	    vect_ix[samp_idx] = (int)V2[samp_idx](3);
	    vect_didx[samp_idx] = (int)V2[samp_idx](4);

	    //cout << "vect_didx[samp_idx]: " << vect_didx[samp_idx] << endl;
	}

	/** END filter repeating samples */

	

	cout << "saving ..." << endl;

	/** save samples */
	QString qsOutputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";
	MATFile *f = matlab_io::mat_open(qsOutputFilename, "wz");
	assert(f != 0);

        matlab_io::mat_save_stdcpp_vector(f, "vect_scale_idx", vect_scale_idx);
        matlab_io::mat_save_stdcpp_vector(f, "vect_rot_idx", vect_rot_idx);
        matlab_io::mat_save_stdcpp_vector(f, "vect_iy", vect_iy); 
        matlab_io::mat_save_stdcpp_vector(f, "vect_ix", vect_ix);
        matlab_io::mat_save_stdcpp_vector(f, "vect_didx", vect_didx);

	matlab_io::mat_close(f);

	cout << "saved samples to " <<qsOutputFilename.toStdString() << endl;

      } // parts
    } // images 

  }


}// namespace 
