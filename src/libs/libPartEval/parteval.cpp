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

#include <algorithm>

#include <libFilesystemAux/filesystem_aux.h>
#include <libMultiArray/multi_array_def.h>

#include <libPartApp/partapp.h>
#include <libPartApp/partapp_aux.hpp>

#include <libMatlabIO/matlab_io.hpp>
#include <libMisc/misc.hpp>

#include <libBoostMath/homogeneous_coord.h>

#include <libMultiArray/multi_array_op.hpp>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libPictStruct/objectdetect.h>

#include <libPartDetect/partdetect.h>
#include <libPartDetect/partdef.h>

#include "parteval.h"


using namespace std;

using boost_math::double_vector;
using boost::multi_array_types::index_range;

void get_bbox_endpoints(PartBBox &bbox, double_vector &endpoint_top, double_vector &endpoint_bottom, double &seg_len)
{
  
  if (bbox.use_endpoints) {
  
    endpoint_top = double_vector(2);
    endpoint_bottom = double_vector(2);

    endpoint_top(0) = bbox.x1;
    endpoint_top(1) = bbox.y1;

    endpoint_bottom(0) = bbox.x2;
    endpoint_bottom(1) = bbox.y2;
    
    seg_len = ublas::norm_2(endpoint_top - endpoint_bottom);
  }
  else {
    endpoint_top = bbox.part_pos + bbox.min_proj_y*bbox.part_y_axis;
    endpoint_bottom = bbox.part_pos + bbox.max_proj_y*bbox.part_y_axis;
    seg_len = bbox.max_proj_y - bbox.min_proj_y;
  }
    
}

void get_bbox_endpoints_xaxis(PartBBox &bbox, double_vector &endpoint_left, double_vector &endpoint_right, double &seg_len)
{
  seg_len = bbox.max_proj_x - bbox.min_proj_x;

  endpoint_left = bbox.part_pos + bbox.min_proj_x*bbox.part_x_axis;
  endpoint_right = bbox.part_pos + bbox.max_proj_x*bbox.part_x_axis;
}

bool is_gt_match(PartBBox &gt_bbox, PartBBox &detect_bbox, bool match_x_axis, float factor) 
{
  //cout << "is_gt_match, match_x_axis: " << match_x_axis << endl;
  assert(factor >= 0 && factor <= 1.0);
    
  if (!match_x_axis) {
    double gt_seg_len;
    double_vector gt_endpoint_top;
    double_vector gt_endpoint_bottom;
    get_bbox_endpoints(gt_bbox, gt_endpoint_top, gt_endpoint_bottom, gt_seg_len);
  
    double detect_seg_len;
    double_vector detect_endpoint_top;
    double_vector detect_endpoint_bottom;
    get_bbox_endpoints(detect_bbox, detect_endpoint_top, detect_endpoint_bottom, detect_seg_len);
    
    bool match_top = (ublas::norm_2(gt_endpoint_top - detect_endpoint_top) < factor*gt_seg_len);
    bool match_bottom = (ublas::norm_2(gt_endpoint_bottom - detect_endpoint_bottom) < factor*gt_seg_len);
    
    bool bVerbose = false;
    
    if (bVerbose){
      cout << "thresh:   " << factor*gt_seg_len << endl;
      cout << "gt top; bottom: (" << gt_endpoint_top(0) << ", " << gt_endpoint_top(1) << "); (" << gt_endpoint_bottom(0) << ", " << gt_endpoint_bottom(1) << ")" << endl;
      cout << "dt top; bottom: (" << detect_endpoint_top(0) << ", " << detect_endpoint_top(1) << "); (" << detect_endpoint_bottom(0) << ", " << detect_endpoint_bottom(1) << ")" << endl;
      cout << "top dist: " << ublas::norm_2(gt_endpoint_top - detect_endpoint_top) << endl;
      cout << "bot dist: " << ublas::norm_2(gt_endpoint_bottom - detect_endpoint_bottom) << endl;
      cout << "gt_s_len: " << gt_seg_len << endl;
      cout << "dt_s_len: " << detect_seg_len << endl;
    }
    return match_top && match_bottom;
  }
  else {
    double gt_seg_len;
    double_vector gt_endpoint_left;
    double_vector gt_endpoint_right;
    get_bbox_endpoints_xaxis(gt_bbox, gt_endpoint_left, gt_endpoint_right, gt_seg_len);

    double detect_seg_len;
    double_vector detect_endpoint_left;
    double_vector detect_endpoint_right;
    get_bbox_endpoints_xaxis(detect_bbox, detect_endpoint_left, detect_endpoint_right, detect_seg_len);
        
    bool match_left = (ublas::norm_2(gt_endpoint_left - detect_endpoint_left) < factor*gt_seg_len);
    bool match_right = (ublas::norm_2(gt_endpoint_right - detect_endpoint_right) < factor*gt_seg_len);
    return match_left && match_right;
  }
}

bool is_gt_bbox_match(PartBBox &gt_bbox, PartBBox &detect_bbox, float thresh) 
{
  //cout << "is_gt_match, match_x_axis: " << match_x_axis << endl;
  assert(thresh >= 0 && thresh <= 1.0);
  
  float x1 = detect_bbox.part_pos(0) + detect_bbox.min_proj_x;
  float x2 = detect_bbox.part_pos(0) + detect_bbox.max_proj_x;
  float x1_detect = min(x1,x2);
  float x2_detect = max(x1,x2);
  float y1 = detect_bbox.part_pos(1) + detect_bbox.min_proj_y;
  float y2 = detect_bbox.part_pos(1) + detect_bbox.max_proj_y;
  float y1_detect = min(y1,y2);
  float y2_detect = max(y1,y2);

  x1 = gt_bbox.part_pos(0) + gt_bbox.min_proj_x;
  x2 = gt_bbox.part_pos(0) + gt_bbox.max_proj_x;
  float x1_gt = min(x1,x2);
  float x2_gt = max(x1,x2);
  y1 = gt_bbox.part_pos(1) + gt_bbox.min_proj_y;
  y2 = gt_bbox.part_pos(1) + gt_bbox.max_proj_y;
  float y1_gt = min(y1,y2);
  float y2_gt = max(y1,y2);
  
  /*
  cout << x1_detect << endl;
  cout << x2_detect << endl;
  cout << y1_detect << endl;
  cout << y2_detect << endl;
  */
  
  float x1_intersect = max(x1_detect, x1_gt);
  float y1_intersect = max(y1_detect, y1_gt);
  float x2_intersect = min(x2_detect, x2_gt);
  float y2_intersect = min(y2_detect, y2_gt);

  float intersect_area = 0;
  if (x1_intersect > x2_intersect || y1_intersect > y2_intersect)
    intersect_area = 0;
  else
    intersect_area = (x2_intersect - x1_intersect)*(y2_intersect - y1_intersect);
  
  float area_detect = (x2_detect - x1_detect) * (y2_detect - y1_detect); 
  float area_gt     = (x2_gt     - x1_gt    ) * (y2_gt     - y1_gt    );
  float union_area =  area_detect + area_gt - intersect_area;
  
  float ratio = intersect_area / union_area;
  /*
  cout << "area_detect " << area_detect << endl;
  cout << "area_gt " << area_gt << endl;
  cout << "intersect_area " << intersect_area << endl;
  cout << "union_area " << union_area << endl;
  cout << "ratio " << ratio << endl;
    //getchar();
  */
  
  return (ratio >= thresh);
}

void loadPartHyp(const PartApp &part_app, QString qsHypDir, int imgidx,
		 vector<object_detect::PartHyp> &part_hyp){
    
  assert(qsHypDir.size() > 0);
  QString qsFilename = qsHypDir + "/pose_est_imgidx" + padZeros(QString::number(imgidx), 4) + ".mat";
  FloatGrid2 best_conf = matlab_io::mat_load_multi_array<FloatGrid2>(qsFilename, "best_conf");
  int nParts = part_app.m_part_conf.part_size();
  
  assert(nParts == best_conf.shape()[0]);
  
  for (int pidx = 0; pidx < nParts; ++pidx){
    object_detect::PartHyp hyp; 
    hyp.fromVect(best_conf[boost::indices[pidx][index_range()]]);
    part_hyp.push_back(hyp);
  }
}


/** 
    merge bboxes as required when two model parts represent the same body parts by encoding positions of joints 
*/

PartBBox bbox_merge(const PartBBox &bbox1, const PartBBox &bbox2, const PartBBox &bbox3, const PartBBox &bbox4)
{
  PartBBox r;
  r = bbox1;
  r.part_pos = 0.25*(bbox1.part_pos + bbox2.part_pos + bbox3.part_pos + bbox4.part_pos);

  r.min_proj_y = std::min(inner_prod(r.part_y_axis, bbox3.part_pos - r.part_pos), inner_prod(r.part_y_axis, bbox4.part_pos - r.part_pos));
  r.max_proj_y = std::max(inner_prod(r.part_y_axis, bbox1.part_pos - r.part_pos), inner_prod(r.part_y_axis, bbox2.part_pos - r.part_pos));

  r.min_proj_x = std::min(inner_prod(r.part_x_axis, bbox1.part_pos - r.part_pos), inner_prod(r.part_x_axis, bbox4.part_pos - r.part_pos));
  r.max_proj_x = std::max(inner_prod(r.part_x_axis, bbox2.part_pos - r.part_pos), inner_prod(r.part_x_axis, bbox3.part_pos - r.part_pos));

  return r;
}

PartBBox bbox_merge(const PartBBox &bbox1, const PartBBox &bbox2)
{
  PartBBox r;
  r = bbox1;
  r.part_pos = 0.5*(bbox1.part_pos + bbox2.part_pos);
 
  r.min_proj_y = inner_prod(r.part_y_axis, bbox2.part_pos - r.part_pos);
  r.max_proj_y = inner_prod(r.part_y_axis, bbox1.part_pos - r.part_pos);

  return r;
}

PartBBox bbox_merge(const PartBBox &bbox1, const PartBBox &bbox2, const ExpParam &exp_param)
{
  PartBBox r;
  //r = bbox1;
  r.part_pos = 0.5*(bbox1.part_pos + bbox2.part_pos);
  
  r.min_proj_x = bbox1.min_proj_x;
  r.max_proj_x = bbox1.max_proj_x;
  
  double min_proj_x = min(bbox1.part_pos(0), bbox2.part_pos(0)) - r.part_pos(0);
  double max_proj_x = max(bbox1.part_pos(0), bbox2.part_pos(0)) - r.part_pos(0);

  double min_proj_y = min(bbox1.part_pos(1), bbox2.part_pos(1)) - r.part_pos(1);
  double max_proj_y = max(bbox1.part_pos(1), bbox2.part_pos(1)) - r.part_pos(1);
  
  if (max_proj_x - min_proj_x > max_proj_y - min_proj_y){
    r.min_proj_y = min_proj_x;
    r.max_proj_y = max_proj_x;
  }
  else{
    r.min_proj_y = min_proj_y;
    r.max_proj_y = max_proj_y;
  }
  /* define part rotation by vector between joint positions */
  boost_math::double_vector y_axis = bbox1.part_pos - bbox2.part_pos;
  
  double part_rot = atan2(y_axis(1),y_axis(0));
  
  double min_diff = DBL_MAX;
  double best_disc_rot = DBL_MAX;
  uint best_rotidx = UINT_MAX;
  
  /* assign the closest rotation from the discrete set */
  for (uint rotidx = 0; rotidx < exp_param.num_rotation_steps(); ++rotidx){
    double disc_rot = rot_from_index(exp_param, rotidx) / 180.0 * M_PI;
    double diff = abs(part_rot - disc_rot);
    if (min_diff > diff){
      min_diff = diff;
      best_rotidx = rotidx;
      best_disc_rot = disc_rot;
    }		    
  }
  
  r.part_y_axis(0) = cos(best_disc_rot);
  r.part_y_axis(1) = sin(best_disc_rot);

  r.part_x_axis(0) = -r.part_y_axis(1);
  r.part_x_axis(1) = r.part_y_axis(0);
  
  return r;
}

float get_shrink_factor_x(double ext_x_pos, int pidx, int rootidx = 4){
  
  float default_offset = 15;
  if (pidx == rootidx)
    default_offset = 30;
  else if (pidx == rootidx+1)
    default_offset = 20;
  
  float shrink_factor_x = 0.7*default_offset/ext_x_pos;
  return shrink_factor_x;
}



/**
   this is a helper function for eval_segments and vis_segments
 
   extract best estimate for each "canonical" body part (this might require conversion from model parts)

   didx is the id of the subject which is available when we use the bounding box prior

   note: there is some HARDCODED shrinking of bounding boxes (this should be looked at when evaluating new part configurations)
         this is also the reason why we need part_conf_eval parameter	 
 */

void vis_eval_helper(const PartApp &part_app, int imgidx, int scaleidx, const int eval_type, const PartConfig &part_conf_eval, 
		     vector<PartBBox> &eval_bbox, QString qsHypDirName = "", 
		     int didx = -1, bool bMerge = true)
{
  assert(eval_bbox.size() == 0);
  
  /**
     part results may be coming from: 
     - unaries
     - pictorial structures inference (dense version)
     - pictorial structures inference (discrete version)       
  */
  vector<object_detect::PartHyp> best_part_hyp;
	  
  if (eval_type == EVAL_TYPE_UNARIES) {
        
    int nParts = part_app.m_window_param.part_size();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    QString qsScoreGridDir = part_app.m_exp_param.scoregrid_dir().c_str();
    QString qsImgName = part_app.m_test_annolist[imgidx].imageName().c_str();
      
    for (int pidx = 0; pidx < nParts; ++pidx) {

      int max_scaleidx = scaleidx;
      int max_ridx= -1;
      int max_ix = -1, max_iy = -1;

      /** 
	  MA: why ???

	  this only works for ramanan's and buffy single scale setting 
      */
      assert(part_app.m_part_conf.part_size() == 10 ||
	     part_app.m_part_conf.part_size() == 6  ||
	     part_app.m_part_conf.part_size() == 21);

      vector<vector<FloatGrid2> > part_detections;
      
      bool flip = false;
      bool bInterpolate = false;
      part_app.loadScoreGrid(part_detections, imgidx, pidx, flip, bInterpolate, qsScoreGridDir, qsImgName);
      assert(part_detections.size() == 1);
      
      assert((int)part_detections[0].size() == nRotations);
      int img_height = part_detections[0][0].shape()[0];
      int img_width = part_detections[0][0].shape()[1];
      cout << "img_width: " << img_width << ", img_height: " << img_height << endl;
      
      max_ridx = 0;
      max_iy = 0;
      max_ix = 0;
      float max_val = part_detections[0][0][0][0];
      
      for (int ridx = 0; ridx < nRotations; ++ridx)
	for (int ix = 0; ix < img_width; ++ix)
	  for (int iy = 0; iy < img_height; ++iy) {
	    if (part_detections[0][ridx][iy][ix] > max_val) {
	      max_ridx = ridx;
	      max_ix = ix;
	      max_iy = iy;
	      max_val = part_detections[0][ridx][iy][ix];
	    }
	  }          
      
      best_part_hyp.push_back(object_detect::PartHyp(part_app.m_exp_param, max_scaleidx, max_ridx, max_ix, max_iy, max_val));
    }// parts
  }
  else if (eval_type == EVAL_TYPE_PS) {
    
    loadPartHyp(part_app, qsHypDirName, imgidx, best_part_hyp);
    
  }
  else if (eval_type == EVAL_TYPE_DISC_PS) {

    for (int pidx = 0; pidx < part_app.m_part_conf.part_size(); ++pidx) {

      int max_scaleidx = scaleidx;
      int max_ridx= -1;
      int max_ix = -1, max_iy = -1;

      /**
	 in current implementation inference is done jointly for all scales 
	 (presumably to accomodate interactions between multiple subjects at different scales)

	 i.e. at this point we can not extract maxima of posteriors independently computed for each scale
      */
      cout << "scaleidx : " << scaleidx << endl;
      assert(scaleidx == -1);

      QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
	padZeros(QString::number(imgidx), 4);

      QString qsInputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";

      QString qsResultsDir;

      /** 
	  again, samples might come from part detections or part posteriors (here we just look at the source dir 
	  since the "dai_samples_type" could be set incorrectly
      */
      if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) {
	qsResultsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
					      "/part_marginals_samples_post");
      }
      else if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
	qsResultsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
					      "/test_scoregrid_samples_post");	  
      }
      else {
	assert(false && "unknown part samples type");
      }

      QString qsResultsFilename = qsResultsDir + "/samples_imgidx" + padZeros(QString::number(imgidx), 4) + "_post.mat";
	
      QString qsVarName = "samples_post_part" + QString::number(pidx);

      vector<int> vect_scale_idx;
      vector<int> vect_rot_idx; 
      vector<int> vect_iy; 
      vector<int> vect_ix;
      vector<int> vect_didx;

      bool res5 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scale_idx", vect_scale_idx);
      bool res1 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_rot_idx", vect_rot_idx);
      bool res2 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_iy", vect_iy);
      bool res3 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_ix", vect_ix);

      bool res6 = false;

      if (didx != -1) {
	res6 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_didx", vect_didx);
	assert(res6);
      }
      
      assert(res1 && res2 && res3 && res5);

      boost_math::double_vector part_posterior;
      matlab_io::mat_load_double_vector(qsResultsFilename, qsVarName, part_posterior);

      assert(part_posterior.size() == vect_rot_idx.size() && 
	     part_posterior.size() == vect_iy.size() && 
	     part_posterior.size() == vect_ix.size() && 
	     part_posterior.size() == vect_scale_idx.size() &&
	     part_posterior.size() > 0);

      if (didx != -1) {
	assert(part_posterior.size() == vect_didx.size());
      }
	
      double maxval;
      int maxidx = -1;

      if (didx == -1 || !res6)
	boost_math::get_max(part_posterior, maxval, maxidx);
      else {

	for (uint idx = 0; idx < part_posterior.size(); ++idx) {
	  if (vect_didx[idx] == didx) {
	    if (maxidx == -1) {
	      maxidx = idx;
	      maxval = part_posterior[idx];
	    }
	    else {
	      if (maxval < part_posterior[idx]) {
		maxval = part_posterior[idx];
		maxidx = idx;
	      }
	    }
	  }
	}

      }
	
      assert(maxidx >= 0);

      max_scaleidx = vect_scale_idx[maxidx];
      max_ridx = vect_rot_idx[maxidx];
      max_ix = vect_ix[maxidx];
      max_iy = vect_iy[maxidx];

      best_part_hyp.push_back(object_detect::PartHyp(part_app.m_exp_param, max_scaleidx, max_ridx, max_ix, max_iy, maxval));
    }// parts
  }
  else if (eval_type == EVAL_TYPE_ROI) {
    
    assert(scaleidx == -1);

    QString qsFilename = qsHypDirName + "/pose_est_imgidx" + padZeros(QString::number(imgidx), 4) + 
      "_roi" + padZeros(QString::number(didx), 4) + ".mat";

    FloatGrid2 best_conf = matlab_io::mat_load_multi_array<FloatGrid2>(qsFilename, "best_conf");
      
    assert((int)best_conf.shape()[0] == part_app.m_part_conf.part_size());
    
    for (int pidx = 0; pidx < (int)best_conf.shape()[0]; ++pidx) {
      object_detect::PartHyp _part_hyp;

      using boost::multi_array_types::index_range;

      _part_hyp.fromVect(best_conf[boost::indices[pidx][index_range()]]);
      best_part_hyp.push_back(_part_hyp);
    }

  }
  else {
    assert(false && "unknown evaluation type");
  }

  for (int pidx = 0; pidx < (int)best_part_hyp.size(); ++pidx) {
    PartBBox bbox;
    best_part_hyp[pidx].getPartBBox(part_app.m_window_param.part(pidx), bbox);
    eval_bbox.push_back(bbox);
  }
  
  /** convert to evaluation parts if necessary */
  if (part_app.m_exp_param.part_conf_type() == "human_full_joints") {
    vector<PartBBox> _eval_bbox;

    assert(eval_bbox.size() == 18);

    /** MA: the order in merge is important (it affects which parts are used to compute min_proj_y and max_proj_y) */

    // legs
    _eval_bbox.push_back(bbox_merge(eval_bbox[0], eval_bbox[1]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[2], eval_bbox[3]));

    _eval_bbox.push_back(bbox_merge(eval_bbox[6], eval_bbox[7]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[4], eval_bbox[5]));

    /** 
	for the "torso" and "head" parts the bounding box is copied directly -> remove offset 
    */

    // torso
    _eval_bbox.push_back(eval_bbox[8]);
    //double scale8 = scale_from_index(part_app.m_exp_param, best_part_pos(8, 0));
    double scale8 = best_part_hyp[8].m_scale;

    _eval_bbox[4].min_proj_y += scale8 * part_conf_eval.part(4).ext_y_neg();
    _eval_bbox[4].max_proj_y -= scale8 * part_conf_eval.part(4).ext_y_pos();

    // head
    _eval_bbox.push_back(eval_bbox[17]);
    //double scale17 = scale_from_index(part_app.m_exp_param, best_part_pos(17, 0));
    double scale17 = best_part_hyp[17].m_scale;
    
    _eval_bbox[5].min_proj_y += scale17 * part_conf_eval.part(5).ext_y_neg();
    _eval_bbox[5].max_proj_y -= scale17 * part_conf_eval.part(5).ext_y_pos();

    // arms
    _eval_bbox.push_back(bbox_merge(eval_bbox[9], eval_bbox[10]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[11], eval_bbox[12]));
      
    _eval_bbox.push_back(bbox_merge(eval_bbox[15], eval_bbox[16]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[13], eval_bbox[14]));
    
    float shrink_factor_x = 0.7;

    /** looks better and doesn't affect evaluation (only endpoints matter) */
    for (uint pidx = 0; pidx < _eval_bbox.size(); ++pidx) {
      shrink_factor_x = get_shrink_factor_x(part_app.m_part_conf.part(pidx).ext_x_pos(),pidx);
      _eval_bbox[pidx].max_proj_x *= shrink_factor_x;
      _eval_bbox[pidx].min_proj_x *= shrink_factor_x;
    }
    
    eval_bbox = _eval_bbox;
  }
  else if (part_app.m_exp_param.part_conf_type() == "human_full_torso4") {
    // same as "human_full_joints" but with different part id's and also need to merge torso
    vector<PartBBox> _eval_bbox;

    assert(eval_bbox.size() == 22);

    // legs
    _eval_bbox.push_back(bbox_merge(eval_bbox[0], eval_bbox[1]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[2], eval_bbox[3]));

    _eval_bbox.push_back(bbox_merge(eval_bbox[6], eval_bbox[7]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[4], eval_bbox[5]));

    /** MA: for torso we could probably just take position of the root part ??? */

    // torso 
    _eval_bbox.push_back(bbox_merge(eval_bbox[16], eval_bbox[17], eval_bbox[18], eval_bbox[19]));
    //_eval_bbox.push_back(bbox_merge(eval_bbox[16], eval_bbox[17]));

    /** 
	for the "head" part the bounding box is copied directly -> remove offset 
    */

    // head
    _eval_bbox.push_back(eval_bbox[20]);
    //double scale20 = scale_from_index(part_app.m_exp_param, best_part_pos(20, 0));
    double scale20 = best_part_hyp[20].m_scale;

    _eval_bbox[5].min_proj_y += scale20 * part_conf_eval.part(5).ext_y_neg();
    _eval_bbox[5].max_proj_y -= scale20 * part_conf_eval.part(5).ext_y_pos();

    // arms
    _eval_bbox.push_back(bbox_merge(eval_bbox[8], eval_bbox[9]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[10], eval_bbox[11]));
      
    _eval_bbox.push_back(bbox_merge(eval_bbox[14], eval_bbox[15]));
    _eval_bbox.push_back(bbox_merge(eval_bbox[12], eval_bbox[13]));

    /** looks better and doesn't affect evaluation (only endpoints matter) */
    for (uint pidx = 0; pidx < _eval_bbox.size(); ++pidx) {

      float shrink_factor_x = 0.7;
      /** torso does not contain extra offset since it is merged from "shoulder" and "hips" parts */
      if (pidx != 4) {
	shrink_factor_x = get_shrink_factor_x(part_app.m_part_conf.part(pidx).ext_x_pos(),pidx);
	_eval_bbox[pidx].max_proj_x *= shrink_factor_x;
	_eval_bbox[pidx].min_proj_x *= shrink_factor_x;
      }
    }
    
    eval_bbox = _eval_bbox;
  }
  else if (part_app.m_exp_param.part_conf_type() == "human_full_14_parts" && bMerge) {
    vector<PartBBox> _eval_bbox;
   
    assert(eval_bbox.size() == 14);
    
    /** MA: the order in merge is important (it affects which parts are used to compute min_proj_y and max_proj_y) */

    // legs
    _eval_bbox.push_back(bbox_merge(eval_bbox[0], eval_bbox[1], part_app.m_exp_param));
    _eval_bbox.push_back(bbox_merge(eval_bbox[1], eval_bbox[2], part_app.m_exp_param));
    _eval_bbox.push_back(bbox_merge(eval_bbox[4], eval_bbox[3], part_app.m_exp_param));
    _eval_bbox.push_back(bbox_merge(eval_bbox[5], eval_bbox[4], part_app.m_exp_param));

    /** 
	for the "torso" and "head" parts the bounding box is copied directly -> remove offset 
    */

    // torso
    _eval_bbox.push_back(eval_bbox[6]);
    //double scale6 = scale_from_index(part_app.m_exp_param, best_part_pos(6, 0));
    double scale6 = best_part_hyp[6].m_scale;

    _eval_bbox[4].min_proj_y += scale6 * part_conf_eval.part(4).ext_y_neg();
    _eval_bbox[4].max_proj_y -= scale6 * part_conf_eval.part(4).ext_y_pos();

    // head
    _eval_bbox.push_back(eval_bbox[7]);
    //double scale7 = scale_from_index(part_app.m_exp_param, best_part_pos(7, 0));
    double scale7 = best_part_hyp[7].m_scale;

    _eval_bbox[5].min_proj_y += scale7 * part_conf_eval.part(5).ext_y_neg();
    _eval_bbox[5].max_proj_y -= scale7 * part_conf_eval.part(5).ext_y_pos();

    // arms
    _eval_bbox.push_back(bbox_merge(eval_bbox[8], eval_bbox[9], part_app.m_exp_param));
    _eval_bbox.push_back(bbox_merge(eval_bbox[9], eval_bbox[10], part_app.m_exp_param));
    _eval_bbox.push_back(bbox_merge(eval_bbox[12], eval_bbox[11], part_app.m_exp_param));
    _eval_bbox.push_back(bbox_merge(eval_bbox[13], eval_bbox[12], part_app.m_exp_param));
                
    vector<double> ext_x_pos;
    for (uint pidx = 0; pidx < 4; ++pidx)
      ext_x_pos.push_back(part_app.m_part_conf.part(pidx).ext_x_pos());
    
    ext_x_pos.push_back(part_app.m_part_conf.part(6).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(7).ext_x_pos());
    
    for (uint pidx = 9; pidx < 13; ++pidx)
      ext_x_pos.push_back(part_app.m_part_conf.part(pidx).ext_x_pos());
        
    float shrink_factor_x = 0.7;
    /** looks better and doesn't affect evaluation (only endpoints matter) */
    for (uint pidx = 0; pidx < _eval_bbox.size(); ++pidx) {
      shrink_factor_x = get_shrink_factor_x(ext_x_pos[pidx],pidx);
      _eval_bbox[pidx].max_proj_x *= shrink_factor_x;
      _eval_bbox[pidx].min_proj_x *= shrink_factor_x;
    }
    
    eval_bbox = _eval_bbox;
  }
  else if (part_app.m_exp_param.part_conf_type() == "human_full_22_parts" && bMerge){
    vector<PartBBox> _eval_bbox;
   
    assert(eval_bbox.size() == 22);
    
    /** MA: the order in merge is important (it affects which parts are used to compute min_proj_y and max_proj_y) */

    // legs
    
    _eval_bbox.push_back(eval_bbox[1]);
    _eval_bbox.push_back(eval_bbox[3]);
    _eval_bbox.push_back(eval_bbox[6]);
    _eval_bbox.push_back(eval_bbox[8]);
    _eval_bbox[0].min_proj_y += part_conf_eval.part(0).ext_y_neg();
    _eval_bbox[0].max_proj_y -= part_conf_eval.part(0).ext_y_pos();
    _eval_bbox[1].min_proj_y += part_conf_eval.part(1).ext_y_neg();
    _eval_bbox[1].max_proj_y -= part_conf_eval.part(1).ext_y_pos();
    _eval_bbox[2].min_proj_y += part_conf_eval.part(2).ext_y_neg();
    _eval_bbox[2].max_proj_y -= part_conf_eval.part(2).ext_y_pos();
    _eval_bbox[3].min_proj_y += part_conf_eval.part(3).ext_y_neg();
    _eval_bbox[3].max_proj_y -= part_conf_eval.part(3).ext_y_pos();
    
    /** 
	for the "torso" and "head" parts the bounding box is copied directly -> remove offset 
    */

    // torso
    _eval_bbox.push_back(eval_bbox[10]);
    double scale10 = best_part_hyp[10].m_scale;

    _eval_bbox[4].min_proj_y += scale10 * part_conf_eval.part(4).ext_y_neg();
    _eval_bbox[4].max_proj_y -= scale10 * part_conf_eval.part(4).ext_y_pos();

    // head
    _eval_bbox.push_back(eval_bbox[11]);
    double scale11 = best_part_hyp[11].m_scale;

    _eval_bbox[5].min_proj_y += scale11 * part_conf_eval.part(5).ext_y_neg();
    _eval_bbox[5].max_proj_y -= scale11 * part_conf_eval.part(5).ext_y_pos();

    // arms
   
    _eval_bbox.push_back(eval_bbox[13]);
    _eval_bbox.push_back(eval_bbox[15]);
    _eval_bbox.push_back(eval_bbox[18]);
    _eval_bbox.push_back(eval_bbox[20]);

    _eval_bbox[6].min_proj_y += part_conf_eval.part(6).ext_y_neg();
    _eval_bbox[6].max_proj_y -= part_conf_eval.part(6).ext_y_pos();
    _eval_bbox[7].min_proj_y += part_conf_eval.part(7).ext_y_neg();
    _eval_bbox[7].max_proj_y -= part_conf_eval.part(7).ext_y_pos();
    _eval_bbox[8].min_proj_y += part_conf_eval.part(8).ext_y_neg();
    _eval_bbox[8].max_proj_y -= part_conf_eval.part(8).ext_y_pos();
    _eval_bbox[9].min_proj_y += part_conf_eval.part(9).ext_y_neg();
    _eval_bbox[9].max_proj_y -= part_conf_eval.part(9).ext_y_pos();
          
    vector<double> ext_x_pos;
    ext_x_pos.push_back(part_app.m_part_conf.part(1).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(3).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(6).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(8).ext_x_pos());
    
    ext_x_pos.push_back(part_app.m_part_conf.part(10).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(11).ext_x_pos());
    
    ext_x_pos.push_back(part_app.m_part_conf.part(12).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(14).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(16).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(18).ext_x_pos());
        
    float shrink_factor_x = 0.7;
    /** looks better and doesn't affect evaluation (only endpoints matter) */
    for (uint pidx = 0; pidx < _eval_bbox.size(); ++pidx) {
      shrink_factor_x = get_shrink_factor_x(ext_x_pos[pidx],pidx);
      _eval_bbox[pidx].max_proj_x *= shrink_factor_x;
      _eval_bbox[pidx].min_proj_x *= shrink_factor_x;
    }
    
    eval_bbox = _eval_bbox;
  }
  else if (part_app.m_exp_param.part_conf_type() == "human_full_12_parts" && bMerge){
    vector<PartBBox> _eval_bbox;
    
    assert(eval_bbox.size() == 12);
    
    /** MA: the order in merge is important (it affects which parts are used to compute min_proj_y and max_proj_y) */

    // torso
    _eval_bbox.push_back(eval_bbox[0]);
    double scale10 = best_part_hyp[0].m_scale;

    _eval_bbox[0].min_proj_y += scale10 * part_conf_eval.part(0).ext_y_neg();
    _eval_bbox[0].max_proj_y -= scale10 * part_conf_eval.part(0).ext_y_pos();

    // head
    _eval_bbox.push_back(eval_bbox[1]);
    double scale11 = best_part_hyp[1].m_scale;

    _eval_bbox[1].min_proj_y += scale11 * part_conf_eval.part(1).ext_y_neg();
    _eval_bbox[1].max_proj_y -= scale11 * part_conf_eval.part(1).ext_y_pos();
    
    _eval_bbox.push_back(eval_bbox[3]);
    _eval_bbox.push_back(eval_bbox[5]);
    _eval_bbox.push_back(eval_bbox[8]);
    _eval_bbox.push_back(eval_bbox[10]);
    _eval_bbox[2].min_proj_y += part_conf_eval.part(2).ext_y_neg();
    _eval_bbox[2].max_proj_y -= part_conf_eval.part(2).ext_y_pos();
    _eval_bbox[3].min_proj_y += part_conf_eval.part(3).ext_y_neg();
    _eval_bbox[3].max_proj_y -= part_conf_eval.part(3).ext_y_pos();
    _eval_bbox[4].min_proj_y += part_conf_eval.part(4).ext_y_neg();
    _eval_bbox[4].max_proj_y -= part_conf_eval.part(4).ext_y_pos();
    _eval_bbox[5].min_proj_y += part_conf_eval.part(5).ext_y_neg();
    _eval_bbox[5].max_proj_y -= part_conf_eval.part(5).ext_y_pos();
          
    vector<double> ext_x_pos;

    ext_x_pos.push_back(part_app.m_part_conf.part(0).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(1).ext_x_pos());
    
    ext_x_pos.push_back(part_app.m_part_conf.part(3).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(5).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(8).ext_x_pos());
    ext_x_pos.push_back(part_app.m_part_conf.part(10).ext_x_pos());
        
    float shrink_factor_x = 0.7;
    /** looks better and doesn't affect evaluation (only endpoints matter) */
    for (uint pidx = 0; pidx < _eval_bbox.size(); ++pidx) {
      shrink_factor_x = get_shrink_factor_x(ext_x_pos[pidx],pidx, 0);
      _eval_bbox[pidx].max_proj_x *= shrink_factor_x;
      _eval_bbox[pidx].min_proj_x *= shrink_factor_x;
    }
    
    eval_bbox = _eval_bbox;
 }
 else {
    /** 
	remove part offset in y dir (this is important for comparison to groundtruth !!!)
	shrink box in x direction

	MA: from now on all customization here should be controlled through "part_conf_type" parameters
    */
    for (uint pidx = 0; pidx < eval_bbox.size(); ++pidx) {

      double scale = best_part_hyp[pidx].m_scale;
      assert(scale > 0);
      
      eval_bbox[pidx].min_proj_y += scale * part_conf_eval.part(pidx).ext_y_neg();
      eval_bbox[pidx].max_proj_y -= scale * part_conf_eval.part(pidx).ext_y_pos();
      
      // shrink x for visualization purposes
      float shrink_factor_x = 0.7;
      if (part_app.m_exp_param.part_conf_type() == "human_full")
	shrink_factor_x = get_shrink_factor_x(part_app.m_part_conf.part(pidx).ext_x_pos(),pidx);
      //shrink_factor_x = get_shrink_factor_x(part_conf_eval.part(pidx_part_conf).ext_x_pos(),pidx);
      
      eval_bbox[pidx].max_proj_x *= shrink_factor_x;
      eval_bbox[pidx].min_proj_x *= shrink_factor_x;
    }
  }
}

/**
   modified copy/paste of "eval_segments"
 */

void vis_segments(const PartApp &part_app, const AnnotationList &annolist, 
		  int firstidx, int lastidx, QString qsHypDirName, const int eval_type)
{
  
  /** load the configuration used for evaluation */
  PartConfig part_conf_eval;
    
  if (part_app.m_exp_param.has_part_conf_eval()) {
    assert(part_app.m_qsExpParam.isEmpty() == false);
    QString qsPartConfEval = complete_relative_path(QString::fromStdString(part_app.m_exp_param.part_conf_eval()), part_app.m_qsExpParam);

    parse_message_from_text_file(qsPartConfEval, part_conf_eval);
  }
  else {
    part_conf_eval = part_app.m_part_conf;
  }

  QString qsSegEvalImagesDir;
  QString qsSegEndPointsDir;

  switch (eval_type) {
  case EVAL_TYPE_UNARIES: {
    qsSegEvalImagesDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
			  "/test_scoregrid/seg_vis_images").c_str();
    break;
  }
  case EVAL_TYPE_PS: {
    
    qsSegEvalImagesDir = qsHypDirName + "/seg_vis_images";
    qsSegEndPointsDir = qsHypDirName + "/seg_endpoints";
    break;
  }
  case EVAL_TYPE_DISC_PS: {

    QString qsSamplesDir;

    /** in each experiment samples come either from posteriors or from part detections */

    if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) {
      qsSamplesDir = "part_marginals_samples_post";
    }
    else if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
      qsSamplesDir = "test_scoregrid_samples_post";
    }
    else {
      assert(false && "unknown part samples type");
    }
      
    qsSegEvalImagesDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir()) + 
      "/" + qsSamplesDir + "/seg_vis_images";
    
    qsSegEndPointsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir()) + 
      "/" + qsSamplesDir + "/seg_endpoints";
    
    break;
  }

  default: {
    assert(false && "unknown evaluation type");
  }
  }

  if (!filesys::check_dir(qsSegEvalImagesDir))
    filesys::create_dir(qsSegEvalImagesDir);

  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {      

    QImage _img;
    assert(_img.load(annolist[imgidx].imageName().c_str()));
    QImage img = _img.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&img);
    painter.setRenderHints(QPainter::Antialiasing);

    if (eval_type == EVAL_TYPE_DISC_PS) {
      
      /** in discrete version we currently compute the posteriors accross all scales */
      uint num_subjects = 1;
      
      if (part_app.m_exp_param.dai_bbox_prior())
	num_subjects = part_app.m_test_annolist[imgidx].m_vRects.size();

      for (uint didx = 0; didx < num_subjects; ++didx) {
	vector<PartBBox> eval_bbox;
	
	if (num_subjects == 1)
	  vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox);
	else
	  vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox, qsHypDirName, didx);

	assert((int)eval_bbox.size() == part_conf_eval.part_size());

	QString qsSegEndPoints = qsSegEndPointsDir + "/endpoints_" + padZeros(QString::number(imgidx), 4) + ".mat";
	boost_math::double_matrix part_endpoints;
	matlab_io::mat_load_double_matrix(qsSegEndPoints, "endpoints", part_endpoints);
	
	int coloridx = didx + 1;
	int pen_width = 3;

	for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx){
	  coloridx = 1 - part_endpoints(pidx,4);
	  if (part_endpoints.size2() == 6 && part_endpoints(pidx,5) == 1)
	    coloridx = 2;
	  draw_bbox(painter, eval_bbox[pidx], coloridx, pen_width);
	}
      }

    }
    else {

      for (uint scaleidx = 0; scaleidx < part_app.m_exp_param.num_scale_steps(); ++scaleidx) {
	
	vector<PartBBox> eval_bbox;
	vis_eval_helper(part_app, imgidx, scaleidx, eval_type, part_conf_eval, eval_bbox, qsHypDirName, -1);
	assert((int)eval_bbox.size() == part_conf_eval.part_size());

	int coloridx = scaleidx + 1;
	int pen_width = 3;

	QString qsSegEndPoints = qsSegEndPointsDir + "/endpoints_" + padZeros(QString::number(imgidx), 4) + ".mat";
	boost_math::double_matrix part_endpoints;
	matlab_io::mat_load_double_matrix(qsSegEndPoints, "endpoints", part_endpoints);

	for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx){  
	  coloridx = 0 ;//1 - part_endpoints(pidx,4);
	  //if (part_endpoints.size2() == 6 && part_endpoints(pidx,5) == 1)
	  //  coloridx = 2;
	  draw_bbox(painter, eval_bbox[pidx], coloridx, pen_width);
	}
      }// scales
    }

    QString qsFilename2 = qsSegEvalImagesDir + "/img_" + padZeros(QString::number(imgidx), 4) + ".png";
    cout << "saving " << qsFilename2.toStdString() << endl;
    assert(img.save(qsFilename2));
    
  }// images
}


/**
   note: there are several peculiarities when evaluating TUD Pedestrians (-> should set the "part_conf_type" properly)
           - shrinking of bounding boxes should be turned off
           - endpoints of foot part are along the x axis, not along the y axis 

           currently all this things are HARDCODED (MA: still true ?)
 */
/*
void eval_segments(const PartApp &part_app, 
		   const AnnotationList &annolist, 
		   int firstidx, int lastidx, double &ratio, 
		   QString qsHypDirName, const int eval_type, 
		   int eval_didx)
{
  int seg_total = 0;
  int seg_correct = 0;
  
  QString qsSegEndPointsDir;
  
  // load the configuration used for evaluation 
  PartConfig part_conf_eval;
    
  if (part_app.m_exp_param.has_part_conf_eval()) {
    assert(part_app.m_qsExpParam.isEmpty() == false);
    QString qsPartConfEval = complete_relative_path(QString::fromStdString(part_app.m_exp_param.part_conf_eval()), part_app.m_qsExpParam);

    parse_message_from_text_file(qsPartConfEval, part_conf_eval);
  }
  else {
    part_conf_eval = part_app.m_part_conf;
  }
  
  vector<int> per_segment_correct(part_conf_eval.part_size(), 0);
    
  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {      

    if (eval_didx >= 0) {
      assert(eval_didx < (int)annolist[imgidx].size());
    }

    int scaleidx = 0;
    double scale = scale_from_index(part_app.m_exp_param, scaleidx);
        
    int num_correct_cur_imgidx = 0;

    vector<PartBBox> eval_bbox;

    if (eval_type == EVAL_TYPE_DISC_PS) {
      
      QString qsSamplesDir;

      // in each experiment samples come either from posteriors or from part detections 
      if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) {
	qsSamplesDir = "part_marginals_samples_post";
      }
      else if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
	qsSamplesDir = "test_scoregrid_samples_post";
      }
      else {
	assert(false && "unknown part samples type");
      }
      
      qsSegEndPointsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir()) + 
	"/" + qsSamplesDir + "/seg_endpoints";
      
      if (eval_didx >= 0) {
	vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox, qsHypDirName, eval_didx);
      }
      else {
	vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox, qsHypDirName);
      }
      
    }
    else {
      qsSegEndPointsDir = qsHypDirName + "/seg_endpoints";
      vis_eval_helper(part_app, imgidx, scaleidx, eval_type, part_conf_eval, eval_bbox, qsHypDirName, -1);
    }
    
    assert((int)eval_bbox.size() == part_conf_eval.part_size());

    if (!filesys::check_dir(qsSegEndPointsDir))
      filesys::create_dir(qsSegEndPointsDir);
    
    boost_math::double_matrix part_endpoints;
    part_endpoints.resize(part_conf_eval.part_size() + 1, 6);
    
    for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) {
      PartBBox bbox = eval_bbox[pidx];
            
      // save endpoints for external evaluation 
      double seg_len;
      double_vector endpoint_top;
      double_vector endpoint_bottom;
      get_bbox_endpoints(bbox, endpoint_top, endpoint_bottom, seg_len);
      
      part_endpoints(pidx,0) = endpoint_bottom(0);
      part_endpoints(pidx,1) = endpoint_bottom(1);
      part_endpoints(pidx,2) = endpoint_top(0);
      part_endpoints(pidx,3) = endpoint_top(1);
      part_endpoints(pidx,4) = 0;
      part_endpoints(pidx,5) = 0;
      
      int rectidx = 0;
 
      if (eval_didx >= 0)
	rectidx = eval_didx;

      assert(annolist[imgidx].size() > 0);
      
      if (annolist[imgidx][rectidx].m_vAnnoPoints.size() > 0){
	PartBBox gt_bbox;
	
	get_part_bbox(annolist[imgidx][rectidx], part_conf_eval.part(pidx), gt_bbox, scale);
	
	gt_bbox.min_proj_y += scale * part_conf_eval.part(pidx).ext_y_neg();
	gt_bbox.max_proj_y -= scale * part_conf_eval.part(pidx).ext_y_pos();
	
	bool match_x_axis = false;
	bool match = is_gt_match(gt_bbox, bbox, match_x_axis);

	if (match) {
	  ++seg_correct;
	  per_segment_correct[pidx]++;
	  part_endpoints(pidx,4) = 1;
	}
	
	++seg_total;
	cout << "pidx: " << pidx << ", match: " << (int)match  << endl;
	
	num_correct_cur_imgidx += (int)match;
      }
    }// parts
    
    cout << imgidx << " " << num_correct_cur_imgidx << endl;
    part_endpoints(part_conf_eval.part_size(),0) = 1.0*num_correct_cur_imgidx / part_conf_eval.part_size();
    part_endpoints(part_conf_eval.part_size(),1) = 0;
    part_endpoints(part_conf_eval.part_size(),2) = 0;
    part_endpoints(part_conf_eval.part_size(),3) = 0;
    part_endpoints(part_conf_eval.part_size(),4) = 0;
    
    QString qsSegEndPoints = qsSegEndPointsDir + "/endpoints_" + padZeros(QString::number(imgidx), 4) + ".mat";
    matlab_io::mat_save_double_matrix(qsSegEndPoints, "endpoints", part_endpoints);
    cout << "save " << qsSegEndPoints.toStdString().c_str() << endl;
    
  }// images

  cout << "seg_correct: " << seg_correct << endl;
  cout << "seg_total: " << seg_total << endl;

  if (seg_total != 0) {
    ratio = seg_correct / (double)seg_total;
    cout << "ratio: " << ratio << endl;
    
    int per_segment_total = seg_total / part_conf_eval.part_size();
    
    cout << endl;
    for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) {  
      cout << "part: " << pidx << 
        ", correct: " << per_segment_correct[pidx] << 
        ", total: " << per_segment_total << 
        ", ratio: " << per_segment_correct[pidx]/(double)per_segment_total << endl;
    }
  }
  else
    ratio = 0;
}
*/

void eval_segments(const PartApp &part_app, 
		   const AnnotationList &annolist, 
		   int firstidx, int lastidx, double &ratio, 
		   QString qsHypDirName, const int eval_type, 
		   int eval_didx)
{
  int seg_total = 0;
  int seg_correct = 0;
  
  QString qsSegEndPointsDir;
  
  /** load the configuration used for evaluation */
  PartConfig part_conf_eval;
    
  if (part_app.m_exp_param.has_part_conf_eval()) {
    assert(part_app.m_qsExpParam.isEmpty() == false);
    QString qsPartConfEval = complete_relative_path(QString::fromStdString(part_app.m_exp_param.part_conf_eval()), part_app.m_qsExpParam);

    parse_message_from_text_file(qsPartConfEval, part_conf_eval);
  }
  else {
    part_conf_eval = part_app.m_part_conf;
  }
  
  vector<int> per_segment_correct(part_conf_eval.part_size(), 0);
  vector<int> per_segment_total_part(part_conf_eval.part_size(), 0);

  //QString qsFilename = "/BS/leonid-people-3d/work/data/new_dataset/annolist_merge_batch1to8/test/truncImgs.mat";
  //cout << "loading " << qsFilename.toStdString().c_str() << endl;
  //boost_math::double_matrix skipidx;
  //matlab_io::mat_load_double_matrix(qsFilename, "skipidx", skipidx);

  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {      

    //if (skipidx(imgidx,0))
    //  continue;
    
    if (eval_didx >= 0) {
      assert(eval_didx < (int)annolist[imgidx].size());
    }
    
    int scaleidx = 0;
    double scale = scale_from_index(part_app.m_exp_param, scaleidx);
        
    int num_correct_cur_imgidx = 0;

    vector<PartBBox> eval_bbox;

    if (eval_type == EVAL_TYPE_DISC_PS) {
      
      /** scaleidx == -1 means ignore scale */
      //vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox, qsHypDirName);

      QString qsSamplesDir;

      /** in each experiment samples come either from posteriors or from part detections */

      if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) {
	qsSamplesDir = "part_marginals_samples_post";
      }
      else if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
	qsSamplesDir = "test_scoregrid_samples_post";
      }
      else {
	assert(false && "unknown part samples type");
      }
      
      qsSegEndPointsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir()) + 
	"/" + qsSamplesDir + "/seg_endpoints";
      
      if (eval_didx >= 0) {
	vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox, qsHypDirName, eval_didx);
      }
      else {
	vis_eval_helper(part_app, imgidx, -1, eval_type, part_conf_eval, eval_bbox, qsHypDirName);
      }
      
    }
    else {
      qsSegEndPointsDir = qsHypDirName + "/seg_endpoints";
      vis_eval_helper(part_app, imgidx, scaleidx, eval_type, part_conf_eval, eval_bbox, qsHypDirName, -1);
    }
    
    assert((int)eval_bbox.size() == part_conf_eval.part_size());

    if (!filesys::check_dir(qsSegEndPointsDir))
      filesys::create_dir(qsSegEndPointsDir);
    
    boost_math::double_matrix part_endpoints;
    part_endpoints.resize(part_conf_eval.part_size() + 1, 6);
    
    int num_segment_cur_imgidx = 0;
    for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) {
      PartBBox bbox = eval_bbox[pidx];
      
      /* save endpoints for external evaluation */
      double seg_len;
      double_vector endpoint_top;
      double_vector endpoint_bottom;
      get_bbox_endpoints(bbox, endpoint_top, endpoint_bottom, seg_len);
      
      part_endpoints(pidx,0) = endpoint_bottom(0);
      part_endpoints(pidx,1) = endpoint_bottom(1);
      part_endpoints(pidx,2) = endpoint_top(0);
      part_endpoints(pidx,3) = endpoint_top(1);
      part_endpoints(pidx,4) = 0;
      part_endpoints(pidx,5) = 0;
            
      int rectidx = 0;
 
      if (eval_didx >= 0)
	rectidx = eval_didx;

      assert(annolist[imgidx].size() > 0);
            
      if (annolist[imgidx][rectidx].m_vAnnoPoints.size() > 0){
	PartBBox gt_bbox;
	
	if (not annorect_has_part(annolist[imgidx][rectidx], part_conf_eval.part(pidx)))
	  continue;

	get_part_bbox(annolist[imgidx][rectidx], part_conf_eval.part(pidx), gt_bbox, scale);

	gt_bbox.min_proj_y += scale * part_conf_eval.part(pidx).ext_y_neg();
	gt_bbox.max_proj_y -= scale * part_conf_eval.part(pidx).ext_y_pos();
	
	bool match_x_axis = false;
	bool match = is_gt_match(gt_bbox, bbox, match_x_axis);

	if (match) {
	  ++seg_correct;
	  per_segment_correct[pidx]++;
	  part_endpoints(pidx,4) = 1;
	}
	
	++seg_total;
	cout << "pidx: " << pidx << ", match: " << (int)match  << endl;
	
	num_correct_cur_imgidx += (int)match;
	num_segment_cur_imgidx++;
	
	per_segment_total_part[pidx]++;
	part_endpoints(pidx,5) = 1;
	//cout << part_endpoints(pidx,5) << endl;
      }
    }// parts
    
    cout << imgidx << " " << num_correct_cur_imgidx << endl;
    part_endpoints(part_conf_eval.part_size(),0) = 1.0*num_correct_cur_imgidx / num_segment_cur_imgidx;
    part_endpoints(part_conf_eval.part_size(),1) = 0;
    part_endpoints(part_conf_eval.part_size(),2) = 0;
    part_endpoints(part_conf_eval.part_size(),3) = 0;
    part_endpoints(part_conf_eval.part_size(),4) = 0;
    
    QString qsSegEndPoints = qsSegEndPointsDir + "/endpoints_" + padZeros(QString::number(imgidx), 4) + ".mat";
    matlab_io::mat_save_double_matrix(qsSegEndPoints, "endpoints", part_endpoints);
    cout << "save " << qsSegEndPoints.toStdString().c_str() << endl;
    
  }// images

  cout << "seg_correct: " << seg_correct << endl;
  cout << "seg_total: " << seg_total << endl;

  if (seg_total != 0) {
    ratio = seg_correct / (double)seg_total;
    cout << "ratio: " << ratio << endl;
    
    int per_segment_total = seg_total / part_conf_eval.part_size();
    
    cout << endl;
    for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) {  
      per_segment_total = per_segment_total_part[pidx];
      if (per_segment_correct[pidx] == 0 && per_segment_total == 0)
	per_segment_total = 1;
      
      cout << "part: " << pidx << 
        ", correct: " << per_segment_correct[pidx] << 
        ", total: " << per_segment_total << 
        ", ratio: " << per_segment_correct[pidx]/(double)per_segment_total << endl;
    }
  }
  else
    ratio = 0;
}


bool compare_vec(vector<float> a, vector<float> b){
  return (a[0] > b[0]);
}

bool compare_hyp(object_detect::PartHyp a, object_detect::PartHyp b){
  return (a.m_score > b.m_score);
}

bool compare_imgidx(object_detect::PartHyp a, object_detect::PartHyp b){
  return (a.m_imgidx < b.m_imgidx);
}

void get_scores(const PartApp &part_app, int imgidx, int scaleidx, int pidx,
		vector<object_detect::PartHyp> &part_hyp_list)
{

  int nHyp = part_hyp_list.size();
  
  int max_scaleidx = scaleidx;
  int max_ridx= -1;
  int max_ix = -1, max_iy = -1;
  
  vector<vector<FloatGrid2> > part_detections;
  
  bool flip = false;
  bool bInterpolate = false;
  QString qsScoreGridDir = part_app.m_exp_param.scoregrid_train_dir().c_str();
  QString qsImgName = part_app.m_train_annolist[imgidx].imageName().c_str();
  part_app.loadScoreGrid(part_detections, imgidx, pidx, flip, bInterpolate, qsScoreGridDir, qsImgName);
  
  assert(part_detections.size() == 1);
  
  max_ridx = 0;
  max_iy = 0;
  max_ix = 0;
  
  float max_val = part_detections[0][0][0][0];
  
  int nRotations = part_app.m_exp_param.num_rotation_steps();
  assert((int)part_detections[0].size() == nRotations);
  int img_height = part_detections[0][0].shape()[0];
  int img_width = part_detections[0][0].shape()[1];
  
  cout << "img_width: " << img_width << ", img_height: " << img_height << endl;
  
  vector<vector<float> > all_detections(nRotations*img_width*img_height ,vector<float>(4));
  int idx = -1;
  for (int ridx = 0; ridx < nRotations; ++ridx)
    for (int ix = 0; ix < img_width; ++ix)
      for (int iy = 0; iy < img_height; ++iy) {
	idx++;
	all_detections[idx][0] = part_detections[0][ridx][iy][ix];
	all_detections[idx][1] = ridx;
	all_detections[idx][2] = ix;
	all_detections[idx][3] = iy;
      }

  sort(all_detections.begin(), all_detections.end(), compare_vec);
  
  for(uint hypidx = 0; hypidx < nHyp; hypidx++){
    max_val = all_detections[hypidx][0];
    max_ridx = all_detections[hypidx][1];
    max_ix = all_detections[hypidx][2];
    max_iy = all_detections[hypidx][3];
    part_hyp_list[hypidx] = object_detect::PartHyp(part_app.m_exp_param, max_scaleidx, max_ridx, max_ix, max_iy, max_val, imgidx);
  }
}

void eval_segments_rpc(const PartApp &part_app, const AnnotationList &annolistClus, 
		       int firstidx, int lastidx, int pidx, int tidx, float thresh)
{
  
  QString qsSegEndPointsDir;
  
  cout << "eval_segments_rpc()" << endl;
  cout << "pidx: " << pidx << endl;
  cout << "tidx: " << tidx << endl;
  
   /** load the configuration used for evaluation */
  PartConfig part_conf_eval = part_app.m_part_conf;
  assert(pidx < part_app.m_part_conf.part_size());
  assert(tidx < part_detect::getNumPartTypes(part_app.m_window_param, pidx));
  int pidx_window_param = part_detect::getPartById(part_app.m_window_param, pidx, tidx);

  AnnotationList annolistFull = part_app.m_train_annolist;
  AnnotationList annolistCur;
  
  bool bUsePCP = false;
  //float thresh = 0.5;
  bool bEvalTrainImgOnly = true;
  
  if (bEvalTrainImgOnly)
    annolistCur = annolistClus;
  else
    annolistCur = annolistFull;
  
  int nHyp = annolistFull.size()*10;

  
  QString qsRPCdir = (part_app.m_exp_param.log_dir() + "/" + 
		      part_app.m_exp_param.log_subdir() + "/rpc").c_str();
  
  if (!filesys::check_dir(qsRPCdir)) {
    cout << "creating " << qsRPCdir.toStdString() << endl;
    assert(filesys::create_dir(qsRPCdir));
  }
  
  int scaleidx = 0;
  double scale = scale_from_index(part_app.m_exp_param, scaleidx);
  
  vector<object_detect::PartHyp> best_part_hyp_list(nHyp);

  vector<int> imgidxs;
      
  for (int imgidx2 = 0; imgidx2 < annolistCur.size(); ++imgidx2) {      
    QString qsFilenameCur = annolistCur[imgidx2].imageName().c_str();  

    QString qsFilenameFull = "";
    int isSame = -100;

    for (int imgidx = 0; imgidx <= annolistFull.size(); ++imgidx) {        
      qsFilenameFull = annolistFull[imgidx].imageName().c_str();
      isSame = qsFilenameCur.compare(qsFilenameFull);
      if (isSame == 0){
	imgidxs.push_back(imgidx);
	break;
      }
    }
  }
  
  assert(imgidxs.size() == annolistCur.size());
  cout << "bEvalTrainImgOnly: " << bEvalTrainImgOnly << endl;
  cout << "Found img: " << imgidxs.size() << endl;
  
  for (int i = 0; i < imgidxs.size(); ++i) {      
    int imgidx = imgidxs[i]; 

    vector<object_detect::PartHyp> part_hyp_list(nHyp);
        
    get_scores(part_app, imgidx, scaleidx, pidx_window_param, part_hyp_list);
    
    if (imgidx > 0){
      vector<object_detect::PartHyp> tmp_part_hyp_list(2*nHyp);
      for(uint hypidx = 0; hypidx < nHyp; hypidx++)
	tmp_part_hyp_list[hypidx] = best_part_hyp_list[hypidx];
      for(uint hypidx = 0; hypidx < nHyp; hypidx++)
	tmp_part_hyp_list[nHyp + hypidx] = part_hyp_list[hypidx];
      
      sort(tmp_part_hyp_list.begin(), tmp_part_hyp_list.end(), compare_hyp);
	
      for(uint hypidx = 0; hypidx < nHyp; hypidx++)
	best_part_hyp_list[hypidx] = tmp_part_hyp_list[hypidx];
    }
    else
      best_part_hyp_list = part_hyp_list;
  }// images
  
  cout << "bUsePCP: " << bUsePCP << endl;
  cout << "nHyp: " << nHyp << endl;

  int seg_total = 0, num_correct = 0, num_correct_pcp = 0;
  vector<vector<float> > detectionList(nHyp, vector<float>(2));
  
  AnnotationList annolist_det, annolist_gt;
  sort(best_part_hyp_list.begin(), best_part_hyp_list.end(), compare_imgidx);
  int imgidx_prev = -1, idxanno = -1;
  
  for(uint hypidx = 0; hypidx < nHyp; hypidx++){
    PartBBox bbox;
    best_part_hyp_list[hypidx].getPartBBox(part_app.m_window_param.part(pidx_window_param), bbox);
    double scale = best_part_hyp_list[hypidx].m_scale;
    double score = best_part_hyp_list[hypidx].m_score;
    bbox.min_proj_y += scale * part_conf_eval.part(pidx).ext_y_neg();
    bbox.max_proj_y -= scale * part_conf_eval.part(pidx).ext_y_pos();
    bbox.min_proj_x += scale * part_conf_eval.part(pidx).ext_x_neg();
    bbox.max_proj_x -= scale * part_conf_eval.part(pidx).ext_x_pos();
    
    PartBBox bbox_rot = bbox;

    double_vector ul(2), lr(2), ll(2), ur(2);
    ul(0) = bbox_rot.min_proj_x + bbox_rot.part_pos(0);
    ul(1) = bbox_rot.min_proj_y + bbox_rot.part_pos(1);
    lr(0) = bbox_rot.max_proj_x + bbox_rot.part_pos(0);
    lr(1) = bbox_rot.max_proj_y + bbox_rot.part_pos(1);
    ll(0) = bbox_rot.min_proj_x + bbox_rot.part_pos(0);
    ll(1) = bbox_rot.max_proj_y + bbox_rot.part_pos(1);
    ur(0) = bbox_rot.max_proj_x + bbox_rot.part_pos(0);
    ur(1) = bbox_rot.min_proj_y + bbox_rot.part_pos(1);
    vector<double_vector> bbox_corners;
    bbox_corners.push_back(ul);
    bbox_corners.push_back(lr);
    bbox_corners.push_back(ll);
    bbox_corners.push_back(ur);
    
    update_bbox_min_max_proj(bbox_rot, bbox_corners);

    bbox_rot.min_proj_x -= scale * part_conf_eval.part(pidx).ext_x_neg();
    bbox_rot.max_proj_x += scale * part_conf_eval.part(pidx).ext_x_pos();

    int imgidx = best_part_hyp_list[hypidx].m_imgidx;
    int rectidx = 0;
    cout << "imgidx: " << imgidx << endl;
    assert(annolistFull[imgidx].size() > 0);
    if (annolistFull[imgidx][rectidx].m_vAnnoPoints.size() > 0){
      PartBBox gt_bbox;
      get_part_bbox(annolistFull[imgidx][rectidx], part_conf_eval.part(pidx), gt_bbox, scale);
      gt_bbox.min_proj_y += scale * part_conf_eval.part(pidx).ext_y_neg();
      gt_bbox.max_proj_y -= scale * part_conf_eval.part(pidx).ext_y_pos();

      bool match_x_axis = false;
      bool match;
      if (bUsePCP)
	match = is_gt_match(gt_bbox, bbox, match_x_axis, thresh);
      else
	match = is_gt_bbox_match(gt_bbox, bbox_rot, thresh);
      
      detectionList[hypidx][0] = score;
      detectionList[hypidx][1] = (float)match;
      
      float x1_det = bbox_rot.part_pos(0) + bbox_rot.min_proj_x;
      float x2_det = bbox_rot.part_pos(0) + bbox_rot.max_proj_x;
      float y1_det = bbox_rot.part_pos(1) + bbox_rot.min_proj_y;
      float y2_det = bbox_rot.part_pos(1) + bbox_rot.max_proj_y;
      
      float x1_gt = gt_bbox.part_pos(0) + gt_bbox.min_proj_x;
      float x2_gt = gt_bbox.part_pos(0) + gt_bbox.max_proj_x;
      float y1_gt = gt_bbox.part_pos(1) + gt_bbox.min_proj_y;
      float y2_gt = gt_bbox.part_pos(1) + gt_bbox.max_proj_y;
      
      if (imgidx != imgidx_prev){
	idxanno++;
	std::string sName = annolistFull[imgidx].imageName();
	
	Annotation anno_gt(sName);
	Annotation anno_det(sName);
	
	annolist_det.addAnnotation(anno_det);
	annolist_gt.addAnnotation(anno_gt);
	
	AnnoRect rect_gt(x1_gt, y1_gt, x2_gt, y2_gt);
	annolist_gt[idxanno].addAnnoRect(rect_gt);
	
	imgidx_prev = imgidx;
      }
      
      AnnoRect rect_det(x1_det, y1_det, x2_det, y2_det, score);
      annolist_det[idxanno].addAnnoRect(rect_det);
      
      ++seg_total;
      num_correct += (int)match;
    }
  }// hyp
  
  QString qsSuff = (bUsePCP ? "-pcp" : "-voc");
  QString qsRPCfilenameDet = qsRPCdir + "/det-pidx-" + padZeros(QString::number(pidx), 4) + "-tidx-" + padZeros(QString::number(tidx), 4) + qsSuff + ".idl";
  QString qsRPCfilenameGT = qsRPCdir + "/gt-pidx-" + padZeros(QString::number(pidx), 4) + "-tidx-" + padZeros(QString::number(tidx), 4) + qsSuff + ".idl";
  cout << "pidx: " << pidx << ", correct: " << num_correct  << ", total: " << seg_total << endl;
  
  annolist_det.saveIDL(qsRPCfilenameDet.toStdString());
  annolist_gt.saveIDL(qsRPCfilenameGT.toStdString());
}

void draw_gt_circles(QPainter &painter, PartBBox &bbox)
{
  double seg_len;
  double_vector endpoint_top;
  double_vector endpoint_bottom;

  get_bbox_endpoints(bbox, endpoint_top, endpoint_bottom, seg_len);
  
  painter.setPen(Qt::yellow);
  painter.drawEllipse((int)(endpoint_top(0) - 0.5*seg_len), (int)(endpoint_top(1) - 0.5*seg_len), 
                      (int)seg_len, (int)seg_len);
  painter.drawEllipse((int)(endpoint_bottom(0) - 0.5*seg_len), (int)(endpoint_bottom(1) - 0.5*seg_len), 
                      (int)seg_len, (int)seg_len);

  painter.drawLine(endpoint_top(0), endpoint_top(1), endpoint_bottom(0), endpoint_bottom(1));
}

void vis_backproject(const PartApp &part_app, int firstidx, int lastidx)
{

  cout << "vis_backproject" << endl;
  cout << "processing images from " << firstidx << " to " << lastidx << endl;

  bool bEvalLoopyLoopy = false;

  /** load samples, load samples ordering */
  QString qsReoderSamplesDir;
  QString qsBackprojectDir;

  if (bEvalLoopyLoopy) {
    qsReoderSamplesDir = (part_app.m_exp_param.log_dir() + "/" + 
                          part_app.m_exp_param.log_subdir() + "/loopy_reordered_samples").c_str();

    qsBackprojectDir = (part_app.m_exp_param.log_dir() + "/" + 
                        part_app.m_exp_param.log_subdir() + "/loopy_backproject").c_str();

  }
  else {
    qsReoderSamplesDir = (part_app.m_exp_param.log_dir() + "/" + 
                          part_app.m_exp_param.log_subdir() + "/tree_reordered_samples").c_str();      

    qsBackprojectDir = (part_app.m_exp_param.log_dir() + "/" + 
                        part_app.m_exp_param.log_subdir() + "/tree_backproject").c_str();

  }

  cout << "qsReoderSamplesDir: " << qsReoderSamplesDir.toStdString() << endl;
  assert(filesys::check_dir(qsReoderSamplesDir));

  if (!filesys::check_dir(qsBackprojectDir)) 
    assert(filesys::create_dir(qsBackprojectDir));

  cout << "storing images in " << qsBackprojectDir.toStdString() << endl;
  
  /** load original annolist */
  
  AnnotationList annolist_original;
  annolist_original.load("");

  assert(annolist_original.size() == part_app.m_test_annolist.size());

  /** load samples */

  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
    QString qsReoderSamplesFile = qsReoderSamplesDir + "/loopy_reorder_imgidx" + padZeros(QString::number(imgidx), 4) + ".mat";   
    assert(filesys::check_file(qsReoderSamplesFile));

    cout << "loading samples order from " << qsReoderSamplesFile.toStdString() << endl;
    FloatGrid2 sorted_sampidx = matlab_io::mat_load_multi_array<FloatGrid2>(qsReoderSamplesFile, "sorted_sampidx");

    QString qsSamplesDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/samples").c_str();
    assert(filesys::check_dir(qsSamplesDir) && "samples dir not found");
    QString qsFilename = qsSamplesDir + "/samples" + 
      "_imgidx" + QString::number(imgidx) +
      "_scaleidx" + QString::number(0) + 
      "_o" + QString::number((int)false) + ".mat";
    assert(filesys::check_file(qsFilename));

    cout << "loading samples from " << qsFilename.toStdString() << endl;
    FloatGrid3 all_samples = matlab_io::mat_load_multi_array<FloatGrid3>(qsFilename, "all_samples");

    cout << "loaded " << all_samples.shape()[0] << " samples" << endl;
        
    int sampidx = (int)sorted_sampidx[sorted_sampidx.shape()[0] - 1][0];
    //int sampidx = 0;

    double sampidx_lik = sorted_sampidx[sorted_sampidx.shape()[0] - 1][1];
    cout << "sampidx: " << sampidx << ", sampidx_lik: " << sampidx_lik << endl;
    
    /** load the transformation */
    QString qsOriginalImg = annolist_original[imgidx].imageName().c_str();
    QString qsProjImg = part_app.m_test_annolist[imgidx].imageName().c_str();

    QString qsTransfDir, qsTmp;
    filesys::split_filename(qsProjImg, qsTransfDir, qsTmp);

    QString qsTransfFilename = qsTransfDir + "/transf" + padZeros(QString::number(imgidx + 1), 4) + ".mat";

    cout << "qsTransfFilename: " << qsTransfFilename.toStdString() << endl;
    assert(filesys::check_file(qsTransfFilename));

    FloatGrid2 _T13 = matlab_io::mat_load_multi_array<FloatGrid2>(qsTransfFilename, "T13");

    boost_math::double_matrix T13(3, 3);
    multi_array_op::array_to_matrix(_T13, T13);

    /** visualize sample in the original image */

    int nParts = part_app.m_part_conf.part_size();

    QImage img_original;
    assert(img_original.load(qsOriginalImg));
    QPainter painter(&img_original);
    painter.setRenderHints(QPainter::Antialiasing);
    painter.setPen(Qt::yellow);

    for (int pidx = 0; pidx < nParts; ++pidx) {

      int max_ridx = (int)all_samples[sampidx][pidx][0];
      int max_ix = (int)all_samples[sampidx][pidx][2];
      int max_iy = (int)all_samples[sampidx][pidx][1];  

      cout << "pidx: " << pidx << 
        ", max_ridx: " << max_ridx << 
        ", max_ix: " << max_ix << 
        ", max_iy: " << max_iy << endl;

      int scaleidx = 0;

      PartBBox bbox;
      bbox_from_pos(part_app.m_exp_param, part_app.m_window_param.part(pidx), 
                    scaleidx, max_ridx, max_ix, max_iy, 
                    bbox);

      /* map bbox to original image */
//       double_vector t(2);
//       hc::map_point2(T13, bbox.part_pos, t);
//       bbox.part_pos = t;

      hc::map_point2(T13, bbox.part_pos, bbox.part_pos);
      hc::map_vector(T13, bbox.part_x_axis, bbox.part_x_axis);
      hc::map_vector(T13, bbox.part_y_axis, bbox.part_y_axis);

      bbox.part_x_axis /= norm_2(bbox.part_x_axis);
      bbox.part_y_axis /= norm_2(bbox.part_y_axis);
      
      double scale = abs(T13(0, 0));

      bbox.min_proj_x *= scale;
      bbox.max_proj_x *= scale;

      bbox.min_proj_y *= scale;
      bbox.max_proj_y *= scale;

      draw_bbox(painter, bbox, 1, 2);
      
    }// parts 

    QString qsOutImg = qsBackprojectDir + "/imgidx_" + padZeros(QString::number(imgidx), 4) + ".png";
    cout << "saving " << qsOutImg.toStdString() << endl;

    assert(img_original.save(qsOutImg));
  }// images
}

/** 
    similar to eval_sgmeents, only run evaluation for each region of interest 
*/

void eval_segments_roi(const PartApp &part_app, int firstidx, int lastidx, QString qsHypDirName)
{
  cout << "eval_segments_roi" << endl;

  int seg_total = 0;
  int seg_correct = 0;

  //int nParts = part_app.m_part_conf.part_size();
  
  /** load the configuration used for evaluation */
  PartConfig part_conf_eval;
    
  if (part_app.m_exp_param.has_part_conf_eval()) {
    assert(part_app.m_qsExpParam.isEmpty() == false);
    QString qsPartConfEval = complete_relative_path(QString::fromStdString(part_app.m_exp_param.part_conf_eval()), part_app.m_qsExpParam);

    parse_message_from_text_file(qsPartConfEval, part_conf_eval);
  }
  else {
    part_conf_eval = part_app.m_part_conf;
  }

  vector<int> per_segment_correct(part_conf_eval.part_size(), 0);

  AnnotationList roi_annolist(part_app.m_exp_param.roi_annolist());
  assert(roi_annolist.size() == part_app.m_test_annolist.size());

  //QString qsHypDirName = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/part_marginals_roi").c_str();

  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {      

    for (int roi_idx = 0; roi_idx < (int)roi_annolist[imgidx].size(); ++roi_idx) {
      int num_correct_cur_imgidx = 0;

      vector<PartBBox> eval_bbox;
      
      int scaleidx = -1;
      vis_eval_helper(part_app, imgidx, scaleidx, EVAL_TYPE_ROI, part_conf_eval, eval_bbox, qsHypDirName, roi_idx);

      assert((int)eval_bbox.size() == part_conf_eval.part_size());

      for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) {
	PartBBox bbox = eval_bbox[pidx];

	/**
	   MA: how to evaluate part detections when there are multiple people in the image?

	   temporary solution -> check all gt rectangles
	*/

	bool match_x_axis = false;
	bool match = false;

	for (int rectidx = 0; rectidx < (int)part_app.m_test_annolist[imgidx].size(); ++rectidx)  {
	  if (part_app.m_test_annolist[imgidx][rectidx].m_vAnnoPoints.size() > 0) {

	    /** 
		this should not matter since endpoints are provided manually and are independent of this scale 

		scale only affects size of the ground-truth bounding box itself
	    */

	    int gt_scaleidx = 0;
	    double gt_scale = scale_from_index(part_app.m_exp_param, gt_scaleidx);

	    PartBBox gt_bbox;
	    get_part_bbox(part_app.m_test_annolist[imgidx][rectidx], part_conf_eval.part(pidx), gt_bbox, gt_scale);

	    match = is_gt_match(gt_bbox, bbox, match_x_axis);

	    /** check until the first match */
	    if (match) {
	      ++seg_correct;
	      per_segment_correct[pidx]++;
	      
	      break;
	    }


	  }
	}

	++seg_total;
	cout << "pidx: " << pidx << ", match: " << (int)match  << endl;

	num_correct_cur_imgidx += (int)match;
      }// parts

      cerr << imgidx << " " << num_correct_cur_imgidx << endl;
    }// roi
    
  }// images

  cout << "seg_correct: " << seg_correct << endl;
  cout << "seg_total: " << seg_total << endl;

  if (seg_total != 0) {
    cout << "ratio: " << seg_correct / (double)seg_total << endl;

    int per_segment_total = seg_total / part_conf_eval.part_size();

    cout << endl;
    for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) {
      cout << "part: " << pidx << 
        ", correct: " << per_segment_correct[pidx] << 
        ", total: " << per_segment_total << 
        ", ratio: " << per_segment_correct[pidx]/(double)per_segment_total << endl;
    }
  }

}

void vis_segments_roi(const PartApp &part_app, int firstidx, int lastidx, QString qsHypDirName)
{

  /** load the configuration used for evaluation */
  PartConfig part_conf_eval;
    
  if (part_app.m_exp_param.has_part_conf_eval()) {
    assert(part_app.m_qsExpParam.isEmpty() == false);
    QString qsPartConfEval = complete_relative_path(QString::fromStdString(part_app.m_exp_param.part_conf_eval()), part_app.m_qsExpParam);

    parse_message_from_text_file(qsPartConfEval, part_conf_eval);
  }
  else {
    part_conf_eval = part_app.m_part_conf;
  }

  QString qsSegEvalImagesDir;

  qsSegEvalImagesDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
			"/part_marginals_roi/seg_vis_images").c_str();

  if (!filesys::check_dir(qsSegEvalImagesDir))
    filesys::create_dir(qsSegEvalImagesDir);

  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {      

    QImage _img;
    assert(_img.load(part_app.m_test_annolist[imgidx].imageName().c_str()));
    QImage img = _img.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&img);
    painter.setRenderHints(QPainter::Antialiasing);

    AnnotationList roi_annolist(part_app.m_exp_param.roi_annolist());
    assert(roi_annolist.size() == part_app.m_test_annolist.size());

    for (int roi_idx = 0; roi_idx < (int)roi_annolist[imgidx].size(); ++roi_idx) {
      vector<PartBBox> eval_bbox;

      int scaleidx = -1;
      vis_eval_helper(part_app, imgidx, scaleidx, EVAL_TYPE_ROI, part_conf_eval, eval_bbox, qsHypDirName, roi_idx);

      assert((int)eval_bbox.size() == part_conf_eval.part_size());

      int coloridx = 1;
      int pen_width = 2;

      for (int pidx = 0; pidx < part_conf_eval.part_size(); ++pidx) 
	draw_bbox(painter, eval_bbox[pidx], coloridx, pen_width);
    }

    QString qsFilename2 = qsSegEvalImagesDir + "/img_" + padZeros(QString::number(imgidx), 3) + ".png";
    cout << "saving " << qsFilename2.toStdString() << endl;
    assert(img.save(qsFilename2));

  }// images
}
