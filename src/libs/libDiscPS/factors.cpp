/** 
    This file is part of the implementation of the human pose estimation model as described in the paper:
    
    L. Pishchulin, M. Andriluka, P. Gehler and B. Schiele
    Strong Appearance and Expressive Spatial Models for Human Pose Estimation
    IEEE Conference on Computer Vision and Pattern Recognition (ICCV'13), Sydney, Australia, December 2013

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  
*/

#include <libMatlabIO/matlab_io.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libPartApp/partapp_aux.hpp>

#include <libPartDetect/partdetect.h>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include "factors.h"
#include "rect_intersection.h"

using namespace std;

namespace disc_ps {

  using boost_math::double_vector;
  using boost_math::double_matrix;
  using object_detect::Joint;

  void Factor1d::dai_print(ostream &fg_out)
  {
    /* number of variables */
    fg_out << "1" << endl;

    /* index of variables */
    fg_out << varidx << endl;

    /* number of discrete values for each variable */
    fg_out << fval.shape()[0] << endl;

    /* number of non-zeros values */
    fg_out << fval.shape()[0] << endl;

    /* non-zero values in the factor table */
    for (int idx = 0; idx < (int)fval.shape()[0]; ++idx) {
      fg_out << idx << " " << fval[idx] << endl;
    }

    fg_out << endl;      
  }

  void Factor2d::dai_print(ostream &fg_out)
  {
    /* number of variables */
    fg_out << "2" << endl;
    
    /* index of variables */
    fg_out << varidx1 << " " << varidx2 << endl;

    /* number of discrete values for each variable */
    fg_out << fval.shape()[0] << " " << fval.shape()[1] << endl;

    /* number of non-zeros values in the factor table */
    fg_out << fval.shape()[0] * fval.shape()[1] << endl;

    /* non-zero values */
    int idx = 0;

    for (int idx2 = 0; idx2 < (int)fval.shape()[1]; ++idx2) 
      for (int idx1 = 0; idx1 < (int)fval.shape()[0]; ++idx1) {
        double prob_val = fval[idx1][idx2];

        fg_out << idx << " " << prob_val << endl;
        ++idx;
      }   
      
    fg_out << endl;
  }

  void dai_print_stream(ostream &fg_out, vector<Factor1d> &factors1d, vector<Factor2d> &factors2d)
  {
    fg_out << factors1d.size() + factors2d.size() << endl << endl;

    for (uint idx = 0; idx < factors1d.size(); ++idx) {
      cout << "unary factor: " << idx << endl;
      factors1d[idx].dai_print(fg_out);
    }

    for (uint idx = 0; idx < factors2d.size(); ++idx) {
      cout << "pairwise factor: " << idx << endl;
      factors2d[idx].dai_print(fg_out);
    }
  }


  bool Factor2d::save_factor(QString qsFilename) {
    MATFile *f = matlab_io::mat_open(qsFilename, "wz");

    assert(f != 0);

    bool bres1 = matlab_io::mat_save_double(f, "varidx1", varidx1);
    bool bres2 = matlab_io::mat_save_double(f, "varidx2", varidx2);
    bool bres3 = matlab_io::mat_save_multi_array(f, "fval", fval);

    matlab_io::mat_close(f);

    return bres1 && bres2 && bres3;
  }

  bool Factor2d::load_factor(QString qsFilename) {
    bool bres = false;

    if (filesys::check_file(qsFilename)) {
      MATFile *f = matlab_io::mat_open(qsFilename, "r");
      assert(f != 0);

      double _varidx;
      bool bres1 = matlab_io::mat_load_double(qsFilename, "varidx1", _varidx);
      varidx1 = (int)_varidx;

      bool bres2 = matlab_io::mat_load_double(qsFilename, "varidx2", _varidx);
      varidx2 = (int)_varidx;

      FloatGrid2 _fval = matlab_io::mat_load_multi_array<FloatGrid2>(f, "fval");

      fval.resize(boost::extents[_fval.shape()[0]][_fval.shape()[1]]);

      for (uint idx1 = 0; idx1 < _fval.shape()[0]; ++idx1)
        for (uint idx2 = 0; idx2 < _fval.shape()[1]; ++idx2)
          fval[idx1][idx2] = _fval[idx1][idx2];

      matlab_io::mat_close(f);

      bres = bres1 && bres2;
    }

    assert(varidx1 >= 0 && varidx2 >= 0);

    return bres;
  }


  bool Factor1d::save_factor(QString qsFilename) {
    MATFile *f = matlab_io::mat_open(qsFilename, "wz");

    assert(f != 0);

    bool bres1 = matlab_io::mat_save_double(f, "varidx", varidx);
    bool bres2 = matlab_io::mat_save_multi_array(f, "fval", fval);

    matlab_io::mat_close(f);

    return bres1 && bres2;
  }

  bool Factor1d::load_factor(QString qsFilename) {
    bool bres = false;

    if (filesys::check_file(qsFilename)) {
      MATFile *f = matlab_io::mat_open(qsFilename, "r");
      assert(f != 0);

      double _varidx;
      bres = matlab_io::mat_load_double(qsFilename, "varidx", _varidx);

      /** for some reason 1d arrays are saved as 2d */
      FloatGrid2 _fval = matlab_io::mat_load_multi_array<FloatGrid2>(f, "fval");
      
      assert(_fval.shape()[1] == 1);

      fval.resize(boost::extents[_fval.shape()[0]]);
      for (uint idx = 0; idx < _fval.shape()[0]; ++idx)
        fval[idx] = _fval[idx][0];
      
      varidx = (int)_varidx;
      matlab_io::mat_close(f);
    }

    assert(varidx >= 0);
      
    return bres;
  }


  void compute_boosting_score_factor(const PartApp &part_app, int pidx, kma::ImageContent *kmaimg, 
                                     const vector<PartBBox> &part_samples, const vector<double> &vect_scales,
                                     Factor1d &factor)
  {
    assert(part_samples.size() == vect_scales.size());

    factor.varidx = pidx;

    /** add border */
    int added_border = 0;
    kma::ImageContent *kmaimg_border = add_image_border2(kmaimg, 50, added_border);

    cout << "added_border: " << added_border << endl;

    AdaBoostClassifier abc;
    part_app.loadClassifier(abc, pidx);

    int num_samples = part_samples.size();
    
    factor.fval.resize(boost::extents[num_samples]);

    for (int sampidx = 0; sampidx < num_samples; ++sampidx) {

      PartBBox bbox = part_samples[sampidx];
      /** add offset to position of bounding box */
      bbox.part_pos(0) += added_border;
      bbox.part_pos(1) += added_border;

      /** comput features */
      vector<float> all_features;
      PartBBox adjusted_rect;

//       bool bres = part_detect::compute_part_bbox_features_scale(part_app.m_abc_param, part_app.m_window_param, 
// 								bbox, kmaimg_border, pidx, 1,
// 								all_features, adjusted_rect);

//      assert(vect_scales[sampidx] == 1.0);

      bool bres = part_detect::compute_part_bbox_features_scale(part_app.m_abc_param, part_app.m_window_param, 
                                                                bbox, kmaimg_border, pidx, vect_scales[sampidx], 
                                                                all_features, adjusted_rect);

      factor.fval[sampidx] = part_app.m_abc_param.min_score();

      /** evaluate classifier */      
      if (bres) {
        float score = abc.evaluateFeaturePoint(all_features, true);

        if (score > part_app.m_abc_param.min_score())
          factor.fval[sampidx] = score;          
      }
      else {
        cout << "warning: failed to compute features" << endl;
      }
      
    }// samples

    delete kmaimg_border;
  }

  double eval_joint_factor(const Joint &joint, const PartBBox& bbox_parent, const PartBBox &bbox_child, const double parent_scale, const double child_scale){
    double_vector joint_pos_offset;
    double joint_rot_offset;
    
    return eval_joint_factor(joint, bbox_parent, bbox_child, parent_scale, child_scale, joint_pos_offset, joint_rot_offset);
  }
  
  double eval_joint_factor(const Joint &joint, const PartBBox& bbox_parent, const PartBBox &bbox_child, 
			   const double parent_scale, const double child_scale, 
			   double_vector &joint_pos_offset, double &joint_rot_offset, const double rot_step_size)
  {

    double rot_sigma = joint.rot_sigma;
    assert(rot_sigma >= 0);
    /*
    if (rot_sigma <= 0) {
      rot_sigma = 5*M_PI / 180.0;
    }
    */
    double child_rot = atan2(bbox_child.part_x_axis(1), bbox_child.part_x_axis(0));
    double parent_rot = atan2(bbox_parent.part_x_axis(1), bbox_parent.part_x_axis(0));

//     assert(child_scale == 1.0);
//     assert(parent_scale == 1.0);

    double_vector joint_pos_child = bbox_child.part_pos + 
      child_scale * bbox_child.part_x_axis * joint.offset_c(0) + 
      child_scale * bbox_child.part_y_axis * joint.offset_c(1);

    double_vector joint_pos_parent = bbox_parent.part_pos + 
      parent_scale * bbox_parent.part_x_axis * joint.offset_p(0) + 
      parent_scale * bbox_parent.part_y_axis * joint.offset_p(1);
    
    /* need to discretize the values as work on integer grid */
    joint_pos_child(0) = round(joint_pos_child(0));
    joint_pos_child(1) = round(joint_pos_child(1));
    joint_pos_parent(0) = round(joint_pos_parent(0));
    joint_pos_parent(1) = round(joint_pos_parent(1));    
    
    joint_pos_offset = joint_pos_child - joint_pos_parent;
    
    /* discretize rotation mean */
    double rot_mean = joint.rot_mean;
    if (rot_step_size > 0 && rot_step_size <= M_PI){
      rot_mean = boost_math::round(joint.rot_mean / rot_step_size) * rot_step_size;
    }
    joint_rot_offset = (child_rot - parent_rot) - rot_mean;
    
    /*
    while (joint_rot_offset < -M_PI)
      joint_rot_offset += M_PI;

    while (joint_rot_offset > M_PI)
      joint_rot_offset -= M_PI;
    */
    /*
    while (joint_rot_offset < -M_PI)
      joint_rot_offset += 2*M_PI;

    while (joint_rot_offset > M_PI)
      joint_rot_offset -= 2*M_PI;
    */
    /*
    while (joint_rot_offset < -2*M_PI)
      joint_rot_offset += 2*M_PI;

    while (joint_rot_offset > 2*M_PI)
      joint_rot_offset -= 2*M_PI;
    */
    /*
    while (joint_rot_offset < 0)
      joint_rot_offset += 2*M_PI;

    while (joint_rot_offset > 2*M_PI)
      joint_rot_offset -= 2*M_PI;
    */
    
    //assert(joint_rot_offset >= -M_PI && joint_rot_offset <= M_PI);
    // no wraparound
    
    //assert(joint_rot_offset >= -2*M_PI && joint_rot_offset <= 2*M_PI);
    
    /**
       how to estimate the vairance for parts at different scales ?
       
       MA: for now use the average scale
     */
    
    double detC1 = joint.detC;
    assert(detC1 > 0);
    
    double avg_scale = 0.5*(parent_scale + child_scale);
    //double avg_scale = std::max(parent_scale,child_scale);
    double_matrix inv_scaleC1 = joint.invC / square(avg_scale);

    //double d_pos = inner_prod(joint_pos_offset, ublas::prod<double_vector>(invC1, joint_pos_offset));
    double d_pos = inner_prod(joint_pos_offset, ublas::prod<double_vector>(inv_scaleC1, joint_pos_offset));
    double d_rot = square(joint_rot_offset) / square(rot_sigma);

    double p_joint;
    
    /** 
	MA: this is supposed to be comparable accross scales 

	the normalization constant does not depend on scales since we have implicitly rescaled positions to training scale
     */
    p_joint = exp(-0.5*(d_pos + d_rot)) / (2*M_PI*sqrt(2*M_PI) * sqrt(detC1) * rot_sigma);
    /*
    cout << "d_pos: " << d_pos << endl;  
    cout << p_joint << endl;
    cout << detC1 << endl;
    cout << rot_sigma << endl;
    */
    //p_joint = exp(-(d_pos + d_rot)); // (2*M_PI*sqrt(2*M_PI) * sqrt(detC1) * rot_sigma);
    /*
     cout << "rot_sigma: "  << rot_sigma << endl;
     cout << "joint.C:" << endl;
     boost_math::print_matrix(joint.C);
     //cout << "invC:" << endl; 
     //boost_math::print_matrix(invC1);
     cout << "d_pos: " << d_pos << endl;  
     cout << "d_rot: " << d_rot << endl; 
     cout << "p_joint: "  << p_joint << endl;
    */
    
    //assert(p_joint >= 0);
    return p_joint;
  }
  
  double eval_joint_factor_debug(const Joint &joint, const PartBBox& bbox_parent, const PartBBox &bbox_child, 
				 const double parent_scale, const double child_scale, 
				 double_vector &joint_pos_offset, double &joint_rot_offset, const double rot_step_size)
  {

    double rot_sigma = joint.rot_sigma;
    assert(rot_sigma >= 0);

    double child_rot = atan2(bbox_child.part_x_axis(1), bbox_child.part_x_axis(0));
    double parent_rot = atan2(bbox_parent.part_x_axis(1), bbox_parent.part_x_axis(0));
    cout << "child_rot = " << child_rot << endl;
    cout << "parent_rot = " << parent_rot << endl;
    cout << "rot_step_size = " << rot_step_size << endl;
    
    double_matrix Tgc = prod(hc::get_rotation_matrix(child_rot), hc::get_scaling_matrix(child_scale));
    double_vector offset_c_10 = hc::get_vector(joint.offset_c(0), joint.offset_c(1));
    double_vector offset_g_10 = prod(Tgc, offset_c_10);
    double_matrix Tjoint_c = hc::get_translation_matrix(offset_g_10(0), offset_g_10(1));
    double_matrix Tjoint2_c = prod(Tjoint_c,boost_math::identity_double_matrix(3));
    double x1_c, y1_c;
    hc::map_point(Tjoint2_c, bbox_child.part_pos(0), bbox_child.part_pos(1), x1_c, y1_c);
    
    double_matrix Tgp = prod(hc::get_rotation_matrix(parent_rot), hc::get_scaling_matrix(parent_scale));
    double_vector offset_p_01 = hc::get_vector(joint.offset_p(0), joint.offset_p(1));
    double_vector offset_g_01 = prod(Tgp, offset_p_01);
    double_matrix Tjoint_p = hc::get_translation_matrix(offset_g_01(0), offset_g_01(1));
    double_matrix Tjoint2_p = prod(Tjoint_p,boost_math::identity_double_matrix(3));
    double x1_p, y1_p;
    hc::map_point(Tjoint2_p, bbox_parent.part_pos(0), bbox_parent.part_pos(1), x1_p, y1_p);
    
//     assert(child_scale == 1.0);
//     assert(parent_scale == 1.0);

    double_vector joint_pos_child = bbox_child.part_pos + 
      child_scale * bbox_child.part_x_axis * joint.offset_c(0) + 
      child_scale * bbox_child.part_y_axis * joint.offset_c(1);

    double_vector joint_pos_parent = bbox_parent.part_pos + 
      parent_scale * bbox_parent.part_x_axis * joint.offset_p(0) + 
      parent_scale * bbox_parent.part_y_axis * joint.offset_p(1);
    
    cout << "round part positions" << endl;
    joint_pos_child(0) = round(joint_pos_child(0));
    joint_pos_child(1) = round(joint_pos_child(1));
    joint_pos_parent(0) = round(joint_pos_parent(0));
    joint_pos_parent(1) = round(joint_pos_parent(1));    
    
    joint_pos_offset = joint_pos_child - joint_pos_parent;
    
    cout << round(y1_c) << " " << round(x1_c) << endl;
    cout << joint_pos_child(1) << " " << joint_pos_child(0) << endl;
    cout << endl;
    cout << round(y1_p) << " " << round(x1_p) << endl;
    cout << joint_pos_parent(1) << " " << joint_pos_parent(0) << endl;
    
    double rot_mean = joint.rot_mean;
    if (rot_step_size > 0 && rot_step_size <= M_PI){
      cout << "round joint mean" << endl;
      rot_mean = boost_math::round(joint.rot_mean / rot_step_size) * rot_step_size;
    }
    joint_rot_offset = (child_rot - parent_rot) - rot_mean;
    /*
    while (joint_rot_offset < -M_PI)
      joint_rot_offset += M_PI;

    while (joint_rot_offset > M_PI)
      joint_rot_offset -= M_PI;
    
    assert(joint_rot_offset >= -M_PI && joint_rot_offset <= M_PI);
    */
    double detC1 = joint.detC;
    assert(detC1 > 0);

    double avg_scale = 0.5*(parent_scale + child_scale);
    //double avg_scale = std::max(parent_scale,child_scale);
    double_matrix inv_scaleC1 = joint.invC / square(avg_scale);
    
    //double d_pos = inner_prod(joint_pos_offset, ublas::prod<double_vector>(invC1, joint_pos_offset));
    double d_pos = inner_prod(joint_pos_offset, ublas::prod<double_vector>(inv_scaleC1, joint_pos_offset));
    double d_rot = square(joint_rot_offset) / square(rot_sigma);

    double p_joint;
      

    p_joint = exp(-0.5*(d_pos + d_rot)) / (2*M_PI*sqrt(2*M_PI) * sqrt(detC1) * rot_sigma);

    //assert(p_joint >= 0);
    return p_joint;
  }
      
  
  void compute_spatial_factor(const PartApp &part_app, const Joint &joint, 
                              const vector<PartBBox> &child_hyp, const vector<PartBBox> &parent_hyp, 
			      const std::vector<double> &child_vect_scales, const std::vector<double> &parent_vect_scales, 
                              Factor2d &factor)
  {
    assert(child_hyp.size() == child_vect_scales.size());
    assert(parent_hyp.size() == parent_vect_scales.size());

    factor.varidx1 = joint.child_idx;
    factor.varidx2 = joint.parent_idx;

    factor.fval.resize(boost::extents[child_hyp.size()][parent_hyp.size()]);

    double rot_step_size = (part_app.m_exp_param.max_part_rotation() - part_app.m_exp_param.min_part_rotation())/part_app.m_exp_param.num_rotation_steps();
    rot_step_size *= M_PI / 180.0;
    
    double_vector joint_pos_offset;
    double joint_rot_offset;
    
    for (uint idx1 = 0; idx1 < child_hyp.size(); ++idx1)
      for (uint idx2 = 0; idx2 < parent_hyp.size(); ++idx2) {
        factor.fval[idx1][idx2] = eval_joint_factor(joint, parent_hyp[idx2], child_hyp[idx1], parent_vect_scales[idx2], child_vect_scales[idx1], joint_pos_offset, joint_rot_offset, rot_step_size);
      }

  }

  void compute_spatial_factor2(const Joint &joint, 
                              const vector<PartBBox> &child_hyp, const vector<PartBBox> &parent_hyp, 
			      const std::vector<double> &child_vect_scales, const std::vector<double> &parent_vect_scales, 
                              Factor2d &factor)
  {
    assert(child_hyp.size() == child_vect_scales.size());
    assert(parent_hyp.size() == parent_vect_scales.size());

    factor.varidx1 = joint.child_idx;
    factor.varidx2 = joint.parent_idx;

    factor.fval.resize(boost::extents[child_hyp.size()][parent_hyp.size()]);

    for (uint idx1 = 0; idx1 < child_hyp.size(); ++idx1)
      for (uint idx2 = 0; idx2 < parent_hyp.size(); ++idx2) {
        factor.fval[idx1][idx2] = eval_joint_factor(joint, parent_hyp[idx2], child_hyp[idx1], parent_vect_scales[idx2], child_vect_scales[idx1]);
      }

  }

  void get_part_polygon2(const PartBBox &part_bbox, QVector<QPointF> &polygon)
  {
    polygon.clear();

    boost_math::double_vector t(2);
    t = part_bbox.part_x_axis*part_bbox.min_proj_x + part_bbox.part_y_axis*part_bbox.min_proj_y + part_bbox.part_pos;
    polygon.push_back(QPointF(t(0), t(1)));
    
    t = part_bbox.part_x_axis*part_bbox.max_proj_x + part_bbox.part_y_axis*part_bbox.min_proj_y + part_bbox.part_pos;
    polygon.push_back(QPointF(t(0), t(1)));

    t = part_bbox.part_x_axis*part_bbox.max_proj_x + part_bbox.part_y_axis*part_bbox.max_proj_y + part_bbox.part_pos;
    polygon.push_back(QPointF(t(0), t(1)));

    t = part_bbox.part_x_axis*part_bbox.min_proj_x + part_bbox.part_y_axis*part_bbox.max_proj_y + part_bbox.part_pos;
    polygon.push_back(QPointF(t(0), t(1)));
  }

  double intersection_area(const PartBBox &box1, const PartBBox &box2)
  {
    QVector<QPointF> rect1;
    QVector<QPointF> rect2;

    get_part_polygon2(box1, rect1);
    get_part_polygon2(box2, rect2);

    return intersection_area(rect1, rect2);
  }

  void compute_repulsive_factor(const vector<PartBBox> &child_hyp, const vector<PartBBox> &parent_hyp, 
                                Factor2d &factor, double min_relative_area, double alpha)
  {
    factor.fval.resize(boost::extents[child_hyp.size()][parent_hyp.size()]);

    for (uint idx1 = 0; idx1 < child_hyp.size(); ++idx1)
      for (uint idx2 = 0; idx2 < parent_hyp.size(); ++idx2) {
        
        double area1 = (child_hyp[idx1].max_proj_x - child_hyp[idx1].min_proj_x) * 
          (child_hyp[idx1].max_proj_y - child_hyp[idx1].min_proj_y);

//         double area2 = (parent_hyp[idx2].max_proj_x - parent_hyp[idx2].min_proj_x) * 
//           (parent_hyp[idx2].max_proj_y - parent_hyp[idx2].min_proj_y);

        //assert(fabs(area1 - area2) < 1e-3);

        double int_area = intersection_area(child_hyp[idx1], parent_hyp[idx2]);
        double rel_int_area = int_area / area1;

        //if (rel_int_area > 0.2) {
        if (rel_int_area > min_relative_area) {
          factor.fval[idx1][idx2] = exp(-alpha); // works well with boosting scores
          //factor.fval[idx1][idx2] = exp(-1.2); // works well with boosting scores
          //factor.fval[idx1][idx2] = exp(-5); // does not work :(
        }
        else {
          factor.fval[idx1][idx2] = 1;
        }
      }

  }

  void compute_attractive_factor(const std::vector<PartBBox> &child_hyp, const std::vector<PartBBox> &parent_hyp,
				 const std::vector<double> &child_vect_scales, const std::vector<double> &parent_vect_scales,
				 Factor2d &factor, const boost_math::double_vector &mu, const boost_math::double_vector &sigma)
  {

    factor.fval.resize(boost::extents[child_hyp.size()][parent_hyp.size()]);

    double_matrix invC(2, 2);

    for (uint idx1 = 0; idx1 < child_hyp.size(); ++idx1)
      for (uint idx2 = 0; idx2 < parent_hyp.size(); ++idx2) {

	double_vector d = child_hyp[idx1].part_pos - parent_hyp[idx2].part_pos - mu;
	
	double d1 = square(d(0)) / square(sigma(0));
	double d2 = square(d(1)) / square(sigma(1));

	factor.fval[idx1][idx2] = exp(-0.5*(d1 + d2));

      }

  }


}
