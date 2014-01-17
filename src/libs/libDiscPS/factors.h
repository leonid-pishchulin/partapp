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

#ifndef _DAI_FACTORS_H_
#define _DAI_FACTORS_H_

#include <ostream>
#include <vector>

#include <QString>

#include <libMultiArray/multi_array_def.h>

#include <libKMA2/kmaimagecontent.h>

#include <libPictStruct/objectdetect.h>

#include <libPartDetect/partdef.h>

#include <libBoostMath/boost_math.h>

namespace disc_ps {

  struct Factor1d {
    Factor1d() : varidx(-1) {}

    int varidx;
    
    FloatGrid1 fval;

    bool save_factor(QString qsFilename);
    bool load_factor(QString qsFilename);

    void dai_print(std::ostream &fg_out);
  };

  struct Factor2d {
    Factor2d() : varidx1(-1), varidx2(-1) {}
    int varidx1;
    int varidx2;

    FloatGrid2 fval;

    bool save_factor(QString qsFilename);
    bool load_factor(QString qsFilename);

    void dai_print(std::ostream &fg_out);
  };

  /** helper functions */
  double eval_joint_factor(const object_detect::Joint &joint, const PartBBox& bbox_parent, const PartBBox &bbox_child, const double parent_scale, const double child_scale);
  double eval_joint_factor(const object_detect::Joint &joint, const PartBBox& bbox_parent, const PartBBox &bbox_child, 
			   const double parent_scale, const double child_scale, 
			   boost_math::double_vector &joint_pos_offset, double &joint_rot_offset, const double rot_step_size = -1.0);
  double eval_joint_factor_debug(const object_detect::Joint &joint, const PartBBox& bbox_parent, const PartBBox &bbox_child, 
				 const double parent_scale, const double child_scale, 
				 boost_math::double_vector &joint_pos_offset, double &joint_rot_offset, const double rot_step_size = -1.0);
  double intersection_area(const PartBBox &box1, const PartBBox &box2);

  void dai_print_stream(std::ostream &fg_out, std::vector<Factor1d> &factors1d, std::vector<Factor2d> &factors2d);

  /** evaluation of unary and pairwise factors */

/*   void compute_boosting_score_factor(const PartApp &part_app, int pidx, int scaleidx,  */
/*                                      kma::ImageContent *kmaimg,  */
/*                                      const std::vector<PartBBox> &part_samples,  */
/*                                      Factor1d &factor); */

  void compute_boosting_score_factor(const PartApp &part_app, int pidx, kma::ImageContent *kmaimg,
                                     const std::vector<PartBBox> &part_samples, const std::vector<double> &vect_scales,
                                     Factor1d &factor);

  void compute_spatial_factor(const PartApp &part_app, const object_detect::Joint &joint, 
                              const std::vector<PartBBox> &child_hyp, const std::vector<PartBBox> &parent_hyp, 
			      const std::vector<double> &child_vect_scales, const std::vector<double> &parent_vect_scales, 
                              Factor2d &factor);

  void compute_repulsive_factor(const std::vector<PartBBox> &child_hyp, const std::vector<PartBBox> &parent_hyp, 
                                Factor2d &factor, double min_relative_area, double alpha);

  void compute_attractive_factor(const std::vector<PartBBox> &child_hyp, const std::vector<PartBBox> &parent_hyp,
				 const std::vector<double> &child_vect_scales, const std::vector<double> &parent_vect_scales, 
				 Factor2d &factor, const boost_math::double_vector &mu, const boost_math::double_vector &sigma);
    

}

#endif 
