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

#ifndef _PARAPP_AUX_HPP_
#define _PARAPP_AUX_HPP_

#include <iostream>
#include <cmath>

#include <libPartApp/ExpParam.pb.h>

inline uint index_from_value(double minval, double maxval, double num_steps, double val)
{

  /*special case */
  if (minval == maxval) {
    assert(num_steps == 1);
    return 0;
  }

  //assert(val >= minval && val < maxval);
  if(!(val >= minval && val < maxval)) {
    std::cout << "val out of bounds, minval: " << minval << ", maxval: " << maxval << ", val: " << val << std::endl;
    assert(false);
  }

  double step_size = (maxval - minval)/num_steps;

  return (uint)floor((val - minval)/step_size);
}

inline double value_from_index(double minval, double maxval, double num_steps, int idx)
{
  /* special case */
  if (minval == maxval) {
    assert(idx == 0);
    
    return minval;
  }

  assert(idx >= 0 && idx < num_steps);
  double step_size = (maxval - minval)/num_steps;

  return minval + step_size*(0.5 + idx);
}


inline uint index_from_scale(const ExpParam &exp_param, double scale)
{
  return index_from_value(exp_param.min_object_scale(), 
                          exp_param.max_object_scale(), 
                          exp_param.num_scale_steps(), 
                          scale);
}

inline uint index_from_scale_clip(const ExpParam &exp_param, double scale)
{
  int scaleidx = -1;
  if (scale < exp_param.min_object_scale()) {
    std::cout << "warning: scale out of bounds" << std::endl;
    scaleidx = 0;
  }
  else if (scale >= exp_param.max_object_scale()) {
    std::cout << "warning: scale out of bounds" << std::endl;
    scaleidx = exp_param.num_scale_steps() - 1;
  }
  else {
    scaleidx = index_from_scale(exp_param, scale);
  }
  return scaleidx;
}

inline double scale_from_index(const ExpParam &exp_param, int scaleidx)
{
  return value_from_index(exp_param.min_object_scale(), 
                          exp_param.max_object_scale(), 
                          exp_param.num_scale_steps(), 
                          scaleidx);
}

inline uint index_from_rot(const ExpParam &exp_param, double rotation)
{
  return index_from_value(exp_param.min_part_rotation(), 
                          exp_param.max_part_rotation(),
                          exp_param.num_rotation_steps(), 
                          rotation);
}

inline uint index_from_rot_clip(const ExpParam &exp_param, double rotation)
{
  int rotidx = -1;
  if (rotation < exp_param.min_part_rotation()) {
    std::cout << "index_from_rot_clip: rotation out of bounds, min_rotation: " << 
      exp_param.min_part_rotation() << ", value: " << rotation << std::endl;

    rotidx = 0;
  }
  else if (rotation >= exp_param.max_part_rotation()) {
    std::cout << "index_from_rot_clip: rotation out of bounds, max_rotation: " << 
      exp_param.max_part_rotation() << ", value: " << rotation << std::endl;

    rotidx = exp_param.num_rotation_steps() - 1;
  }
  else {
    rotidx = index_from_rot(exp_param, rotation);
  }
  return rotidx;
}

inline double rot_from_index(const ExpParam &exp_param, int rotidx)
{
  return value_from_index(exp_param.min_part_rotation(), 
                          exp_param.max_part_rotation(),
                          exp_param.num_rotation_steps(), 
                          rotidx);
}



#endif
