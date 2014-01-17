/** 
    This file is part of the implementation of the human pose estimation model as described in the paper:
    
    Leonid Pishchulin, Micha Andriluka, Peter Gehler and Bernt Schiele
    Strong Appearance and Expressive Spatial Models for Human Pose Estimation
    IEEE International Conference on Computer Vision and Pattern Recognition (ICCV'13), Sydney, Australia, December 2013

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  
*/

#ifndef _HOMOGENEOUS_COORD_H_
#define _HOMOGENEOUS_COORD_H_

#include <libBoostMath/boost_math.h>

//#include "boost_matrix.h"
//#include "boost_math.h"
//#include "math_helpers.h"

/* enum TransformationMethod { */
/*   TM_NEAREST = 0, */
/*   TM_BILINEAR = 1, */
/*   TM_DIRECT = 2 */
/* }; */

namespace hc {

  boost_math::double_matrix get_homogeneous_matrix(const boost_math::double_matrix &R, double dx, double dy);

  boost_math::double_matrix inverse(const boost_math::double_matrix &T);

  void eig2d(const boost_math::double_matrix &M, boost_math::double_matrix &V, boost_math::double_matrix &E);

  /* alpha is angle in radians */
  boost_math::double_matrix get_rotation_matrix(double rad);
  boost_math::double_matrix get_scaling_matrix(double scale);
  boost_math::double_matrix get_translation_matrix(double dx, double dy);

  boost_math::double_vector get_vector(double v1, double v2);
  boost_math::double_vector get_point(double p1, double p2);

  double norm(const boost_math::double_vector &v);
  //void print_vector(const boost_math::double_vector &v);
  //void print_matrix(const boost_math::double_matrix &M);

  void get_transformed_size(const boost_math::double_matrix &T21, 
                            int width, int height, 
                            int &transformed_width, int &transformed_height);

  void get_transformed_bbox(const boost_math::double_matrix &T21, 
			    int in_width, int in_height, 
			    double &minx, double  &miny, double &maxx, double &maxy);

  template <class T>
  typename T::value_type min(const T &sequence) 
  {
    return *(std::min_element(sequence.begin(), sequence.end()));
  }

  template <class T>
  typename T::value_type max(const T &sequence) 
  {
    return *(std::max_element(sequence.begin(), sequence.end()));
  }

  inline void map_point(const boost_math::double_matrix &M, double x, double y, double &out_x, double &out_y) {
    assert(M.size1() == 3 && M.size2() == 3);
//     boost_math::double_vector p = get_point(x, y);
//     boost_math::double_vector mp = prod(M, p);

//     out_x = mp(0);
//     out_y = mp(1);
    out_x = M(0, 0)*x + M(0, 1)*y + M(0, 2);
    out_y = M(1, 0)*x + M(1, 1)*y + M(1, 2);
  }

  inline void map_point2(const boost_math::double_matrix &M, 
			 const boost_math::double_vector p_in, 
			 boost_math::double_vector &p_out) {
    assert(M.size1() == 3 && M.size2() == 3);
    assert(p_in.size() == 2);

    p_out.resize(2);
    p_out(0) = M(0, 0)*p_in(0) + M(0, 1)*p_in(1) + M(0, 2);
    p_out(1) = M(1, 0)*p_in(0) + M(1, 1)*p_in(1) + M(1, 2);
  }

  inline void map_vector(const boost_math::double_matrix &M, 
			 const boost_math::double_vector p_in, 
			 boost_math::double_vector &p_out) {
    assert(M.size1() == 3 && M.size2() == 3);
    assert(p_in.size() == 2);

    p_out.resize(2);
    p_out(0) = M(0, 0)*p_in(0) + M(0, 1)*p_in(1);
    p_out(1) = M(1, 0)*p_in(0) + M(1, 1)*p_in(1);
  }

} 


#endif
