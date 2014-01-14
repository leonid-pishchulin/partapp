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

#include <algorithm>
#include <iostream>
#include <cmath>

#include <libBoostMath/boost_math.hpp>

#include <libBoostMath/homogeneous_coord.h>

using std::cout;
using std::endl;

using boost_math::double_matrix;
using boost_math::identity_double_matrix;
using boost_math::zero_double_matrix;
using boost_math::double_vector;

using namespace boost::numeric::ublas;

/* some basic operations with homogeneous coordinates + auxilliary functions useful for image transformations */
namespace hc {

  double_matrix get_homogeneous_matrix(const double_matrix &R, double dx, double dy)
  {
    double_matrix T = zero_double_matrix(3, 3) ;
    assert(R.size1() == 2 && R.size2() == 2);
    subrange(T, 0, 2, 0, 2) = R;
    T(0, 2) = dx;
    T(1, 2) = dy;
    T(2, 2) = 1;
    return T;
  }

  double_matrix inverse(const double_matrix &T)
  {
    assert(T.size1() == 3 && T.size2() == 3);
    assert(T(2, 0) == 0 && T(2, 1) == 0 && T(2, 2) == 1);

    double_matrix invT(3, 3);
    double D = T(0, 0)*T(1, 1) - T(1, 0)*T(0, 1);
    
    //subrange(invT, 0, 2, 0, 2) = trans(subrange(T, 0, 2, 0, 2));
    invT(0, 0) = T(1, 1)/D;
    invT(0, 1) = -T(0, 1)/D;
    invT(1, 0) = -T(1, 0)/D;
    invT(1, 1) = T(0, 0)/D;

    double_matrix t = -prod(subrange(invT, 0, 2, 0, 2), subrange(T, 0, 2, 2, 3));
    subrange(invT, 0, 2, 2, 3) = t;
    invT(2, 0) = 0;
    invT(2, 1) = 0;
    invT(2, 2) = 1;
    return invT;
  }

  double_matrix get_scaling_matrix(double scale)
  {
    double_matrix R(3, 3);
    R = identity_double_matrix(3,3);
    R(0,0) = scale;
    R(1,1) = scale;
    return R;
  }

  /* alpha is angle in radians */
  double_matrix get_rotation_matrix(double rad)
  {
    double_matrix R(3, 3);
    R = zero_double_matrix(3,3);
    double ca = cos(rad);
    double sa = sin(rad);
    R(0, 0) = ca;
    R(0, 1) = -sa;
    R(1, 0) = sa;
    R(1, 1) = ca;
    R(2, 2) = 1;
    return R;
  }

  double_matrix get_translation_matrix(double dx, double dy)
  {
    double_matrix T(3, 3);
    T = identity_double_matrix(3,3);

    T(0, 2) = dx;
    T(1, 2) = dy;
    return T;
  }

  double_vector get_vector(double v1, double v2)
  {
    double_vector v(3);
    v(0) = v1;
    v(1) = v2;
    v(2) = 0;
    return v;
  }

  double_vector get_point(double p1, double p2)
  {
    double_vector p(3);
    p(0) = p1;
    p(1) = p2;
    p(2) = 1;
    return p;
  }

  double norm(const double_vector &v)
  {
    assert(v.size() == 3);
    return sqrt(v(1)*v(1) + v(2)*v(2));
  }

  void get_transformed_size(const double_matrix &T21, 
                            int width, int height, 
                            int &transformed_width, int &transformed_height) {
    double minx, miny, maxx, maxy;
    get_transformed_bbox(T21, width, height, 
                         minx, miny, maxx, maxy);

    transformed_width = (int)ceil(maxx - minx);
    transformed_height = (int)ceil(maxy - miny);
  }

  void get_transformed_bbox(const double_matrix &T21, 
			    int in_width, int in_height, 
			    double &minx, double  &miny, double &maxx, double &maxy) {
    double_matrix corners(4, 3);
    corners = zero_double_matrix(4, 3);

    row(corners, 0) = prod(T21, get_point(0, 0));
    row(corners, 1) = prod(T21, get_point(in_width-1, 0));
    row(corners, 2) = prod(T21, get_point(0, in_height-1));
    row(corners, 3) = prod(T21, get_point(in_width-1, in_height-1));
  //hc::print_matrix(T21);

    minx = min(column(corners, 0));
    maxx = max(column(corners, 0));
    miny = min(column(corners, 1));
    maxy = max(column(corners, 1)); 
  }
  
}// namespace hc
