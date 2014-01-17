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

#ifndef _BOOST_MATH_DEF_H_
#define _BOOST_MATH_DEF_H_

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

const double LOG_ZERO = -1e6;
const double NEG_LOG_ZERO = -LOG_ZERO;

namespace ublas = boost::numeric::ublas;

inline double deg_from_rad(double r) {return r/M_PI*180.0;}
inline double rad_from_deg(double d) {return d/180.0*M_PI;}

namespace boost_math 
{
  typedef boost::numeric::ublas::vector<float> float_vector;

  typedef boost::numeric::ublas::vector<int> int_vector;
  typedef boost::numeric::ublas::zero_vector<int> zero_int_vector;
  
  typedef boost::numeric::ublas::vector<double> double_vector;

  typedef boost::numeric::ublas::zero_vector<double> zero_double_vector;
  typedef boost::numeric::ublas::scalar_vector<double> scalar_double_vector;
  typedef zero_double_vector double_zero_vector;
  typedef scalar_double_vector double_scalar_vector;

  typedef boost::numeric::ublas::matrix<int> int_matrix;
  typedef boost::numeric::ublas::matrix<int> zero_int_matrix;

  typedef boost::numeric::ublas::matrix<double> double_matrix;

  typedef boost::numeric::ublas::zero_matrix<double> zero_double_matrix;
  typedef boost::numeric::ublas::scalar_matrix<double> scalar_double_matrix;
  typedef boost::numeric::ublas::identity_matrix<double> identity_double_matrix;
  typedef boost::numeric::ublas::zero_matrix<double> double_zero_matrix;
  typedef boost::numeric::ublas::scalar_matrix<double> double_scalar_matrix;
  typedef boost::numeric::ublas::identity_matrix<double> double_identity_matrix;

  void print_vector(const boost_math::double_vector &v);
  void print_matrix(const boost_math::double_matrix &M);

  void get_max(const boost_math::double_vector &v, double &maxval, int &maxidx);
  void get_min(const boost_math::double_vector &v, double &minval, int &minidx);
  void comp_exp(boost_math::double_vector &v);

  void eig2d(const double_matrix &M, double_matrix &V, double_matrix &E);

  void get_gaussian_filter(boost_math::double_vector &f, double sigma, bool bNormalize);

}// namespace 

#endif
