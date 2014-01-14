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

#include <libMisc/misc.hpp>

#include "boost_math.h"
#include "boost_math.hpp"

namespace boost_math 
{

  void print_vector(const double_vector &v) {
    for (uint i = 0; i < v.size(); ++i)
      cout << v[i] << " ";
    cout << std::endl;
  }

  void print_matrix(const double_matrix &M) {
    for (uint i = 0; i < M.size1(); ++i) {
      for (uint j = 0; j < M.size2(); ++j) {
        cout << M(i, j) << " ";
      }
      cout << endl;
    }
  }

  void eig2d(const double_matrix &M, double_matrix &V, double_matrix &E)
  {
    assert(M.size1() == 2 && M.size2() == 2 && 
           V.size1() == 2 && V.size2() == 2 && 
           E.size1() == 2 && E.size2() == 2);

    typedef double_matrix::value_type value_type;
    value_type m11 = M(0, 0);
    value_type m12 = M(0, 1);
    value_type m21 = M(1, 0);
    value_type m22 = M(1, 1);
    assert(m21 == m12);

    /* eigenvalues */
    value_type e1, e2;
    
    /* eigenvector of smallest eigenvalue */
    value_type v11, v21;

    if (m12 != 0) {
      value_type sqrtD = sqrt((m11 - m22)*(m11 - m22) + 4*m12*m12);
      /* smallest eigenvaue */
      e1 = 0.5*(m11 + m22 - sqrtD);

      /* largest eigenvalue */
      e2 = 0.5*(m11 + m22 + sqrtD); 

      v11 = 0.5*(m11 - m22 - sqrtD)/m12;
      v21 = 1;
    }
    else {
      if (m11 < m22) {
        e1 = m11;
        e2 = m22;
        v11 = 1;
        v21 = 0;
      }
      else {
        e1 = m22;
        e2 = m11;
        v11 = 0;
        v21 = 1;
      }
    }
      
    value_type norm_v1 = sqrt(square(v11) + square(v21));
    v11 /= norm_v1;
    v21 /= norm_v1;
    value_type v12 = -v21;
    value_type v22 = v11;

    E = zero_double_matrix(2, 2);
    E(0, 0) = e1;
    E(1, 1) = e2;

    V(0, 0) = v11;
    V(1, 0) = v21;
    V(0, 1) = v12;
    V(1, 1) = v22;
  }

  /**
     compute gaussian filter 
   */
  void get_gaussian_filter(boost_math::double_vector &f, double sigma, bool bNormalize)
  {
    int ksize = (int)floor(3*sigma + 0.5);
    f.resize(2*ksize+1);

    f[ksize] = 1.0;
    for (int i = 1; i <= ksize; ++i) {
      f[ksize+i] = exp(-i*i/(2*sigma*sigma));
      f[ksize-i] = f[ksize+i];
    }
 
    if (bNormalize) 
      f /= norm_1(f);
  }

  void get_max(const boost_math::double_vector &v, double &maxval, int &maxidx)
  {
    assert(v.size() > 0);

    maxidx = 0;
    maxval = v[0];

    for (int idx = 0; idx < (int)v.size(); ++idx) {
      if (maxval < v[idx]) {
        maxval = v[idx];
        maxidx = idx;
      }
    }

  }


  void get_min(const boost_math::double_vector &v, double &minval, int &minidx)
  {
    assert(v.size() > 0);

    minidx = 0;
    minval = v[0];

    for (int idx = 0; idx < (int)v.size(); ++idx) {
      if (minval > v[idx]) {
        minval = v[idx];
        minidx = idx;
      }
    }
    
  }

  void comp_exp(boost_math::double_vector &v)
  {
    for (size_t idx = 0; idx < v.size(); ++idx)
      v[idx] = exp(v[idx]);
  }


}// namespace 
