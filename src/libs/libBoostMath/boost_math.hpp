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

#ifndef _BOOST_MATH_HPP_
#define _BOOST_MATH_HPP_

#include <iostream>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <libBoostMath/boost_math.h>

namespace boost_math 
{
  using std::cout;
  using std::endl;
  using std::cerr;

  inline int round(double val) {
    return (int)floor(val + 0.5);
  }

  inline double_matrix get_rotation_matrix(double radians)
  {
    if (radians < -2*M_PI || radians > 2*M_PI) 
      cerr << "get_rotation_matrix: " << radians << ", radians?" << endl;

    double_matrix R(2,2);
    R(0, 0) = cos(radians);
    R(1, 0) = sin(radians);
    R(0, 1) = -R(1, 0);
    R(1, 1) = R(0, 0);
    return R;
  }

  /** 
      General matrix determinant. 
      It uses lu_factorize in uBLAS.

      note: copied from 
      http://lists.boost.org/MailArchives/ublas/2005/12/0916.php
  */
  template<class M>
  double det(M const& m) {
    /*   JFR_PRECOND(m.size1() == m.size2(), */
    /* 	      "ublasExtra::lu_det: matrix must be square"); */

    assert(m.size1() == m.size2());

    if (m.size1() == 2) {
      return m(0, 0)*m(1, 1) - m(1, 0)*m(0, 1);
    }
    else {
      // create a working copy of the input
      //mat mLu(m);
      M mLu(m);
      using namespace boost::numeric::ublas;
      permutation_matrix<std::size_t> pivots(m.size1());

      lu_factorize(mLu, pivots);

      double det = 1.0;
   
      for (std::size_t i=0; i < pivots.size(); ++i) {
        if (pivots(i) != i)
          det *= -1.0;
        det *= mLu(i,i);
      }
      return det;
    }
  }

  /**  
       Matrix inversion routine.
       Uses lu_factorize and lu_substitute in uBLAS to invert a matrix 
       
       note: this is copied from 
       http://www.crystalclearsoftware.com/cgi-bin/boost_wiki/wiki.pl?LU_Matrix_Inversion
       http://lists.boost.org/MailArchives/ublas/2005/12/0916.php

  */
  template <class T>
  bool inv(const ublas::matrix<T>& input, ublas::matrix<T>& inverse) {

    using namespace boost::numeric::ublas;
    assert(input.size1() == input.size2());

    if (input.size1() == 2) {
      double D = det(input);
      inverse.resize(2, 2);
      inverse(0, 0) = input(1, 1)/D;
      inverse(0, 1) = -input(0, 1)/D;
      inverse(1, 0) = -input(1, 0)/D;
      inverse(1, 1) = input(0, 0)/D;
    }
    else {
      typedef permutation_matrix<std::size_t> pmatrix;

      // create a working copy of the input
      matrix<T> A(input);
      // create a permutation matrix for the LU-factorization
      pmatrix pm(A.size1());

      // perform LU-factorization
      int res = lu_factorize(A,pm);
      if( res != 0 ) return false;

      // create identity matrix of "inverse"
      inverse.assign(ublas::identity_matrix<T>(A.size1()));

      // backsubstitute to get the inverse
      lu_substitute(A, pm, inverse);
    }

    return true;
  }
  
}// namespace 

#endif
