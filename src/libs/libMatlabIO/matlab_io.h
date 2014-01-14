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

#ifndef _MATLAB_IO_H
#define _MATLAB_IO_H

/* MATLAB header file */
#include <mat.h>

#include <QString>

#include <libBoostMath/boost_math.h>

namespace matlab_io {

  /**
     open .mat file, mode is "w" for writing and "r" for reading, "wz" for writing compressed data

     return false if file could not be opened
  */
  MATFile *mat_open(QString qsFilename, const char *mode);

  void mat_close(MATFile *f);

  /**
     save/load ublas matrices and vectors
   */

  bool mat_save_double_vector(QString qsFilename, QString qsVarName, const boost_math::double_vector &v);
  bool mat_save_double_matrix(QString qsFilename, QString qsVarName, const boost_math::double_matrix &m);

  bool mat_save_double_vector(MATFile *f, QString qsVarName, const boost_math::double_vector &v);
  bool mat_save_double_matrix(MATFile *f, QString qsVarName, const boost_math::double_matrix &m);

  bool mat_load_double_vector(QString qsFilename, QString qsVarName, boost_math::double_vector &v); 
  bool mat_load_double_matrix(QString qsFilename, QString qsVarName, boost_math::double_matrix &m);

  /** 
      save/load scalar doubles
   */
  bool mat_save_double(QString qsFilename, QString qsVarName, double d); 
  bool mat_save_double(MATFile *f, QString qsVarName, double d); 

  bool mat_load_double(QString qsFilename, QString qsVarName, double &d); 

  bool mat_save_std_vector(MATFile *f, QString qsVarName, const std::vector<float> &v); 


}// namespace 

#endif

