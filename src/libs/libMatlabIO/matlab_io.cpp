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

#include <libMultiArray/multi_array_def.h>

#include "matlab_io.h"
#include "matlab_io.hpp"

namespace matlab_io {

  MATFile *mat_open(QString qsFilename, const char *mode)
  {
    return matOpen(qsFilename.toStdString().c_str(), mode);
  }

  void mat_close(MATFile *f)
  {
    assert(f != 0);
    matClose(f);
  }

  bool mat_save_double_vector(QString qsFilename, QString qsVarName, const boost_math::double_vector &v)
  {
    MATFile *f = mat_open(qsFilename, "wz");
    bool res = false;

    if (f != 0) {
      res = mat_save_double_vector(f, qsVarName, v);
      mat_close(f);
    }

    return res;
  }

  bool mat_save_double_matrix(QString qsFilename, QString qsVarName, const boost_math::double_matrix &m)
  {
    MATFile *f = mat_open(qsFilename, "wz");
    bool res = false;

    if (f != 0) {
      res = mat_save_double_matrix(f, qsVarName, m);
      mat_close(f);
    }

    return res;
  }

  bool mat_save_double_vector(MATFile *f, QString qsVarName, const boost_math::double_vector &v)
  {
    DoubleGrid2 grid2(boost::extents[v.size()][1]);
    for (uint i = 0; i < v.size(); ++i)
      grid2[i][0] = v(i);

    return mat_save_multi_array(f, qsVarName, grid2);
  }

  bool mat_save_double_matrix(MATFile *f, QString qsVarName, const boost_math::double_matrix &m)
  {
    DoubleGrid2 grid2(boost::extents[m.size1()][m.size2()]);
    for (uint i1 = 0; i1 < m.size1(); ++i1)
      for (uint i2 = 0; i2 < m.size2(); ++i2)
        grid2[i1][i2] = m(i1, i2);

    return mat_save_multi_array(f, qsVarName, grid2);
  }

  bool mat_load_double_vector(QString qsFilename, QString qsVarName, boost_math::double_vector &v)
  {
    DoubleGrid2 grid2 = mat_load_multi_array<DoubleGrid2>(qsFilename, qsVarName);

    if (grid2.shape()[1] == 1 && grid2.shape()[0] > 0) {
      v.resize(grid2.shape()[0]);
      for (uint i = 0; i < v.size(); ++i)
        v(i) = grid2[i][0];
      
      return true;
    }
    else if (grid2.shape()[0] == 1 && grid2.shape()[1] > 0) {
      v.resize(grid2.shape()[1]);
      for (uint i = 0; i < v.size(); ++i)
        v(i) = grid2[0][i];
      
      return true;
    }

    assert(false && "found matrix instead of vector");
    return false;
  }

  bool mat_load_double_matrix(QString qsFilename, QString qsVarName, boost_math::double_matrix &m)
  {
    DoubleGrid2 grid2 = mat_load_multi_array<DoubleGrid2>(qsFilename, qsVarName);

    if (grid2.shape()[0] > 0 && grid2.shape()[1] > 0) {

      m.resize(grid2.shape()[0], grid2.shape()[1]);

      for (uint i1 = 0; i1 < m.size1(); ++i1)
        for (uint i2 = 0; i2 < m.size2(); ++i2)
          m(i1, i2) = grid2[i1][i2];

      return true;
    }

    return false;
  }

  bool mat_save_double(QString qsFilename, QString qsVarName, double d)
  {
    MATFile *f = mat_open(qsFilename, "wz");
    bool res = false;

    if (f != 0) {
      res = mat_save_double(f, qsVarName, d);
    }
    else {
      assert(false && "could not open file");
    }

    return res;
  }

  bool mat_save_double(MATFile *f, QString qsVarName, double d)
  {
    assert(f != 0);

    DoubleGrid2 grid2(boost::extents[1][1]);
    grid2[0][0] = d;
    return mat_save_multi_array(f, qsVarName, grid2);
  }

  bool mat_load_double(QString qsFilename, QString qsVarName, double &d)
  {
    DoubleGrid2 grid2 = mat_load_multi_array<DoubleGrid2>(qsFilename, qsVarName);

    if (grid2.shape()[0] == 1 && grid2.shape()[1] == 1) {
      d = grid2[0][0];

      return true;
    }

    return false;
  }

  bool mat_save_std_vector(MATFile *f, QString qsVarName, const std::vector<float> &v)
  {
    assert(f != 0);
    FloatGrid2 grid2(boost::extents[v.size()][1]);
    for (uint i = 0; i < v.size(); ++i)
      grid2[i][0] = v[i];

    return mat_save_multi_array(f, qsVarName, grid2);    
  }

}// namespace 
