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

#ifndef _MISC_HPP_
#define _MISC_HPP_

#include <QString>

typedef unsigned int uint;

/**
   various helper functions 
*/

template <class T>
inline T square(const T &x) { return x*x; };

/* test that the value is in [min, max) */
template<class T>
inline bool check_bounds(T val, T min, T max) 
{
  return (val >= min && val < max);
}

template <class T> 
inline void check_bounds_and_update(T &val, const T &min, const T &max)
{
  assert(min <= max - 1);

  if (val < min)
    val = min;

  if (val >= max)
    val = max - 1;
}

inline QString padZeros(QString qsStr, int npad)
{
  QString qsRes = qsStr;

  if (qsRes.length() < npad) 
    qsRes = QString(npad - qsRes.length(), '0') + qsRes;

  return qsRes;
}

#endif
