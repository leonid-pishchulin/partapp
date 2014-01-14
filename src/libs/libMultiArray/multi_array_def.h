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

#ifndef _MULTI_ARRAY_DEF_H_
#define _MULTI_ARRAY_DEF_H_


/** 
#warning BOOST_DISABLE_ASSERTS is ON
#define BOOST_DISABLE_ASSERTS 
*/

#include <boost/multi_array.hpp>

typedef boost::multi_array<float, 1> FloatGrid1;
typedef boost::multi_array<float, 2> FloatGrid2;

typedef boost::multi_array<double, 2> DoubleGrid2;

typedef boost::multi_array<long double, 2> LongDoubleGrid2;

typedef boost::multi_array<float, 3> FloatGrid3;
typedef boost::multi_array<float, 4> FloatGrid4;
typedef boost::multi_array<float, 5> FloatGrid5;

typedef boost::array_view_gen<FloatGrid2, 1>::type FloatGrid2View1;

typedef boost::array_view_gen<FloatGrid3, 1>::type FloatGrid3View1;
typedef boost::array_view_gen<FloatGrid3, 2>::type FloatGrid3View2;

typedef boost::array_view_gen<FloatGrid4, 1>::type FloatGrid4View1;
typedef boost::array_view_gen<FloatGrid4, 2>::type FloatGrid4View2;

typedef boost::array_view_gen<FloatGrid5, 3>::type FloatGrid5View3;

typedef boost::const_array_view_gen<const FloatGrid2, 1>::type ConstFloatGrid2View1;

typedef boost::multi_array<unsigned long, 3> LongGrid3;
typedef boost::array_view_gen<LongGrid3, 1>::type LongGrid3View1;
typedef boost::array_view_gen<LongGrid3, 2>::type LongGrid3View2;

typedef boost::multi_array<unsigned int, 3> IntGrid3;
typedef boost::array_view_gen<IntGrid3, 1>::type IntGrid3View1;
#endif
