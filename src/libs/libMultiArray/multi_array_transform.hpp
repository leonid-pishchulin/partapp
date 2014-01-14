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

#ifndef _MULTI_ARRAY_TRANSFORM_HPP_
#define _MULTI_ARRAY_TRANSFORM_HPP_

#include <limits>

#include <libMisc/misc.hpp>

#include <libBoostMath/homogeneous_coord.h>
#include <libBoostMath/boost_math.h>

#include <libMultiArray/multi_array_def.h>

using boost_math::double_matrix;
using boost_math::identity_double_matrix;

enum TransformationMethod { 
  TM_NEAREST = 0, 
  TM_BILINEAR = 1,
  TM_DIRECT = 2
}; 

namespace multi_array_op {

  template <typename Array1, typename Array2>
  void transform_grid_vals_helper(const Array1 &grid_in, Array2 &grid_out, 
				  const double_matrix &T21, const double_matrix &T21_child, double_matrix &T23, 
				  typename Array1::element default_value,
				  TransformationMethod method)
  {
    using namespace boost::numeric::ublas;

    assert(Array1::dimensionality == 2 && Array2::dimensionality == 2);
    assert(T21.size1() == 3 && T21.size2() == 3);

    double_matrix T12 = hc::inverse(T21);
    double_matrix T12_child = hc::inverse(T21_child);

    int in_width = grid_in.shape()[1];
    int in_height = grid_in.shape()[0];

    int out_width, out_height;
    double_matrix T32;

    out_width = (int)grid_out.shape()[1];
    out_height = (int)grid_out.shape()[0];
    T32 = identity_double_matrix(3);
    T23 = identity_double_matrix(3);
    
    assert(method == TM_NEAREST);
    
    double_matrix T13 = prod(T12, T23);
    double_matrix T13_child = prod(T12_child, T23);

    for (int x3 = 0; x3 < out_width; ++x3)
      for (int y3 = 0; y3 < out_height; ++y3) {
	grid_out[y3][x3] = default_value;

	double x1, y1;
	hc::map_point(T13, (double)x3, (double)y3, x1, y1);
	
	int ix1, iy1;
	ix1 = (int)floor(x1 + 0.5);
	iy1 = (int)floor(y1 + 0.5);
        
	if (check_bounds(ix1, 0, in_width) && check_bounds(iy1, 0, in_height)) {
	  
	  unsigned long flat_idx = grid_in[iy1][ix1];
	  
	  int ridx = flat_idx / (in_width*in_height);
	  flat_idx = flat_idx % (in_width*in_height);
	  int yidx = flat_idx / in_width;
	  int xidx = flat_idx % in_width;
	  	  
	  double xidx_new_tmp, yidx_new_tmp;
	  hc::map_point(T13_child, (double)xidx, (double)yidx, xidx_new_tmp, yidx_new_tmp);
	
	  int xidx_new, yidx_new;
	  xidx_new = (int)floor(xidx_new_tmp + 0.5);
	  yidx_new = (int)floor(yidx_new_tmp + 0.5);
	  
	  if (check_bounds(xidx_new, 0, out_width) && check_bounds(yidx_new, 0, out_height)) 
	    grid_out[y3][x3] = xidx_new + in_width*yidx_new + ridx*(in_width*in_height);
	  
	}// check_bounds
      }// positions
  }

  /*                                                                                     */
  /* apply orthogonal transformation to 2-d array                                        */
  /*                                                                                     */
  /* transformation is defined by T21                                                    */
  /* T23 corrects transformation so that transformed positions of all cells              */
  /* have non-negative coordinates,                                                      */
  /* positions in new array which have no pre-image are set to default_vale              */
  /*                                                                                     */
  /*                                                                                     */
  /* note: T21 is the matrix used for TM_DIRECT, inv(T21) is used for TM_BILINEAR        */
  /*                                                                                     */
  /* note: T21 means 2T1, T21 = inv(T12)                                                 */
  /*                                                                                     */
  /* cs1 - original coordinate system (the one of grid_in)                               */
  /* cs2 - target coordinate system (the one of grid_out)                                */
  /* columns of T21 are basis vectors of cs1 expressed in cs2                            */
  /* T21*x_1 gives coordinates of x_1 in cs2                                             */

  template <typename Array1, typename Array2>
  void transform_grid_helper(const Array1 &grid_in, Array2 &grid_out, 
                             const double_matrix &T21, double_matrix &T23, 
                             typename Array1::element default_value,
                             TransformationMethod method, bool bAdaptive)
  {
    //cout << "multi_array_op::transform_grid_helper" << endl;
    using namespace boost::numeric::ublas;

    assert(Array1::dimensionality == 2 && Array2::dimensionality == 2);
    assert(T21.size1() == 3 && T21.size2() == 3);

    double_matrix T12 = hc::inverse(T21);

    int in_width = grid_in.shape()[1];
    int in_height = grid_in.shape()[0];

    int out_width, out_height;
    double_matrix T32;

    if (bAdaptive) {
      double minx, miny, maxx, maxy;
      hc::get_transformed_bbox(T21, in_width, in_height,
                               minx, miny, maxx, maxy);

      out_width = (int)ceil(maxx - minx);
      out_height = (int)ceil(maxy - miny);    

      T23 = hc::get_translation_matrix(minx, miny);
      T32 = hc::get_translation_matrix(-minx, -miny);

      // at this point resize is not possible, the function should also work with views and subarrays which can not be resized 
      //grid_out.resize(boost::extents[out_height][out_width]); 
      
      assert((int)grid_out.shape()[1] == out_width && 
             (int)grid_out.shape()[0] == out_height);

    }
    else {
      out_width = (int)grid_out.shape()[1];
      out_height = (int)grid_out.shape()[0];
      T32 = identity_double_matrix(3);
      T23 = identity_double_matrix(3);
    }

    if (method == TM_DIRECT) {
      /* initialize image to background */
      for (int x3 = 0; x3 < out_width; ++x3)
        for (int y3 = 0; y3 < out_height; ++y3)
          grid_out[y3][x3] = default_value;

      double_matrix T31 = prod(T32, T21);

      /* map points from original image to rotated image */
      for (int x1 = 0; x1 < in_width; ++x1) 
        for (int y1 = 0; y1 < in_height; ++y1) {
          if (grid_in[y1][x1] != default_value) {
            double x3, y3;
            hc::map_point(T31, (double)x1, (double)y1, x3, y3);

            int ix3 = (int)floor(x3 + 0.5);
            int iy3 = (int)floor(y3 + 0.5);

            if (check_bounds(ix3, 0, out_width) && check_bounds(iy3, 0, out_height)) {
              grid_out[iy3][ix3] = grid_in[y1][x1];
            }
          }
        
        }

    }// direct
    else {
      double_matrix T13 = prod(T12, T23);

      for (int x3 = 0; x3 < out_width; ++x3)
        for (int y3 = 0; y3 < out_height; ++y3) {
          grid_out[y3][x3] = default_value;

          double x1, y1;
          hc::map_point(T13, (double)x3, (double)y3, x1, y1);
	  
          int ix1, iy1;
          if (method == TM_BILINEAR) {
            ix1 = (int)floor(x1);
            iy1 = (int)floor(y1);
          }
          else {
            ix1 = (int)floor(x1 + 0.5);
            iy1 = (int)floor(y1 + 0.5);
          }

          if (check_bounds(ix1, 0, in_width) && check_bounds(iy1, 0, in_height)) {
            if (method == TM_NEAREST) {
              grid_out[y3][x3] = grid_in[iy1][ix1];
            }
            else if (method == TM_BILINEAR) {
              float a = x1 - ix1;
              float b = y1 - iy1;

              if (a < 10*std::numeric_limits<typename Array1::element>::epsilon() && 
                  b < 10*std::numeric_limits<typename Array1::element>::epsilon()) 
                grid_out[y3][x3] = grid_in[iy1][ix1];
              else if (ix1 < in_width - 1 && iy1 < in_height - 1) {
                //assert(out_img(x3, y3).value() == 0.0);

                grid_out[y3][x3] = (1.0f - b)*(1.0f - a)*grid_in[iy1][ix1] + 
                  (1.0f-b) * a * grid_in[iy1][ix1+1] +
                  b * (1.0f-a) * grid_in[iy1+1][ix1] +
                  b * a * grid_in[iy1+1][ix1+1];

              }
            }
            else {
              assert(false && "unknown transformation method");
            }// methods
          }// check_bounds
        }// positions
    }// bilinear or nearest


  }

  template <typename Array1, typename Array2>
  void transform_grid_fixed_size(const Array1 &grid_in, Array2 &grid_out,
                                 const double_matrix &T21, 
                                 typename Array1::element default_value,
                                 TransformationMethod method)
  {
    assert(Array1::dimensionality == 2);
    assert(Array2::dimensionality == 2);
    assert(grid_out.shape()[0] > 0 && grid_out.shape()[1] > 0);

    double_matrix T23;
    bool bAdaptive = false;
    transform_grid_helper(grid_in, grid_out, T21, T23, default_value, method, bAdaptive);
  }

  template <typename Array1, typename Array2>
  void transform_grid_vals_fixed_size(const Array1 &grid_in, Array2 &grid_out,
				      const double_matrix &T21, const double_matrix &T21_child, 
				      typename Array1::element default_value,
				      TransformationMethod method)
  {
    assert(Array1::dimensionality == 2);
    assert(Array2::dimensionality == 2);
    assert(grid_out.shape()[0] > 0 && grid_out.shape()[1] > 0);

    double_matrix T23;
    bool bAdaptive = false;
    transform_grid_vals_helper(grid_in, grid_out, T21, T21_child, T23, default_value, method);
  }

  template <typename Array1, typename Array2>
  void transform_grid(const Array1 &grid_in, Array2 &grid_out, 
                      const double_matrix &T21, double_matrix &T23, 
                      typename Array1::element default_value,
                      TransformationMethod method) 
  {
    bool bAdaptive = true;
    transform_grid_helper(grid_in, grid_out, T21, T23, default_value, method, bAdaptive);    
  }

  template <typename Array1>
  void transform_grid_resize(const Array1 &grid_in, FloatGrid2 &grid_out, 
                             const double_matrix &T21, double_matrix &T23, 
                             typename Array1::element default_value,
                             TransformationMethod method) 
  {
    double minx, miny, maxx, maxy;

    int in_width = grid_in.shape()[1];
    int in_height = grid_in.shape()[0];

    hc::get_transformed_bbox(T21, in_width, in_height,
                             minx, miny, maxx, maxy);

    int out_width = (int)ceil(maxx - minx);
    int out_height = (int)ceil(maxy - miny);    

    grid_out.resize(boost::extents[out_height][out_width]); 

    bool bAdaptive = true;
    transform_grid_helper(grid_in, grid_out, T21, T23, default_value, method, bAdaptive);    
  }

}// namespace 

#endif
