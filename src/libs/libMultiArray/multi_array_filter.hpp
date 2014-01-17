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

#ifndef _MULTI_ARRAY_FILTER_HPP_
#define _MULTI_ARRAY_FILTER_HPP_

#include <cblas.h>

#include <libBoostMath/boost_math.h>

#include <libBoostMath/homogeneous_coord.h>
#include <libMultiArray/multi_array_transform.hpp>

namespace multi_array_op 
{
  using boost_math::double_matrix;
  using boost_math::double_vector;
  using boost::multi_array_types::index_range;

  template <typename Array1, typename Array2> 
  void grid_filter_1d(const Array1 &grid_in, Array2 &grid_out, const double_vector &f)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);
    assert(grid_in.shape()[0] == grid_out.shape()[0]);

    int grid_size = grid_in.shape()[0];
    int ksize = (f.size() - 1)/2;

    for (int s = 0; s < grid_size; ++s) {
      int n1 = std::max(s - ksize, 0);
      int n2 = std::min(s + ksize, grid_size-1);
      float val = 0.0;
      for (int s1 = n1; s1 <= n2; ++s1)
        val += grid_in[s1]*f[s1 - (s - ksize)];

      grid_out[s] = val;
    }
  }

  template <typename Array1, typename Array2>
  void grid_filter_1d_blas(const Array1 &_grid_in, Array2 &_grid_out, const float *f, int f_len)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);

    int width = _grid_in.shape()[0];
    assert((int)_grid_out.shape()[0] == width);

    FloatGrid1 grid_in = _grid_in;
    const float *ptr_in = grid_in.data();
  
    assert(f_len % 2 == 1);
    int nx = (f_len - 1) / 2;

    for (int x = 0; x < width; ++x) {
      int n1 = std::max(0, x - nx);
      int n2 = std::min(width - 1, x + nx);

      const float *cur_ptr_in = ptr_in + n1;
      const float *ptr_f = f + (n1 - (x - nx));

      int _len = n2 - n1 + 1;
      _grid_out[x] = cblas_sdot(_len, cur_ptr_in, 1, ptr_f, 1);   
    }
  }


  template <typename Array1, typename Array2> 
  void grid_filter_1d_wraparound(const Array1 &grid_in, Array2 &grid_out, const double_vector &f)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);
    assert(grid_in.shape()[0] == grid_out.shape()[0]);

    int grid_size = grid_in.shape()[0];
    int ksize = (f.size() - 1)/2;

    int maxidx = grid_size - 1;

    for (int s = 0; s < grid_size; ++s) {

      float val = 0.0;
      for (int s1 = s-ksize; s1 <= s+ksize; ++s1) {
	float g;

	if (s1 >= 0 && s1 <= maxidx) {
	  g = grid_in[s1];
	}
	else if (s1 < 0) {
	  g = grid_in[grid_size + s1];
	}
	else if (s1 > maxidx) {
	  g = grid_in[s1 - grid_size];
	}

        val += g*f[s1 - (s - ksize)];
      }

      grid_out[s] = val;
    }
  }


  template <typename Array1, typename Array2>
  void grid_filter_1d_blas_wraparound(const Array1 &_grid_in, Array2 &_grid_out, const float *f, int f_len)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);
    assert(_grid_out.shape()[0] == _grid_in.shape()[0]);

    assert(f_len % 2 == 1);
    int nx = (f_len - 1) / 2;

    int owidth = _grid_in.shape()[0];

    /** copy the main part */
    FloatGrid1 grid_in(boost::extents[owidth + 2*nx]);
    grid_in[boost::indices[index_range(nx, nx + owidth)]] = _grid_in;

    //std::cout << _grid_in.shape()[0] << std::endl;
    //std::cout << f_len << std::endl;
    
    //assert((int)_grid_in.shape()[0] > f_len);
    
    /** copy parts for wrap-around */
    for (int idx = 0; idx < nx; ++idx)
      grid_in[nx - 1 - idx] = _grid_in[owidth - 1 - idx];

    for (int idx = 0; idx < nx; ++idx)
      grid_in[nx + owidth + idx] = _grid_in[idx];

    const float *ptr_in = grid_in.data();
    
    for (int x = nx; x < nx + owidth; ++x) {
      int n1 = x - nx;
      int n2 = x + nx;

      assert(n1 >= 0);
      assert(n2 < (int)grid_in.shape()[0]);

      const float *cur_ptr_in = ptr_in + n1;

      _grid_out[x - nx] = cblas_sdot(f_len, cur_ptr_in, 1, f, 1);   
    }
  }


  /**
     apply isotropic Gaussian filter to 2d array 

     bNormalize true if the filter should be normalized
  */
  template <typename Array1, typename Array2>
  void gaussFilterDiag2dPlain(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, bool bNormalize) {

    //cout << "gaussFilterDiag2dPlain"  << endl;


    assert(C.size1() == 2 && C.size2() == 2);
    assert(C(0, 1) == 0 && C(1, 0) == 0);
    assert(C(0, 0) > 0 && C(1, 1) > 0);

    assert(Array1::dimensionality == 2);
    assert(Array2::dimensionality == 2);

    int grid_width = grid_in.shape()[1];
    int grid_height = grid_in.shape()[0];

    // it is not possible to resize grid_out if it is a view, enforce correct size
    assert((int)grid_out.shape()[0] == grid_height);
    assert((int)grid_out.shape()[1] == grid_width);

    double sigma_x = sqrt(C(0, 0));
    double sigma_y = sqrt(C(1, 1));

    double_vector f_x, f_y;
    boost_math::get_gaussian_filter(f_x, sigma_x, bNormalize);
    boost_math::get_gaussian_filter(f_y, sigma_y, bNormalize);

    FloatGrid2 grid_smooth_x(boost::extents[grid_height][grid_width]);

    for (int iy = 0; iy < grid_height; ++iy) {
      ConstFloatGrid2View1 view_in = grid_in[boost::indices[iy][index_range()]];
      FloatGrid2View1 view_out = grid_smooth_x[boost::indices[iy][index_range()]];
      multi_array_op::grid_filter_1d(view_in, view_out, f_x);
    }

    for (int ix = 0; ix < grid_width; ++ix) {
      FloatGrid2View1 view_in = grid_smooth_x[boost::indices[index_range()][ix]];
      FloatGrid2View1 view_out = grid_out[boost::indices[index_range()][ix]];
      multi_array_op::grid_filter_1d(view_in, view_out, f_y);
    }
  }

  /**
     apply isotropic Gaussian filter to 2d array (use dot-product from BLAS)

     bNormalize true if the filter should be normalized
  */

  template <typename Array1, typename Array2>
  void gaussFilterDiag2d(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, bool bNormalize) {
    //cout << "gaussFilterDiag2dBlas"  << endl;

    //void gaussFilterDiag2dBlas(const FloatGrid2 &grid_in, FloatGrid2 &grid_out, const double_matrix &C, bool bNormalize) {

    assert(C.size1() == 2 && C.size2() == 2);
    assert(C(0, 1) == 0 && C(1, 0) == 0);
    assert(C(0, 0) > 0 && C(1, 1) > 0);

    //assert(Array1::dimensionality == 2);
    //assert(Array2::dimensionality == 2);

    int grid_width = grid_in.shape()[1];
    int grid_height = grid_in.shape()[0];
  
    // it is not possible to resize grid_out if it is a view, enforce correct size
    assert((int)grid_out.shape()[0] == grid_height);
    assert((int)grid_out.shape()[1] == grid_width);

    double sigma_x = sqrt(C(0, 0));
    double sigma_y = sqrt(C(1, 1));

    double_vector _f_x, _f_y;
    boost_math::get_gaussian_filter(_f_x, sigma_x, bNormalize);
    boost_math::get_gaussian_filter(_f_y, sigma_y, bNormalize);

    const uint F_SIZE = 1000;

    float f_x[F_SIZE];
    float f_y[F_SIZE];

    assert(_f_x.size() < F_SIZE);
    assert(_f_y.size() < F_SIZE);

    for (uint idx = 0; idx < _f_x.size(); ++idx)
      f_x[idx] = _f_x(idx);
  
    for (uint idx = 0; idx < _f_y.size(); ++idx)
      f_y[idx] = _f_y(idx);

    FloatGrid2 grid_smooth_x(boost::extents[grid_height][grid_width]);


    /** 
	filter using dot product from BLAS
    */

    FloatGrid2 _grid_in = grid_in;
    FloatGrid2 _grid_out(boost::extents[grid_height][grid_width]);

    const float *ptr_in = _grid_in.data();

    std::vector<const float*> rows(grid_height, (float*)0);
    for (int y = 0; y < grid_height; ++y) {
      rows[y] = ptr_in + y*grid_width;
    }

    int f_x_len = _f_x.size();
    assert(f_x_len % 2 == 1);
    int nx = (f_x_len - 1) / 2;

    /** 
	filter in x direction
    */

    for (int y = 0; y < grid_height; ++y) {
      for (int x = 0; x < grid_width; ++x) {
	int n1 = std::max(0, x - nx);
	int n2 = std::min(grid_width - 1, x + nx);

	const float *ptr_rows = rows[y] + n1;
	float *ptr_f = f_x + (n1 - (x - nx));
	int _len = n2 - n1 + 1;
	float elem = cblas_sdot(_len, ptr_rows, 1, ptr_f, 1);
	grid_smooth_x[y][x] = elem;
      }
    }

    /**
       filter in y direction
    */

    float *ptr_smooth_x = grid_smooth_x.data();
    std::vector<float*> cols(grid_width, (float*)0);

    for (int x = 0; x < grid_width; ++x)
      cols[x] = ptr_smooth_x + x;

    int f_y_len = _f_y.size();
    assert(f_y_len % 2 == 1);
    int ny = (f_y_len - 1) / 2;

    for (int y = 0; y < grid_height; ++y) {
      for (int x = 0; x < grid_width; ++x) {
      
	int n1 = std::max(0, y - ny);
	int n2 = std::min(grid_height - 1, y + ny);

	float *ptr_cols = cols[x] + n1*grid_width;
	float *ptr_f = f_y + (n1 - (y - ny));
      
	int _len = n2 - n1 + 1;
	float elem = cblas_sdot(_len, ptr_cols, grid_width, ptr_f, 1);
	_grid_out[y][x] = elem;
      }
    }

    grid_out = _grid_out;

  }


  /**
     apply Gaussian filter with arbitrary covariance matrix to 2d array

     - transform array to CS in which covariance matrix is diagonal (use bilinear interpolation)
     - apply isotropic filter
     - transform back

     bIsSparse: if true the input array is sparce (bilinear interpolation is replaced with direct mapping)
     offset: offset the grid simultaneously with the back transformation 
  */
  template <typename Array1, typename Array2>
  void gaussFilter2dOffset(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, 
                           const double_vector &offset, bool bNormalize, bool bIsSparse) {

    const int PADDING_VALUE = 0;

    assert(C.size1() == 2 && C.size2() == 2);
    assert(offset.size() == 2);
 
    // it is not possible to resize grid_out if it is a view, enforce correct size
    assert(grid_out.shape()[0] == grid_in.shape()[0]);
    assert(grid_out.shape()[1] == grid_in.shape()[1]);

    double_matrix V(2, 2);
    double_matrix E(2, 2);

    /* K = V*E*V', invK = V*inv(E)V' */
    boost_math::eig2d(C, V, E);
    double_matrix T21 = hc::get_homogeneous_matrix(trans(V), 0, 0);
    double_matrix T23;
    FloatGrid2 grid_in_transformed;

    if (bIsSparse) 
      multi_array_op::transform_grid_resize(grid_in, grid_in_transformed, T21, T23, PADDING_VALUE, TM_DIRECT);
    else
      multi_array_op::transform_grid_resize(grid_in, grid_in_transformed, T21, T23, PADDING_VALUE, TM_BILINEAR);

    FloatGrid2 grid_in_transformed_smooth(boost::extents[grid_in_transformed.shape()[0]][grid_in_transformed.shape()[1]]);

    gaussFilterDiag2d(grid_in_transformed, grid_in_transformed_smooth, E, bNormalize);
    double_matrix T42 = hc::get_homogeneous_matrix(V, -offset(0), -offset(1));
    double_matrix T43 = prod(T42, T23);

    multi_array_op::transform_grid_fixed_size(grid_in_transformed_smooth, grid_out,
                                              T43, PADDING_VALUE, TM_BILINEAR);    
  }

  /**
     convinience function (offset = 0)
  */
  template <typename Array1, typename Array2>
  void gaussFilter2d(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, 
                     bool bNormalize, bool bIsSparse) {
    assert(C.size1() == 2 && C.size2() == 2);
    
    bool bIsDiag = (C(0,1) == 0 && C(1,0) == 0);

    if (bIsDiag) { 
      gaussFilterDiag2d(grid_in, grid_out, C, bNormalize);
    }
    else {
      double_vector offset = boost_math::double_zero_vector(2);
      gaussFilter2dOffset(grid_in, grid_out, C, offset, bNormalize, bIsSparse);
    }
  }
  


}// namespace 

#endif
