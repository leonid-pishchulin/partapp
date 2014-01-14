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

#ifndef _MULTI_ARRAY_OP_H_
#define _MULTI_ARRAY_OP_H_

#include <cassert>
#include <climits>
#include <cstdio>

#include <libMultiArray/multi_array_def.h>
#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <boost/lambda/lambda.hpp>

#include <limits.h>
#include <float.h>

#define INF 1E30

using boost::multi_array_types::index_range;

namespace multi_array_op 
{
  using std::cout;
  using std::endl;

  static inline int square(int x) { return x*x; }

  template <typename Array>
  void normalize(Array &a)
  {
    using namespace boost::lambda;

    typename Array::element *pData = a.data();
    uint nElements = a.num_elements();
 
    typename Array::element *pDataEnd = pData + nElements;
    typename Array::element val_sum = std::accumulate(pData, pDataEnd, 0.0);

    assert(val_sum > 0);

    std::for_each(pData, pDataEnd, _1 = _1 / val_sum);
  }

  template <typename Array> 
  void getMinMax(const Array &a, typename Array::element &minval, typename Array::element &maxval)
  {
    const typename Array::element *pData = a.data();
    uint nElements = a.num_elements();
  
    minval = std::numeric_limits<typename Array::element>::infinity();
    maxval = -std::numeric_limits<typename Array::element>::infinity();

    for (uint i = 0; i < nElements; ++i) {
      if (pData[i] > maxval) 
        maxval = pData[i];

      if (pData[i] < minval)
        minval = pData[i];
    }
    
  }

  template <typename Array>
  void multGrid(Array &grid, typename Array::element scalar)
  {
    typename Array::element *data = grid.data();
    int nElements = grid.num_elements();

    for (int i = 0; i < nElements; ++i) 
      data[i] = scalar*data[i];
  }

  template <typename Array> 
  void setGrid(Array &grid, typename Array::element num) {
    typename Array::element *pData = grid.data();
    int nElements = grid.num_elements();

    for (int i = 0; i < nElements; ++i)
      pData[i] = num;
  }

  template <typename Array> 
  void addGrid1(Array &grid, typename Array::element num)
  {
    typename Array::element *pData = grid.data();
    int nElements = grid.num_elements();

    for (int i = 0; i < nElements; ++i)
      pData[i] += num;
  }

  template <typename Array1, typename Array2> 
  void addGrid2(Array1 &grid1, const Array2 &grid2)
  {
    assert(grid1.storage_order() == grid2.storage_order());

    typename Array1::element *pData1 = grid1.data();
    const typename Array2::element *pData2 = grid2.data();

    assert(grid1.num_elements() == grid2.num_elements());

    int nElements = grid1.num_elements();
    for (int i = 0; i < nElements; ++i)
      pData1[i] += pData2[i];
  }

  template <typename Array1, typename Array2> 
  void multGrid2(Array1 &grid1, const Array2 &grid2)
  {
    assert(grid1.storage_order() == grid2.storage_order());

    typename Array1::element *pData1 = grid1.data();
    const typename Array2::element *pData2 = grid2.data();

    assert(grid1.num_elements() == grid2.num_elements());

    int nElements = grid1.num_elements();
    for (int i = 0; i < nElements; ++i)
      pData1[i] *= pData2[i];
  }

  template <typename Array1, typename Array2> 
  void divGrid2(Array1 &grid1, const Array2 &grid2)
  {
    assert(grid1.storage_order() == grid2.storage_order());

    typename Array1::element *pData1 = grid1.data();
    const typename Array2::element *pData2 = grid2.data();

    assert(grid1.num_elements() == grid2.num_elements());

    int nElements = grid1.num_elements();
    for (int i = 0; i < nElements; ++i)
      pData1[i] /= pData2[i];
  }

  template <typename Array>
  void computeLogGrid(Array &prob_grid)
  {
    typename Array::element *data = prob_grid.data();
    int nElements = prob_grid.num_elements();

    for (int i = 0; i < nElements; ++i) {
      assert(data[i] >= 0);

      if (data[i] == 0)
        data[i] = LOG_ZERO;
      else
        data[i] = log(data[i]);
    }
  }

  template <typename Array>
  void computeExpGrid(Array &prob_grid)
  {
    typename Array::element *data = prob_grid.data();
    int nElements = prob_grid.num_elements();

    for (int i = 0; i < nElements; ++i) {
      assert(data[i] < std::numeric_limits<double>::max());
      data[i] = exp(data[i]);
      assert(data[i] < std::numeric_limits<double>::max());
    }
  }

  template <typename Array>
  void computeNegLogGrid(Array &prob_grid)
  {
    typename Array::element *data = prob_grid.data();
    int nElements = prob_grid.num_elements();

    for (int i = 0; i < nElements; ++i) {
      assert(data[i] >= 0);

      if (data[i] == 0)
        data[i] = NEG_LOG_ZERO;
      else
        data[i] = -log(data[i]);
    }
  }

  template <typename Array> 
  void array_to_matrix(const Array &ar, boost_math::double_matrix &m)
  {
    int width = ar.shape()[1];
    int height = ar.shape()[0];

    m.resize(height, width);
    for (int ix = 0; ix < width; ++ix)
      for (int iy = 0; iy < height; ++iy)
        m(iy, ix) = ar[iy][ix];
  }

  template <typename Array>
  void clip_scores(Array &grid, float clip_val) {
    uint nElements = grid.num_elements();
    typename Array::element *pData = grid.data();

    for (uint i3 = 0; i3 < nElements; ++i3) {
      if (pData[i3] < clip_val)
        pData[i3] = clip_val;
    }     
  }

  /**
     this is supposed to be used for setting the classifier scores computed 
     at positions with fixed offset to some minimum value 

     note: it is assumed that unclassified positions have score = 0

     note: we only change scores at evaluated locations (i.e. score != 0),  
           setting scores at all positions to some min_val > 0 would make 
           marginals at different scales incomparable 
   */
  template <typename Array>
  void clip_scores_grid_minval(Array &grid, double min_val = 1e-4) {
    //double min_val = 0.0001;
    
    uint nElements = grid.num_elements();
    typename Array::element *pData = grid.data();

    //int set_count = 0;

    for (uint i3 = 0; i3 < nElements; ++i3) {
      if (pData[i3] < 0) {
        pData[i3] = min_val;
        //++set_count;
      }
    }     
    //cout << "clip_scores_test: minval: " << min_val << ", set_count: " << set_count << endl;
  }
  
  template <typename Array1, typename Array2> 
  void distance_transform_1d(const Array1 &_grid_in, Array2 &_grid_out, float c, int mean_idx = 0, int step = 1)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);

    int n = _grid_in.shape()[0];
    assert((int)_grid_out.shape()[0] == n);

    FloatGrid1 grid_in = _grid_in;
    const float *f = grid_in.data();
  
    int *v = new int[n];
    float *z = new float[n+1];
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= n-1; q++) {
      float s  = ((f[q*step] - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      while (s <= z[k]) {
	k--;
	s  = ((f[q]*step - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = +INF;
    }
    
    k = 0;
    for (int q = 0; q <= n-1; q++) {
      while (z[k+1] < q)
	k++;
      _grid_out[q*step] = c*square(q-v[k]-mean_idx) + f[v[k]*step];
      //_grid_out[q*step] = c*square(q-v[k]) + f[v[k]*step];
    }
    
    delete [] v;
    delete [] z;
  }

  template <typename Array1, typename Array2, typename Array3> 
  void distance_transform_backtrace_1d(const Array1 &_grid_in, Array2 &_grid_out, Array3 &_idx_out, float c, int mean_idx = 0, int step = 1)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);

    int n = _grid_in.shape()[0];
    assert((int)_grid_out.shape()[0] == n);

    FloatGrid1 grid_in = _grid_in;
    const float *f = grid_in.data();
  
    int *v = new int[n];
    float *z = new float[n+1];
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= n-1; q++) {
      float s  = ((f[q*step] - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      while (s <= z[k]) {
	k--;
	s  = ((f[q]*step - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = +INF;
    }
    
    k = 0;
    for (int q = 0; q <= n-1; q++) {
      while (z[k+1] < q)
	k++;
      _grid_out[q*step] = c*square(q-v[k]-mean_idx) + f[v[k]*step];
      _idx_out[q*step] = v[k];
      //_grid_out[q*step] = c*square(q-v[k]) + f[v[k]*step];
    }
    
    delete [] v;
    delete [] z;
  }

  template <typename Array1, typename Array2> 
  void distance_transform_substr_mean_1d(const Array1 &_grid_in, Array2 &_grid_out, float c, int mean_idx = 0, int step = 1)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);

    int n = _grid_in.shape()[0];
    assert((int)_grid_out.shape()[0] == n);

    FloatGrid1 grid_in = _grid_in;
    const float *f = grid_in.data();
  
    int *v = new int[n];
    float *z = new float[n+1];
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int qq = 1; qq <= n-1; qq++) {
      int q = qq + mean_idx;
      if (q >= 1 && q <= n-1){
      float s  = ((f[q*step] - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      while (s <= z[k]) {
	k--;
	s  = ((f[q]*step - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = +INF;
      }
    }
    
    k = 0;
    for (int qq = 0; qq <= n-1; qq++) {
      int q = qq + mean_idx;
      if (q >= 0 && q <= n-1){
      while (z[k+1] < q)
	k++;
      _grid_out[q*step] = c*square(q-v[k]) + f[v[k]*step];
      }
      //_grid_out[q*step] = c*square(q-v[k]) + f[v[k]*step];
    }
    
    delete [] v;
    delete [] z;
  }

  template <typename Array1, typename Array2> 
  void distance_transform_wrap_2pi_1d(const Array1 &_grid_in, Array2 &_grid_out, float c, int mean_idx = 0, int step = 1)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);
    
    int n = _grid_in.shape()[0];
    assert((int)_grid_out.shape()[0] == n);
    
    FloatGrid1 grid_in = _grid_in;
    const float *f = grid_in.data();
  
    int *v = new int[n];
    float *z = new float[n+1];
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= n-1; q++) {
      float s  = ((f[q*step] - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      while (s <= z[k]) {
	k--;
	s  = ((f[q]*step - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = +INF;
    }
    
    k = 0;
    for (int q = 0; q <= n-1; q++) {
      while (z[k+1] < q)
	k++;
      int diff = q-v[k]-mean_idx;
      
      while (diff < 0)
	diff += n;
      while (diff > n)
	diff -= n;
      
      _grid_out[q*step] = c*square(diff) + f[v[k]*step];
    }
    
    delete [] v;
    delete [] z;
  }

  template <typename Array1, typename Array2> 
  void distance_transform_wraparound_1d(const Array1 &_grid_in, Array2 &_grid_out, float c, int step = 1)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);
    assert(_grid_out.shape()[0] == _grid_in.shape()[0]);
    
    int f_len = _grid_in.shape()[0];
    
    assert(f_len % 2 == 0);
    int nx = f_len / 2;

    //assert(f_len % 2 == 1);
    //int nx = (f_len - 1) / 2;

    int owidth = _grid_in.shape()[0];
    
    /** copy the main part */
    FloatGrid1 grid_in(boost::extents[owidth + 2*nx]);
    grid_in[boost::indices[index_range(nx, nx + owidth)]] = _grid_in;

    //assert((int)_grid_in.shape()[0] > f_len);

    /** copy parts for wrap-around */
    for (int idx = 0; idx < nx; ++idx)
      grid_in[nx - 1 - idx] = _grid_in[owidth - 1 - idx];

    for (int idx = 0; idx < nx; ++idx)
      grid_in[nx + owidth + idx] = _grid_in[idx];

    
    
    int n = grid_in.shape()[0];
    FloatGrid1 grid_out(boost::extents[owidth + 2*nx]);
    assert((int)grid_out.shape()[0] == n);

    //FloatGrid1 grid_in = _grid_in;
    const float *f = grid_in.data();
  
    int *v = new int[n];
    float *z = new float[n+1];
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= n-1; q++) {
      float s  = ((f[q*step] - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      while (s <= z[k]) {
	k--;
	s  = ((f[q]*step - f[v[k]*step]) + c*(square(q) - square(v[k]))) / (2*c*(q-v[k]));
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = +INF;
    }
    
    k = 0;
    for (int q = 0; q <= n-1; q++) {
      while (z[k+1] < q)
	k++;
      grid_out[q*step] = c*square(q-v[k]) + f[v[k]*step];
    }
    
    delete [] v;
    delete [] z;

    for (int x = nx; x < nx + owidth; ++x) {
      _grid_out[x - nx] = grid_out[x];
    }
    
  }

}// namespace 



#endif
