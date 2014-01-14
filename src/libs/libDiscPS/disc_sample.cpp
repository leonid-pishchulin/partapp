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

#include <iostream>
#include <cmath>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

#include "disc_ps.h"

using namespace std;

namespace disc_ps {

  void flatten_array(const FloatGrid4 &grid4, FloatGrid1 &grid1) 
  {
   int num_elements = grid4.num_elements();

    grid1.resize(boost::extents[num_elements]);
    int flat_idx = 0;

    int shape0 = grid4.shape()[0];
    int shape1 = grid4.shape()[1];
    int shape2 = grid4.shape()[2];
    int shape3 = grid4.shape()[3];

    for (int idx1 = 0; idx1 < shape0; ++idx1)
      for (int idx2 = 0; idx2 < shape1; ++idx2) 
	for (int idx3 = 0; idx3 < shape2; ++idx3)
	  for (int idx4 = 0; idx4 < shape3; ++idx4) {
	    grid1[flat_idx] = grid4[idx1][idx2][idx3][idx4];
	    ++flat_idx;
	  }
  }


  void flatten_array(const FloatGrid3 &grid3, FloatGrid1 &grid1) 
  {
    int num_elements = grid3.num_elements();

    grid1.resize(boost::extents[num_elements]);
    int flat_idx = 0;

    int shape0 = grid3.shape()[0];
    int shape1 = grid3.shape()[1];
    int shape2 = grid3.shape()[2];

    for (int idx1 = 0; idx1 < shape0; ++idx1)
      for (int idx2 = 0; idx2 < shape1; ++idx2) 
        for (int idx3 = 0; idx3 < shape2; ++idx3) {
          grid1[flat_idx] = grid3[idx1][idx2][idx3];
          ++flat_idx;
        }
  }

  void flatten_array(const FloatGrid2 &grid2, FloatGrid1 &grid1) 
  {
    int num_elements = grid2.num_elements();

    grid1.resize(boost::extents[num_elements]);
    int flat_idx = 0;

    int shape0 = grid2.shape()[0];
    int shape1 = grid2.shape()[1];
    for (int idx1 = 0; idx1 < shape0; ++idx1)
      for (int idx2 = 0; idx2 < shape1; ++idx2) {
        grid1[flat_idx] = grid2[idx1][idx2];
        ++flat_idx;
      }
  }

  void index_from_flat(const vector<int> shape, int flat_idx, vector<int> &idx) {
    if (shape.size() == 4) {
      int idx0, idx1, idx2, idx3;
      index_from_flat4(shape[0], shape[1], shape[2], shape[3], flat_idx, idx0, idx1, idx2, idx3);

      assert(idx.size() == 0);
      idx.clear();
      idx.push_back(idx0);
      idx.push_back(idx1);
      idx.push_back(idx2);
      idx.push_back(idx3);
    }
    else if (shape.size() == 3) {
      int idx0, idx1, idx2;
      index_from_flat3(shape[0], shape[1], shape[2], flat_idx, idx0, idx1, idx2);

      assert(idx.size() == 0);
      idx.clear();
      idx.push_back(idx0);
      idx.push_back(idx1);
      idx.push_back(idx2);
    }
    else {
      assert(false);
    }
  }
    
  void index_from_flat2(int shape0, int shape1, int flat_idx, int &idx1, int &idx2)
  {
    int num_elements = shape0*shape1;
    assert(flat_idx >= 0 && flat_idx < num_elements);

    idx2 = (flat_idx % shape1);
    idx1 = (flat_idx - idx2) / shape1;
  }

  void index_from_flat3(int shape0, int shape1, int shape2, int flat_idx, int &idx1, int &idx2, int &idx3)
  {
    int num_elements = shape0*shape1*shape2;

    assert(flat_idx >= 0 && flat_idx < num_elements);

    idx1 = flat_idx / (shape1*shape2);
    flat_idx = flat_idx % (shape1*shape2);

    idx2 = flat_idx / shape2;
    idx3 = flat_idx % shape2;

    assert(idx1 >= 0 && idx1 < shape0);
    assert(idx2 >= 0 && idx2 < shape1);
    assert(idx3 >= 0 && idx3 < shape2);
  }

  void index_from_flat4(int shape0, int shape1, int shape2, int shape3, int flat_idx, int &idx1, int &idx2, int &idx3, int &idx4)
  {
    uint num_elements = shape0*shape1*shape2*shape3;

    assert(flat_idx >= 0 && flat_idx < num_elements);

    idx1 = flat_idx / (shape1*shape2*shape3);
    flat_idx = flat_idx % (shape1*shape2*shape3);

    idx2 = flat_idx / (shape2*shape3);
    flat_idx = flat_idx % (shape2*shape3);

    idx3 = flat_idx / shape3;
    idx4 = flat_idx % shape3;

    assert(idx1 >= 0 && idx1 < shape0);
    assert(idx2 >= 0 && idx2 < shape1);
    assert(idx3 >= 0 && idx3 < shape2);
    assert(idx4 >= 0 && idx4 < shape3);
  }


  /**
     sample from discrete 3 dimensional distribution

     prob_grid: grid with probability for each discretized position
     n: number of samples
  */
//   void discrete_sample3(const FloatGrid3 &prob_grid, int n, 
//                         vector<int> &sample_dim1, vector<int> &sample_dim2, vector<int> &sample_dim3, int rnd_seed)
//   {
//     cout << "discrete_sample" << endl;

//     int shape0 = prob_grid.shape()[0];
//     int shape1 = prob_grid.shape()[1];
//     int shape2 = prob_grid.shape()[2];

//     FloatGrid1 flat_grid;
//     flatten_array(prob_grid, flat_grid);
//     int num_elements = flat_grid.num_elements();

//     FloatGrid1 sum_grid(boost::extents[num_elements+1]);

//     for (int idx = 0; idx < num_elements; ++idx) {
//       assert(flat_grid[idx] >= 0);
//       sum_grid[idx+1] = sum_grid[idx] + flat_grid[idx];
//     }
  
//     double total_sum = sum_grid[num_elements];
//     cout << "total sum: " << total_sum << endl;

//     /** initialize random number generator */
//     double min_rand = 0.0;
//     double max_rand = 1e6;

//     boost::mt19937 rng(42u + rnd_seed);
//     boost::uniform_real<> dist(min_rand, max_rand);
//     boost::variate_generator<boost::mt19937, boost::uniform_real<> > gen(rng, dist);

//     sample_dim1.clear();
//     sample_dim2.clear();
//     sample_dim3.clear();

//     /** generate samples */
//     for (int sampleidx = 0; sampleidx < n; ++sampleidx) {
//       double r = gen();
//       assert(r >= min_rand && r <= max_rand);
//       if (r == max_rand)
//         r -= 1e-6;

//       double c = total_sum / max_rand;
//       double r1 = r * c;

//       assert(r1 >= 0 && r1 < sum_grid[num_elements]);

//       int flat_idx = -1;


//       /** binary search BEGIN */

//       uint minidx = 0;
//       uint maxidx = num_elements;

//       int numiter = 0;

//       while (true) {
// 	++numiter;

// 	assert(maxidx > minidx);

// 	int curidx = floor(0.5*(maxidx + minidx));
// 	assert(curidx < num_elements);
	  
// 	if (r1 >= sum_grid[curidx] && r1 < sum_grid[curidx+1]) {
// 	  flat_idx = curidx;
// 	  break;
// 	}
// 	else if (r1 < sum_grid[curidx]) {
// 	  maxidx = curidx;
// 	}
// 	else if (r1 >= sum_grid[curidx+1]) {
// 	  minidx = curidx;
// 	} 

// 	assert(numiter < 1000);
//       }

//       assert(flat_idx > 0);
//       assert(flat_idx < num_elements);
//       assert(r1 >= sum_grid[flat_idx]);
//       assert(r1 < sum_grid[flat_idx + 1]);

//       /** binary search END */
      

//       if (flat_idx < 0) {
//         cout << "****************************************" << endl;
//         cout << "warning, negative index in discrete_sample3" << endl;
//         cout << "\t min_rand: " << min_rand << endl;
//         cout << "\t max_rand: " << max_rand << endl;
//         cout << "\t r: " << r << endl;
//         cout << "\t r1: " << r1 << endl;
//         cout << "\t sum_grid[0]: " << sum_grid[0] << endl;
//         cout << "\t sum_grid[end]: " << sum_grid[num_elements] << endl;
//         cout << "\t flat_idx: " << flat_idx << endl;
//         cout << "****************************************" << endl << endl;

//         flat_idx = 0;
//       }

//       //assert(flat_idx >= 0);

// //       cout << "sampleidx: " << sampleidx << 
// //         ", flat_idx: " << flat_idx << 
// //         ", r1: " << r1 << 
// //         ", seg_left: " << sum_grid[flat_idx] << 
// //         ", seg_right: " << sum_grid[flat_idx+1] << endl;

//       int idx1;
//       int idx2;
//       int idx3;
    
//       index_from_flat3(shape0, shape1, shape2, flat_idx, 
//                        idx1, idx2, idx3);

//       sample_dim1.push_back(idx1);
//       sample_dim2.push_back(idx2);
//       sample_dim3.push_back(idx3);
//     }
//   }


}
