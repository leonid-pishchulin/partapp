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
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

#include <libBoostMath/boost_math.h>

#include "disc_ps.h"

namespace disc_ps {


  /**
     sample from discrete 3 dimensional distribution

     prob_grid: grid with probability for each discretized position
     n: number of samples
  */
  template <typename Array>
  void discrete_sample(const Array &prob_grid, int n, boost_math::int_matrix &sample_dim, int rnd_seed)
  {
    std::cout << "template version" << std::endl;

    sample_dim.resize(n, Array::dimensionality);
    sample_dim = boost_math::zero_int_matrix(n, Array::dimensionality);

    FloatGrid1 flat_grid;
    flatten_array(prob_grid, flat_grid);
    int num_elements = flat_grid.num_elements();

    FloatGrid1 sum_grid(boost::extents[num_elements+1]);

    for (int idx = 0; idx < num_elements; ++idx) {
      assert(flat_grid[idx] >= 0);
      sum_grid[idx+1] = sum_grid[idx] + flat_grid[idx];
    }
  
    double total_sum = sum_grid[num_elements];
    std::cout << "total sum: " << total_sum << std::endl;

    /** initialize random number generator */
    double min_rand = 0.0;
    double max_rand = 1e6;

    boost::mt19937 rng(42u + rnd_seed);
    boost::uniform_real<> dist(min_rand, max_rand);
    boost::variate_generator<boost::mt19937, boost::uniform_real<> > gen(rng, dist);

    /** generate samples */
    for (int sampleidx = 0; sampleidx < n; ++sampleidx) {
      double r = gen();
      assert(r >= min_rand && r <= max_rand);
      if (r == max_rand)
        r -= 1e-6;

      double c = total_sum / max_rand;
      double r1 = r * c;

      assert(r1 >= 0 && r1 < sum_grid[num_elements]);

      int flat_idx = -1;


      /** binary search BEGIN */

      uint minidx = 0;
      uint maxidx = num_elements;

      int numiter = 0;

      while (true) {
	++numiter;

	assert(maxidx > minidx);

	int curidx = floor(0.5*(maxidx + minidx));
	assert(curidx < num_elements);
	  
	if (r1 >= sum_grid[curidx] && r1 < sum_grid[curidx+1]) {
	  flat_idx = curidx;
	  break;
	}
	else if (r1 < sum_grid[curidx]) {
	  maxidx = curidx;
	}
	else if (r1 >= sum_grid[curidx+1]) {
	  minidx = curidx;
	} 

	assert(numiter < 1000);
      }

      assert(flat_idx > 0);
      assert(flat_idx < num_elements);
      assert(r1 >= sum_grid[flat_idx]);
      assert(r1 < sum_grid[flat_idx + 1]);

      /** binary search END */
      

      if (flat_idx < 0) {
        std::cout << "****************************************" << std::endl;
        std::cout << "warning, negative index in discrete_sample3" << std::endl;
        std::cout << "\t min_rand: " << min_rand << std::endl;
        std::cout << "\t max_rand: " << max_rand << std::endl;
        std::cout << "\t r: " << r << std::endl;
        std::cout << "\t r1: " << r1 << std::endl;
        std::cout << "\t sum_grid[0]: " << sum_grid[0] << std::endl;
        std::cout << "\t sum_grid[end]: " << sum_grid[num_elements] << std::endl;
        std::cout << "\t flat_idx: " << flat_idx << std::endl;
        std::cout << "****************************************" << std::endl << std::endl;

        flat_idx = 0;
      }

      std::vector<int> shape;
      for (uint idx = 0; idx < Array::dimensionality; ++idx)
	shape.push_back(prob_grid.shape()[idx]);

      std::vector<int> v_idx;
      index_from_flat(shape, flat_idx, v_idx);

      for (uint idx = 0; idx < Array::dimensionality; ++idx) {
	sample_dim(sampleidx, idx) = v_idx[idx];
      }


    }
  }



}
