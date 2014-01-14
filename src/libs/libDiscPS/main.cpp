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

// 
// test file for the libDiscPS library
// 

#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

#include "disc_sample.h"

using namespace std;

int main()
{

  cout << "running tests for the libDiscPS" << endl;

  FloatGrid3 prob_grid(boost::extents[3000][300][300]);

  boost::mt19937 rng(42);
  boost::uniform_int<> distidx(0, 299);
  boost::uniform_real<> dist(0, 1);
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > genidx(rng, distidx);
  boost::variate_generator<boost::mt19937, boost::uniform_real<> > gen(rng, dist);

  for (uint idx = 0; idx < 300*300; ++idx)
    prob_grid[int(genidx())][int(genidx())][int(genidx())] = gen();

//   prob_grid[1][1][1] = 4;
//   prob_grid[1][0][1] = 1;
//   prob_grid[1][2][1] = 1;

  std::vector<int> sample_dim1;
  std::vector<int> sample_dim2;
  std::vector<int> sample_dim3;

  int rnd_seed = 10;
  uint num_samples = 300;

  disc_ps::discrete_sample3(prob_grid, num_samples, sample_dim1, sample_dim2, sample_dim3, rnd_seed, false);

  for (uint idx = 0; idx < num_samples; ++idx)
    cout << "sample " << idx << ": " << sample_dim1[idx] << " " << sample_dim2[idx] << " " << sample_dim3[idx] << endl;


  disc_ps::discrete_sample3(prob_grid, num_samples, sample_dim1, sample_dim2, sample_dim3, rnd_seed);

  for (uint idx = 0; idx < num_samples; ++idx)
    cout << "sample " << idx << ": " << sample_dim1[idx] << " " << sample_dim2[idx] << " " << sample_dim3[idx] << endl;

  return 0;
}
