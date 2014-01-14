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

#ifndef _OBJECT_DETECT_SAMPLE_H_
#define _OBJECT_DETECT_SAMPLE_H_

#include <vector>
#include <libMultiArray/multi_array_def.h>

namespace object_detect {

  void flatten_array3(const FloatGrid3 &grid3, FloatGrid1 &grid1);
  void flatten_array(const FloatGrid2 &grid2, FloatGrid1 &grid1);
  void index_from_flat(int shape0, int shape1, int flat_idx, int &idx1, int &idx2);
  void index_from_flat3(int shape0, int shape1, int shape2, int flat_idx, int &idx1, int &idx2, int &idx3);
    
  void discrete_sample(const FloatGrid2 &prob_grid, int n, std::vector<int> &sample_x, std::vector<int> &sample_y, int rnd_seed);
  void discrete_sample3(const FloatGrid3 &prob_grid, int &dim1, int &dim2, int &dim3, int rnd_seed);
  void discrete_sample3(const FloatGrid3 &prob_grid, int n, 
                        std::vector<int> &sample_dim1, std::vector<int> &sample_dim2, std::vector<int> &sample_dim3, int rnd_seed);

}// namespace 


#endif
