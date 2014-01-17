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

#ifndef _DISC_PS_H_
#define _DISC_PS_H_

#include <vector>

#include <libMultiArray/multi_array_def.h>

#include <libPartApp/partapp.h>


namespace disc_ps {

  /**
     disc_sample.cpp
  */
  void flatten_array(const FloatGrid4 &grid4, FloatGrid1 &grid1);
  void flatten_array(const FloatGrid3 &grid3, FloatGrid1 &grid1); 
  void flatten_array(const FloatGrid2 &grid2, FloatGrid1 &grid1);

  void index_from_flat(const std::vector<int> shape, int flat_idx, std::vector<int> &idx);
  void index_from_flat2(int shape0, int shape1, int flat_idx, int &idx1, int &idx2);
  void index_from_flat3(int shape0, int shape1, int shape2, int flat_idx, int &idx1, int &idx2, int &idx3);
  void index_from_flat4(int shape0, int shape1, int shape2, int shape3, int flat_idx, int &idx1, int &idx2, int &idx3, int &idx4);

/*   void discrete_sample3(const FloatGrid3 &prob_grid, int n,  */
/*                         std::vector<int> &sample_dim1, std::vector<int> &sample_dim2, std::vector<int> &sample_dim3,  */
/* 			int rnd_seed); */


  /** 
      disc_ps.cpp
   */
  
  void partSample(const PartApp &part_app, int firstidx, int lastidx);
  void partSampleWithPrior(const PartApp &part_app, int firstidx, int lastidx);

  void findObjDai(const PartApp &part_app, int firstidx, int lastidx, bool bForceRecompute);

  void visSamples(const PartApp &part_app, int firstidx, int lastidx);
}


#endif
