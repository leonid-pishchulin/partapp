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

#ifndef _OBJECTDETECT_AUX_HPP_
#define _OBJECTDETECT_AUX_HPP_

namespace object_detect {
  
  enum objType {
    OBJ_TEST = 0,
    OBJ_TRAIN_POS = 1,
    OBJ_TRAIN_NEG = 2,
    OBJ_TRAIN_HARD = 3
  };

  /** 
      map negative classifier scores to small positive values

      Assume that classifier was not evaluated at locations with score == 0.

      It is important that scores are computed on sparse grid, otherwise marginals at different scales 
      will become uncomparable (spatial uncertainty grows with scale and we currently use unnormalized Gaussians
      for marginalization).

      Setting the minimum score to small positive value slightly improves the results over simply setting 
      it to 0.
  */
  template <typename Array>
  void clip_scores_fill(Array &grid, double min_val = 0.0001) {
    
    uint nElements = grid.num_elements();
    typename Array::element *pData = grid.data();

    int set_count = 0;
    //DEBUG
    //min_val = 0.1;
    for (uint i3 = 0; i3 < nElements; ++i3) {
      //if (pData[i3] < min_val) {
      
      if (pData[i3] < 0) {	
        pData[i3] = min_val;
        ++set_count;
      }
      
    }     
  }

}// namespace

#endif
