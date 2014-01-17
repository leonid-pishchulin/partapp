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

#ifndef _FEATURE_GRID_H_
#define _FEATURE_GRID_H_

#include <vector>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/boost_math.hpp>

#include <libMultiArray/multi_array_def.h>

#include <libPartDetect/partdef.h>

class ScoreGrid {
 public:

  ScoreGrid(): Ti2(3, 3), T2g(3, 3) { 
    Ti2 = boost_math::identity_double_matrix(3);
    T2g = boost_math::identity_double_matrix(3);
  }

  FloatGrid2 grid;
  boost_math::double_matrix getTig() {return prod(Ti2, T2g);}

  boost_math::double_matrix Ti2; //
  boost_math::double_matrix T2g;
};

class FeatureGrid 
{
public:
  FeatureGrid(int img_width, int img_height, 
	      boost_math::double_vector x_axis, boost_math::double_vector y_axis, 
	      double _desc_step, int _desc_size);

  FeatureGrid(const PartBBox &_rect, double _desc_step, int _desc_size);

  void init();

  bool concatenate(int grid_pos_x, int grid_pos_y, 
		   int grid_x_count, int grid_y_count, 
		   double grid_step, std::vector<float> &allfeatures);

  inline const boost_math::double_vector &origin() {return rect.part_pos;}
  inline const boost_math::double_vector &x_axis() {return rect.part_x_axis;}
  inline const boost_math::double_vector &y_axis() {return rect.part_y_axis;}

  /**
     grid_x_count - number of descriptors needed for complete object window feature
     grid_step - distance between relevant descriptors on the grid

     return index of last descriptor for which complete object window feature can be computed by
     concatenation of individual features with step "grid_step"
   */

  int last_valid_grid_pos_x(int grid_x_count, double grid_step) 
  {return nx - boost_math::round(grid_step*(grid_x_count - 1)) - 1;};

  int last_valid_grid_pos_y(int grid_y_count, double grid_step) 
  {return ny - boost_math::round(grid_step*(grid_y_count - 1)) - 1;};    

  /* definition of grid coordinate system */
  PartBBox rect;

  /* distance between descriptors on the grid*/
  double desc_step;

  /* size of descriptor in pixels */
  int desc_size;

  /* grid size */
  int nx;
  int ny;

  /* descriptors, empty if outside of the image */
  std::vector<std::vector< std::vector<float> > > desc;
};

#endif
