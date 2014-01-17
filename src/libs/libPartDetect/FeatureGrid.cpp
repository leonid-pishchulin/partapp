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

#include "FeatureGrid.h"

using std::vector;


FeatureGrid::FeatureGrid(int img_width, int img_height, 
                         boost_math::double_vector x_axis, boost_math::double_vector y_axis, 
                         double _desc_step, int _desc_size):desc_step(_desc_step), desc_size(_desc_size)
{
  assert(img_width > 0 && img_height > 0);
  assert(x_axis.size() == 2 && y_axis.size() == 2);

  rect.part_x_axis = x_axis;
  rect.part_y_axis = y_axis;

  rect.part_pos = boost_math::zero_double_vector(2);

  /* find bounding box in the rotated space */
  boost_math::double_matrix T12(2,2);
  column(T12, 0) = x_axis;
  column(T12, 1) = y_axis;
  boost_math::double_matrix T21(2, 2);
  boost_math::inv(T12, T21);

  boost_math::double_matrix corners = boost_math::zero_double_matrix(2, 4);
  corners(0, 0) = 0;              
  corners(1, 0) = 0;

  corners(0, 1) = img_width;
  corners(1, 1) = 0;

  corners(0, 2) = img_width;
  corners(1, 2) = img_height;

  corners(0, 3) = 0;
  corners(1, 3) = img_height;

  boost_math::double_matrix corners2 = prod(T21, corners);
  assert(corners2.size1() == 2 && corners2.size2() == 4);

  rect.min_proj_x = corners2(0, 0);
  rect.max_proj_x = corners2(0, 0);
  rect.min_proj_y = corners2(1, 0);
  rect.max_proj_y = corners2(1, 0);

  for (int i = 1; i < 4; ++i) {
    if (rect.min_proj_x > corners2(0, i))
      rect.min_proj_x = corners2(0, i);

    if (rect.max_proj_x < corners2(0, i))
      rect.max_proj_x = corners2(0, i);

    if (rect.min_proj_y > corners2(1, i))
      rect.min_proj_y = corners2(1, i);

    if (rect.max_proj_y < corners2(1, i))
      rect.max_proj_y = corners2(1, i);
  }

  /* same as in the other constructor */
  init();
}


FeatureGrid::FeatureGrid(const PartBBox &_rect, double _desc_step, int _desc_size):rect(_rect),
                                                                   desc_step(_desc_step),
                                                                   desc_size(_desc_size) 
{
  init();
}

void FeatureGrid::init()
{
  /* bounding box in grid coordinate system */
  double mx = rect.min_proj_x;
  double Mx = rect.max_proj_x;

  double my = rect.min_proj_y;
  double My = rect.max_proj_y;
    
  /* grid size */
  nx = (int)floor(((Mx - mx + 1) / desc_step)) + 1;
  ny = (int)floor(((My - my + 1) / desc_step)) + 1;

  desc.resize(ny, vector<vector<float> >(nx, vector<float>()));
}

// bool FeatureGrid::concatenate(int grid_pos_x, int grid_pos_y, 
//                               int grid_x_count, int grid_y_count, 
//                               double grid_step, vector<float> &allfeatures)
// {
//   assert(grid_pos_x >= 0 && grid_pos_x + boost_math::round(grid_step*(grid_x_count - 1)) < nx);
//   assert(grid_pos_y >= 0 && grid_pos_y + boost_math::round(grid_step*(grid_y_count - 1)) < ny);
//   allfeatures.clear();

//   // quickly test if all features are available
//   for (int iy = 0; iy < grid_y_count; ++iy) 
//     for (int ix = 0; ix < grid_x_count; ++ix) {

//       int x = grid_pos_x + boost_math::round(ix*grid_step);
//       int y = grid_pos_y + boost_math::round(iy*grid_step);
//       if (desc[y][x].size() == 0)
//         return false;
//     }

//   // concatenate feature vectors
//   for (int iy = 0; iy < grid_y_count; ++iy) 
//     for (int ix = 0; ix < grid_x_count; ++ix) {
//       int x = grid_pos_x + boost_math::round(ix*grid_step);
//       int y = grid_pos_y + boost_math::round(iy*grid_step);
      
//       allfeatures.insert(allfeatures.begin(), desc[y][x].begin(), desc[y][x].end());
//     }
  
//   return true;
// }

/*
  Michael: sped-up version of concatenate(.)
*/
bool FeatureGrid::concatenate(int grid_pos_x, int grid_pos_y, 
                              int grid_x_count, int grid_y_count, 
                              double grid_step, vector<float> &allfeatures)
{
  assert(grid_pos_x >= 0 && grid_pos_x + boost_math::round(grid_step*(grid_x_count - 1)) < nx);
  assert(grid_pos_y >= 0 && grid_pos_y + boost_math::round(grid_step*(grid_y_count - 1)) < ny);
  assert(allfeatures.size() == 0);
  //allfeatures.clear();

  // quickly test if all features are available (also find the length of descriptor)
  int descSize = -1;

  for (int iy = 0; iy < grid_y_count; ++iy) 
    for (int ix = 0; ix < grid_x_count; ++ix) {

      int x = grid_pos_x + boost_math::round(ix*grid_step);
      int y = grid_pos_y + boost_math::round(iy*grid_step);
      if (desc[y][x].size() == 0){
	//std::cout << "desc[" << y << "][" << x << "].size() == 0" << std::endl;
        return false;
      }
      if (descSize == -1 && desc[y][x].size() > 0) {
        descSize = desc[y][x].size();
      }
    }

  assert(descSize > 0);
  const int allFeaturesSize = grid_x_count * grid_y_count * descSize;
  allfeatures.resize(allFeaturesSize);

  vector<float>::iterator it = allfeatures.end();

  // concatenate feature vectors
  for (int iy = 0; iy < grid_y_count; ++iy) 
    for (int ix = 0; ix < grid_x_count; ++ix) {
      int x = grid_pos_x + boost_math::round(ix*grid_step);
      int y = grid_pos_y + boost_math::round(iy*grid_step);
      
      it -= descSize;

      std::copy(desc[y][x].begin(), desc[y][x].end(), it);
    }

  return true;
}


