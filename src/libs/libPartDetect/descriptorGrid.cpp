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
#include <libKMA2/gauss_iir/gauss_iir.h>
#include <libKMA2/kmaimagecontent.h>
#include <libKMA2/ShapeDescriptor.h>
#include <libKMA2/descriptor/feature.h>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/boost_math.hpp>
#include <libBoostMath/homogeneous_coord.h>

using boost_math::double_matrix;
using boost_math::double_vector;

using namespace std;

namespace part_detect { 

  /**
     compute desriptors on the uniform grid 

     grid.rect - define position, orientation and size of grid in the image
     
     desc_step - distance between descriptors
     desc_size - descriptor size
  */

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid)
  {
    initPatchMask(PATCH_SIZE);

    /* descriptor patch */
    DARY *patch = new DARY(PATCH_SIZE, PATCH_SIZE);

    /* descritor */
    int _ShapeSize = kma::shape::SrSize * kma::shape::ScSize * kma::shape::SOriSize;
    vector<float> ds(_ShapeSize);

    double patch_scale = (2*grid.desc_size + 1) / (float)PATCH_SIZE;

    DARY *source_image = input_image;
    DARY *smooth_input_image = NULL;

    /* smooth once for all descriptors if patch < descriptor */
    if (patch_scale > 1.0) {
      smooth_input_image = new DARY(input_image->y(), input_image->x());
      smooth(input_image, smooth_input_image, patch_scale);
      source_image = smooth_input_image;
    }

    boost_math::double_matrix patch_lcpos;
    boost_math::double_matrix patch_lrpos;

    kma::precompute_patch_idx(patch_lcpos, patch_lrpos, (int)floor(PATCH_SIZE/2), kma::shape::SrSize, kma::shape::ScSize);

    assert((int)grid.desc.size() == grid.ny);
    
    int pos_skipped = 0;
    
    for (int iy = 0; iy < grid.ny; ++iy) { 

      if (!((int)grid.desc[iy].size() == grid.nx)) {
        assert(false);
      }

      for (int ix = 0; ix < grid.nx; ++ix) {
        ds.assign(ds.size(), 0.0);
        grid.desc[iy][ix].clear();

        /* grid coordinates */
        double x2 = grid.rect.min_proj_x + ix*grid.desc_step;
        double y2 = grid.rect.min_proj_y + iy*grid.desc_step;
	
        /* image coordinates */
        boost_math::double_vector imgpos = grid.origin() + x2*grid.x_axis() + y2*grid.y_axis();
	
        /* test that descriptor is inside the image */
        if (imgpos(0) >= grid.desc_size + 1 && imgpos(0) <= source_image->x() - (grid.desc_size + 1) && 
            imgpos(1) >= grid.desc_size + 1 && imgpos(1) <= source_image->y() - (grid.desc_size + 1)) {

          /* map image region to patch */
          patch->interpolate(source_image, imgpos(0), imgpos(1), 
                             patch_scale*grid.x_axis()(0), patch_scale*grid.y_axis()(0), 
                             patch_scale*grid.x_axis()(1), patch_scale*grid.y_axis()(1));

          /* mean/var normalization, rescale with x = 128 + 50*x, clip to [0, 255] (last 3 parameters are not important) */
          normalize(patch, 0, 0, 0);
          //patch->writePNG(qsFilename.toStdString().c_str());

          assert(patch_lcpos.size1() > 0 && patch_lcpos.size2() > 0);

          /* compute SHAPEv5 */
          kma::KMAcomputeShape(patch_lcpos, patch_lrpos, patch, ds);

          grid.desc[iy][ix].insert(grid.desc[iy][ix].end(), ds.begin(), ds.end());
        }
      }
    }// grid positions

    delete smooth_input_image;
    delete patch;    
  }
  
}// namespace 
