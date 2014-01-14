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

#include <libAdaBoost/AdaBoost.h>

#include <libKMA2/gauss_iir/gauss_iir.h>
#include <libKMA2/kmaimagecontent.h>
#include <libKMA2/ShapeDescriptor.h>
#include <libKMA2/descriptor/feature.h>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/boost_math.hpp>
#include <libBoostMath/homogeneous_coord.h>

#include <libMisc/misc.hpp>
#include <libMatlabIO/matlab_io.h>

#include <libPartApp/partapp.h>

#include "partdetect.h"
#include "partdef.h"

using boost_math::double_matrix;
using boost_math::double_vector;

using namespace std;

namespace part_detect { 

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC)
  {
    FloatGrid2 detection_mask;
    QRect roi(0, 0, input_image->x() - 1, input_image->y() - 1);

    computeDescriptorGrid(input_image, grid, qsDESC, detection_mask, roi);
  }

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC, const FloatGrid2 &detection_mask)
  {
    QRect roi(0, 0, input_image->x() - 1, input_image->y() - 1);

    computeDescriptorGrid(input_image, grid, qsDESC, detection_mask, roi);
  }

  void computeDescriptorGridRoi(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC, const QRect& roi)
  {
    FloatGrid2 detection_mask;

    computeDescriptorGrid(input_image, grid, qsDESC, detection_mask, roi);
  }


  /**
     compute desriptors on the uniform grid 

     grid.rect - define position, orientation and size of grid in the image
     
     desc_step - distance between descriptors
     desc_size - descriptor size

     roi - region of interest
  */

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC, 
			     const FloatGrid2 &detection_mask, const QRect& roi)
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

    assert(qsDESC == "SHAPEv5" && "unknown descriptor type");

    kma::precompute_patch_idx(patch_lcpos, patch_lrpos, (int)floor(PATCH_SIZE/2), kma::shape::SrSize, kma::shape::ScSize);

    assert((int)grid.desc.size() == grid.ny);

    if (detection_mask.shape()[0] > 0) {
      assert(detection_mask.shape()[0] == source_image->y() && detection_mask.shape()[1] == source_image->x());
    }

    int pos_skipped = 0;
    int pos_skipped_roi = 0;

    bool bTestRoi;

    if (roi.left() == 0 && roi.right() == source_image->x() - 1 && roi.top() == 0 && roi.bottom() == source_image->y() - 1) 
      bTestRoi = false;
    else
      bTestRoi = true;

    // debug 
    // assert(bTestRoi);
    
    for (int iy = 0; iy < grid.ny; ++iy) { 
      //assert((int)grid.desc[iy].size() == grid.nx);

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

	  /* test that image point is on the detection mask */
 	  if (detection_mask.shape()[0] > 0) {
	    if (detection_mask[imgpos(1)][imgpos(0)] == 0) {
	      ++pos_skipped;
	      continue;
	    }
 	  }
	  
	  /* test ROI */
	  if (bTestRoi) {
	    if (!(imgpos(0) >= roi.left() && imgpos(0) <= roi.right() && imgpos(1) >= roi.top() && imgpos(1) <= roi.bottom())) {
	      ++pos_skipped_roi;
	      continue;
	    }	      
	  }
	  
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

    if (detection_mask.shape()[0] > 0) {
      cout << "pos_skipped: " << pos_skipped << ", pos_total: " << grid.ny * grid.nx << endl;;
      cout << "pos_skipped_roi: " << pos_skipped_roi << ", pos_total: " << grid.ny * grid.nx << endl;;
    }
    delete smooth_input_image;
    delete patch;    
  }

  void getDescCoord(FeatureGrid &grid, FloatGrid2 &imgposGrid)
  {
    assert((int)grid.desc.size() == grid.ny);

    imgposGrid.resize(boost::extents[grid.nx*grid.ny][4]);

    for (int iy = 0; iy < grid.ny; ++iy) { 

      if (!((int)grid.desc[iy].size() == grid.nx)) {
        assert(false);
      }

      for (int ix = 0; ix < grid.nx; ++ix) {

        /* grid coordinates */
        double x2 = grid.rect.min_proj_x + ix*grid.desc_step;
        double y2 = grid.rect.min_proj_y + iy*grid.desc_step;
	
        /* image coordinates */
        boost_math::double_vector imgpos = grid.origin() + x2*grid.x_axis() + y2*grid.y_axis();
	
	imgposGrid[ix + grid.nx*iy][0] = imgpos(0);
	imgposGrid[ix + grid.nx*iy][1] = imgpos(1);
	imgposGrid[ix + grid.nx*iy][2] = ix;
	imgposGrid[ix + grid.nx*iy][3] = iy;
      }
    }// grid positions
  }

  int get_app_group_idx(const PartConfig &part_conf, int pid)
  {
    for (int agidx = 0; agidx < part_conf.app_group_size(); ++agidx) 
      for (int idx = 0; idx < part_conf.app_group(agidx).part_id_size(); ++idx) {
        if (part_conf.app_group(agidx).part_id(idx) == pid)
          return agidx;
      }

    return -1;
  }

  int pidx_from_pid(const PartConfig &part_conf, int pid)
  {
    for (int pidx = 0; pidx < part_conf.part_size(); ++pidx) {
      assert(part_conf.part(pidx).has_part_id());

      if (part_conf.part(pidx).part_id() == pid)
        return pidx;
    }

    assert(false && "part id not found");
    return -1;
  }

  void get_window_feature_counts(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param, 
                                 int &grid_x_count, int &grid_y_count)
  {
    grid_x_count = part_window_param.window_size_x() / abc_param.desc_step() + 1;
    grid_y_count = part_window_param.window_size_y() / abc_param.desc_step() + 1;
  }
  
  void sample_random_partrect(boost::variate_generator<boost::mt19937, boost::uniform_real<> > &gen_01,
                              double min_scale, double max_scale,
                              double min_rot, double max_rot, 
                              double min_pos_x, double max_pos_x,
                              double min_pos_y, double max_pos_y,
                              int rect_width, int rect_height, 
                              PartBBox &partrect, double &scale, double &rot)
  {
    scale = min_scale + (max_scale - min_scale)*gen_01();
    rot = min_rot + (max_rot - min_rot)*gen_01();

    partrect.part_pos(0) = min_pos_x + (max_pos_x - min_pos_x)*gen_01();
    partrect.part_pos(1) = min_pos_y + (max_pos_y - min_pos_y)*gen_01();
    //cout << alpha << ", " << scale << ", " << partrect.part_pos(0) <<  ", " << partrect.part_pos(1) << endl;

    partrect.part_x_axis(0) = cos(rot);
    partrect.part_x_axis(1) = sin(rot);

    partrect.part_y_axis(0) = -partrect.part_x_axis(1);
    partrect.part_y_axis(1) = partrect.part_x_axis(0);

    rect_width = (int)(rect_width*scale);
    rect_height = (int)(rect_height*scale);

    partrect.min_proj_x = -floor(rect_width/2);
    partrect.max_proj_x = partrect.min_proj_x + rect_width - 1;
  
    partrect.min_proj_y = -floor(rect_height/2);
    partrect.max_proj_y = partrect.min_proj_y + rect_height - 1;
  }
  
  
}// namespace 
