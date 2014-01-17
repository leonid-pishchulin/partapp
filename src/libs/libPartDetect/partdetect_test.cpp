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

#include <libMisc/misc.hpp>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <libMatlabIO/matlab_cell_io.hpp>
#include <libMatlabIO/matlab_io.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libPartApp/partapp_aux.hpp>

#include <libMultiArray/multi_array_transform.hpp>

#include <libKMA2/kmaimagecontent.h>

#include <libPictStruct/objectdetect.h>

#include "partdetect.h"

//#include "Timer.h"

using namespace std;

using boost_math::double_matrix;
using boost_math::double_vector;

namespace part_detect {

  double get_runtime(void)
  {
    clock_t start;
    start = clock();
    return(((double)start*100.0/(double)CLOCKS_PER_SEC));
  }

  void partdetect_dense(const ExpParam &exp_param, const AbcDetectorParam &abc_param, const PartWindowParam &window_param, 
                        const AnnotationList &test_annolist, vector<AdaBoostClassifier> &v_abc,
                        QString qsScoreGridDir, 
                        int firstidx, int lastidx, bool flip, 
                        bool bSaveImageScoreGrid, 
			bool bAddImageBorder,
			std::vector<std::vector<FloatGrid3> > &part_detections, 
			bool bSaveScoreGrid) 
  {
    assert(firstidx >= 0 && firstidx <= (int)test_annolist.size());
    assert(lastidx < (int)test_annolist.size());

    uint nScales = exp_param.num_scale_steps();
    uint nRotations = exp_param.num_rotation_steps();
    uint nParts = v_abc.size();
    
    /* compute diagonal of the biggest part (needed to add extra border to the image) */
    /* it might be still impossible to evaluate classifier at some points since part  */
    /* center might not coinside with the center of the part bounding box             */
    double max_part_diag = 0;
    for (uint pidx = 0; pidx < nParts; ++pidx) {
      double cur_part_diag = sqrt(square(window_param.part(pidx).window_size_x()) + 
                                  square(window_param.part(pidx).window_size_y()));

      if (cur_part_diag > max_part_diag)
        max_part_diag = cur_part_diag;
    }
    
    bool bNormalizeBoost = true;
        
    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {

      /** 
	  addon: use person detection from elsewhere (e.g. tracking) to reduce the number 
	  of positions where we evaluate part detectors
      */
      FloatGrid3 detection_mask;

      if (exp_param.detection_masks_dir().length() > 0) {
	QString qsMaskFilename = exp_param.detection_masks_dir().c_str() + QString("/detection_mask_imgidx") + padZeros(QString::number(imgidx + 1), 5) + ".mat";

	FloatGrid3 _detection_mask = matlab_io::mat_load_multi_array<FloatGrid3>(qsMaskFilename, "detection_mask");
	detection_mask.resize(boost::extents[_detection_mask.shape()[0]][_detection_mask.shape()[1]][_detection_mask.shape()[2]]);
	detection_mask = _detection_mask;

	cout << "found detection mask in " << qsMaskFilename.toStdString() << endl;
      }

      cout << "computing features on image " << imgidx << endl;
      cout << "feature type: " << abc_param.feature_type() << endl;
      vector<vector<vector<FloatGrid2> > > vv_score_grid(nParts, vector<vector<FloatGrid2> > (nScales, vector<FloatGrid2>()));

      /* this is for scores backprojected to image coordinates */
      vector<vector<vector<FloatGrid2> > > vv_img_score_grid(nParts, vector<vector<FloatGrid2> > (nScales, vector<FloatGrid2>()));

      vector<FloatGrid4> transform_Ti2(nParts, FloatGrid4(boost::extents[nScales][nRotations][3][3]));
      vector<FloatGrid4> transform_T2g(nParts, FloatGrid4(boost::extents[nScales][nRotations][3][3]));    

      kma::ImageContent *kmaimg = kma::load_convert_gray_image(test_annolist[imgidx].imageName().c_str());
      int img_height = kmaimg->y();
      int img_width = kmaimg->x();
      
      /* compute descriptors for all scales and rotations, evaluate classifier */
      for (uint scaleidx = 0; scaleidx < nScales; ++scaleidx) {
	
        double scale = scale_from_index(exp_param, scaleidx);
        double desc_step = scale * abc_param.desc_step();
        int desc_size = boost_math::round(scale * abc_param.desc_size());
      
        /* add border such that classifier can be evaluated for all image positions,  */
        /* desc_size is radius of the feature, not diameter!                          */
        /* this border should be rotation dependent!!!                                */
        //int ext_border = (int)ceil(0.5*desc_size + 0.5*scale*max_part_diag);

        int ext_border = 0;
	
        if (bAddImageBorder)
          ext_border = (int)ceil(desc_size + 0.5*scale*max_part_diag);
	int corrected_border;
        kma::ImageContent *ext_kmaimg = add_image_border2(kmaimg, ext_border, corrected_border);
	ext_border = corrected_border;

	FloatGrid2 cur_detection_mask;
	
	if (detection_mask.shape()[0] > 0) {
	  //assert(scaleidx < detection_mask.shape()[0]);
	  int imgheight = detection_mask.shape()[1];
	  int imgwidth = detection_mask.shape()[2];

  	  cur_detection_mask.resize(boost::extents[imgheight + 2*ext_border][imgwidth + 2*ext_border]);
		 
	  typedef boost::multi_array_types::index_range range;
	  cur_detection_mask[ boost::indices[range(ext_border, ext_border + imgheight)][range(ext_border, ext_border + imgwidth)] ] = detection_mask[scaleidx];

	}

        double window_step;
        double grid_step;

        if (abc_param.window_desc_step_ratio() > 0) {
          window_step = abc_param.window_desc_step_ratio() * desc_step;

          if (window_step < 1)
            window_step = 1;

        }
        else {
          window_step = 1;
          //grid_step = desc_step / window_step;
        }

        grid_step = desc_step / window_step;

        cout << "desc_step: " << desc_step << 
          ", desc_size: " << desc_size << 
          ", window_step: " << window_step << 
          ", grid_step: " << grid_step << endl;
	
	for (uint rotidx = 0; rotidx < nRotations; ++rotidx) {
	  double rt_rot = get_runtime();
          double rotation = rot_from_index(exp_param, rotidx);

          cout << "scale: " << scale << ", rotation: " << rotation << ", flip: " << flip << endl;
          rotation *= M_PI / 180.0;        

          double_vector ax(2);
          double_vector ay(2);
          ax(0) = cos(rotation);
          ax(1) = sin(rotation);
          ay(0) = -sin(rotation);
          ay(1) = cos(rotation);

          if (flip) {
            ax(0) = -ax(0);
            ax(1) = -ax(1);
          }
	  
          FeatureGrid feature_grid(ext_kmaimg->x(), ext_kmaimg->y(), ax, ay, window_step, desc_size);
	  double rt2 = get_runtime();
	  computeDescriptorGrid(ext_kmaimg, feature_grid, abc_param.feature_type().c_str(), cur_detection_mask);
	  //cout << "descriptorGrid: " << (get_runtime()-rt2)/100 << endl;
	  rt2 = get_runtime();
	  /* evaluate classifiers of different parts */
	  
          for (uint pidx = 0; pidx < nParts; ++pidx) {
	    ScoreGrid score_grid;
	    computeScoreGrid(abc_param, window_param.part(pidx),
			     v_abc[pidx], grid_step, scale, feature_grid, score_grid, true, bNormalizeBoost);
	    
            double_matrix Ti_ei = hc::get_translation_matrix(-ext_border, -ext_border);
          
            //double_matrix Tig = prod(Ti_ei, score_grid.Tig);

            double_matrix Tig = prod(Ti_ei, score_grid.getTig());
            double_matrix Ti2 = prod(Ti_ei, score_grid.Ti2);
            double_matrix T2g = score_grid.T2g;
      
            vv_score_grid[pidx][scaleidx].push_back(score_grid.grid);
          
            for (uint i1 = 0; i1 < 3; ++i1)
              for (uint i2 = 0; i2 < 3; ++i2) {
                transform_T2g[pidx][scaleidx][rotidx][i1][i2] = T2g(i1, i2);
                transform_Ti2[pidx][scaleidx][rotidx][i1][i2] = Ti2(i1, i2);
              }

            /* map score grid to image coordinates */
            FloatGrid2 img_score_grid(boost::extents[kmaimg->y()][kmaimg->x()]);

            TransformationMethod transform_method = TM_BILINEAR;
            //TransformationMethod transform_method = TM_DIRECT;
            multi_array_op::transform_grid_fixed_size(score_grid.grid, img_score_grid, Tig,
                                                      NO_CLASS_VALUE, transform_method);

            //cout << "interpolating from grid to image, method: " << transform_method << endl;
            vv_img_score_grid[pidx][scaleidx].push_back(img_score_grid);
          }// parts
	  //cout << "parts scoreGrid: " << (get_runtime()-rt2)/100 << endl;
	  //cout << "single rot: " << (get_runtime()-rt_rot)/100 << endl;
        }// rotations

        delete ext_kmaimg;      
      }// scales
      
      delete kmaimg;

      cout << "bSaveScoreGrid: " << bSaveScoreGrid << endl;
      
      if (bSaveScoreGrid)  
	for (uint pidx = 0; pidx < nParts; ++pidx) {
	  QString qsFilename = qsScoreGridDir + "/imgidx" + QString::number(imgidx) + 
	    "-pidx" + QString::number(pidx) + "-o" + QString::number((int)flip) + "-scoregrid.mat";
	  
	  cout << "saving scoregrid and transformations to " << qsFilename.toStdString() << endl;
	  
	  MATFile *f = matlab_io::mat_open(qsFilename, "wz");
	  assert(f != 0);
	  
	  matlab_io::mat_save_multi_array_vec2(f, "cell_scoregrid", vv_score_grid[pidx]);
	  //multi_array_op::mat_save_multi_array_vec2(f, "cell_Tig", vv_transform[pidx]);
	  
	  if (bSaveImageScoreGrid)
	    matlab_io::mat_save_multi_array_vec2(f, "cell_img_scoregrid", vv_img_score_grid[pidx]);
	  
	  matlab_io::mat_save_multi_array(f, "transform_Ti2", transform_Ti2[pidx]);
	  matlab_io::mat_save_multi_array(f, "transform_T2g", transform_T2g[pidx]);
	  
	  if (f != 0)
	    matlab_io::mat_close(f);
	}
      else{
	assert(firstidx == lastidx);
	
	//copy scoregrid values
	//cout << img_height << endl;
	//cout << img_width << endl;
	part_detections.resize(nParts, vector<FloatGrid3>(nScales, 
							  FloatGrid3(boost::extents[nRotations][img_height][img_width])));
	for (uint pidx = 0; pidx < nParts; ++pidx) 
	  for (uint scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	    for (uint rotidx = 0; rotidx < nRotations; ++rotidx){ 	  
	  
	      double_matrix T2g(3,3);
	      double_matrix Ti2(3,3);

	      for (uint i1 = 0; i1 < 3; ++i1)
		for (uint i2 = 0; i2 < 3; ++i2) {
		  T2g(i1, i2) = transform_T2g[pidx][scaleidx][rotidx][i1][i2];
		  Ti2(i1, i2) = transform_Ti2[pidx][scaleidx][rotidx][i1][i2];
              }
	      
	      double_matrix Tig = prod(Ti2, T2g);
	      FloatGrid2 grid(boost::extents[img_height][img_width]);
	      multi_array_op::transform_grid_fixed_size(vv_score_grid[pidx][scaleidx][rotidx], grid, Tig, 
	      						part_detect::NO_CLASS_VALUE, TM_DIRECT);
	      for (int iy = 0; iy < img_height; ++iy) 
		for (int ix = 0; ix < img_width; ++ix)
		  part_detections[pidx][scaleidx][rotidx][iy][ix] = grid[iy][ix];
	      
	    }// rotidx
      }
    }// images
  }
    
  void partdetect(const PartApp &part_app, int firstidx, int lastidx, bool flip, bool bSaveImageScoreGrid, 
		  std::vector<std::vector<FloatGrid3> > &part_detections, const AnnotationList &annolist,
		  QString qsScoreGridDir, bool bSaveScoreGrid)
  {
    //AnnotationList annolist = part_app.m_test_annolist;
    //QString qsScoreGridDir = part_app.m_exp_param.scoregrid_dir().c_str();
    
    if (!filesys::check_dir(qsScoreGridDir)) {
      cout << "creating " << qsScoreGridDir.toStdString() << endl;
      if (!filesys::create_dir(qsScoreGridDir))
	cout << "WARNING: dir wasn't created!" << endl;
    }
    
    uint nParts = part_app.m_part_conf.part_size();
    
    /* load classifiers */
    vector<AdaBoostClassifier> v_abc;
    for (uint pidx = 0; pidx < nParts; ++pidx) {

      if (part_app.m_part_conf.part(pidx).is_detect()) {
        if (!part_app.m_part_conf.part(pidx).has_mult_types()) {
	  AdaBoostClassifier abc;
	  part_app.loadClassifier(abc, pidx);
	  v_abc.push_back(abc);
	}
	else{
	  int nTypes = getNumPartTypes(part_app.m_window_param, pidx);
	  //cout << "pidx = " << pidx << ", ntypes: " << nTypes << endl;
	  for(uint tidx = 0; tidx < nTypes; ++tidx){
	    AdaBoostClassifier abc;
	    int boostrap_type = 2;
	    part_app.loadClassifier(abc, pidx, boostrap_type, tidx);
	    v_abc.push_back(abc);
	  }
	}
      }
    }
    
    bool bAddImageBorder = true;
    
    assert(!part_app.m_bExternalScoregridDir && "can not update external scoregrids");
    
    partdetect_dense(part_app.m_exp_param, part_app.m_abc_param, part_app.m_window_param, 
		     annolist, v_abc, qsScoreGridDir, firstidx, lastidx, flip, bSaveImageScoreGrid, 
		     bAddImageBorder, part_detections, bSaveScoreGrid);
    
  }

  void computeScoreGrid(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param,
                        const AdaBoostClassifier &abc, double grid_step, double part_scale, 
                        FeatureGrid &feature_grid, ScoreGrid &score_grid, 
                        bool bSqueeze, bool bNormalizeBoost)
  {
    //  cout << "computeScoreGrid" << endl;
    //double rt1 = get_runtime();
    assert(abc_param.window_desc_step_ratio() > 0 && abc_param.window_desc_step_ratio() <= 1);
    assert(grid_step > 0);

    /* get how many features fit into part window */
    int grid_x_count, grid_y_count;
    get_window_feature_counts(abc_param, part_window_param, grid_x_count, grid_y_count);
    assert(grid_x_count > 0 && grid_y_count > 0);

    int last_ix = feature_grid.last_valid_grid_pos_x(grid_x_count, grid_step);
    int last_iy = feature_grid.last_valid_grid_pos_y(grid_y_count, grid_step);

    //   cout << "grid dimensions: ny: " << feature_grid.ny << " " << ", nx: " << feature_grid.nx << endl;
    //   cout << "window dimensions: " << grid_y_count << " " << grid_x_count << endl;
    //   cout << "last valid position: " << last_iy << " " << last_ix << endl;
  
    score_grid.grid.resize(boost::extents[feature_grid.ny][feature_grid.nx]);
    for (int iy = 0; iy <= last_iy; ++iy)
      for (int ix = 0; ix <= last_ix; ++ix) {
        vector<float> all_features;
        if (feature_grid.concatenate(ix, iy, grid_x_count, grid_y_count, grid_step, all_features)) { 
          //cout << "ix: " << ix << ", iy: " << iy << endl;
          score_grid.grid[iy][ix] = abc.evaluateFeaturePoint(all_features, bNormalizeBoost);
        }
        else {
	  score_grid.grid[iy][ix] = NO_CLASS_VALUE;
        }
      }
    
    double_matrix T3g = hc::get_scaling_matrix(feature_grid.desc_step);
    T3g(0, 2) = part_scale * part_window_param.pos_offset_x();
    T3g(1, 2) = part_scale * part_window_param.pos_offset_y();

    double_matrix T32 = hc::get_translation_matrix(-feature_grid.rect.min_proj_x, -feature_grid.rect.min_proj_y);
    double_matrix T23 = hc::inverse(T32);

    double_matrix Ti2 = boost_math::zero_double_matrix(3, 3);
    Ti2(0, 2) = feature_grid.rect.part_pos(0);
    Ti2(1, 2) = feature_grid.rect.part_pos(1);

    Ti2(0, 0) = feature_grid.rect.part_x_axis(0);
    Ti2(1, 0) = feature_grid.rect.part_x_axis(1);

    Ti2(0, 1) = feature_grid.rect.part_y_axis(0);
    Ti2(1, 1) = feature_grid.rect.part_y_axis(1);

    Ti2(2, 2) = 1;

    score_grid.Ti2 = Ti2;
    score_grid.T2g = prod(T23, T3g);

    if (bSqueeze)
      squeezeScoreGrid(score_grid);
    //cout << (get_runtime()-rt1)/100 << endl;  
    //cout << "done." << endl;
  }

  void squeezeScoreGrid(ScoreGrid &score_grid)
  {
    //cout << "squeezeScoreGrid" << endl;

    int grid_width = score_grid.grid.shape()[1];
    int grid_height = score_grid.grid.shape()[0];

    int minx = grid_width;
    int maxx = -1;
    int miny = grid_height;
    int maxy = -1;

    for (int ix = 0; ix < grid_width; ++ix) 
      for (int iy = 0; iy < grid_height; ++iy) {
        if (score_grid.grid[iy][ix] != NO_CLASS_VALUE) {
          if (ix < minx)
            minx = ix;

          if (ix > maxx)
            maxx = ix;

          if (iy < miny)
            miny = iy;

          if (iy > maxy)
            maxy = iy;
        }
      }

    /** now we can have empty grids because of the detection_mask */
    if (maxx > minx && maxy > miny) {
      //assert(maxx > minx && maxy > miny && "empty grid");

      int new_grid_width = maxx - minx + 1;
      int new_grid_height = maxy - miny + 1;

      FloatGrid2 new_grid(boost::extents[new_grid_height][new_grid_width]);

      boost_math::double_matrix Tg4 = hc::get_translation_matrix(minx, miny);
      boost_math::double_matrix T4g = hc::get_translation_matrix(-minx, -miny);

      multi_array_op::transform_grid_fixed_size(score_grid.grid, new_grid, T4g, 
						NO_CLASS_VALUE, TM_DIRECT);

      score_grid.grid.resize(boost::extents[new_grid_height][new_grid_width]);
      score_grid.grid = new_grid;

      boost_math::double_matrix M = prod(score_grid.T2g, Tg4);
      score_grid.T2g = M; 
    }
    else {
      // TODO: what to do if the grid is empty? do something smart here

      //cout << "empty scoregrid " << endl;

    }

    //cout << "squeezeScoreGrid - done" << endl;
    
  }


}// namespace 
