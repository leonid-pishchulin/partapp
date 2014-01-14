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

#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_transform.hpp>
#include <libMultiArray/multi_array_filter.hpp>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <libPartApp/partapp_aux.hpp>
#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartEval/parteval.h>

#include <libPartDetect/partdetect.h>

#include <libKMA2/kmaimagecontent.h>

#include "objectdetect_aux.hpp"
#include "objectdetect.h"

using namespace std;
using boost_math::double_vector;
using boost::multi_array_types::index_range;

namespace object_detect {
  
  void findObjectRoiHelper(PartApp part_app, AnnoRect &rect, QString qsImageName,
  			   const std::vector<AdaBoostClassifier> &v_abc,
  			   std::vector<Joint> joints,
  			   std::vector<std::vector<PartHyp> > &best_part_det,
  			   std::vector<std::vector<PartHyp> > &best_part_hyp,
			   QString qsScoreGridDir,
			   int imgidx, bool bRandConf)
  {

    const ExpParam &exp_param = part_app.m_exp_param;
    const AbcDetectorParam &abc_param = part_app.m_abc_param;
    const PartWindowParam &window_param = part_app.m_window_param;
    const PartConfig &part_conf = part_app.m_part_conf;

    uint nRotations = exp_param.num_rotation_steps();
    int nParts = part_conf.part_size();
    uint nScales = 1;
    int nJoints = part_conf.joint_size();
    bool flip = false;

    int roi_x1 = rect.m_x1;
    int roi_x2 = rect.m_x2;

    int roi_y1 = rect.m_y1;
    int roi_y2 = rect.m_y2;

    double scale;
    if (rect.scale() > 0) {
      scale = rect.scale();
    }
    else {
      scale = abs(roi_y2 - roi_y1) / window_param.train_object_height();
    }
    
    scale = 2.0;
    
    /** hacky way to set scale used during inference */
    part_app.m_exp_param.set_min_object_scale(scale);
    part_app.m_exp_param.set_max_object_scale(scale);
    part_app.m_exp_param.set_num_scale_steps(1);

    double desc_step = scale * abc_param.desc_step();
    int desc_size = boost_math::round(scale * abc_param.desc_size());

    double window_step;
    double grid_step;

    if (abc_param.window_desc_step_ratio() > 0) {
      window_step = abc_param.window_desc_step_ratio() * desc_step;

      if (window_step < 1)
	window_step = 1;
    }
    else {
      window_step = 1;
    }

    grid_step = desc_step / window_step;

    cout << "desc_step: " << desc_step << 
      ", desc_size: " << desc_size << 
      ", window_step: " << window_step << 
      ", grid_step: " << grid_step << endl;

    /**
       compute diagonal of the biggest part (needed to add extra border to the image) 
       it might be still impossible to evaluate classifier at some points since part  
       center might not coinside with the center of the part bounding box             
    */
    double max_part_diag = 0;
    for (int pidx = 0; pidx < nParts; ++pidx) {
      double cur_part_diag = sqrt(square(window_param.part(pidx).window_size_x()) + 
				  square(window_param.part(pidx).window_size_y()));

      if (cur_part_diag > max_part_diag)
	max_part_diag = cur_part_diag;
    }

    /** extend the roi in order to enable part detection at the border */
    int ext_border_part = (int)ceil(desc_size + 0.5*scale*max_part_diag);

    kma::ImageContent *kmaimg = kma::load_convert_gray_image(qsImageName.toStdString().c_str());

    int img_width = kmaimg->x();
    int img_height = kmaimg->y();

    int ext_border_x = ext_border_part;
    int ext_border_y = ext_border_part;

    if (exp_param.has_roi_extend_x())
      ext_border_x = scale*exp_param.roi_extend_x();
      
    if (exp_param.has_roi_extend_y())
      ext_border_y = scale*exp_param.roi_extend_y();

    cout << "ext_border_x: " << ext_border_x << endl;
    cout << "ext_border_x: " << ext_border_y << endl;

    roi_x1 = max(roi_x1 - ext_border_x, 0);
    roi_x2 = min(roi_x2 + ext_border_x, img_width - 1);

    roi_y1 = max(roi_y1 - ext_border_y, 0);
    roi_y2 = min(roi_y2 + ext_border_y, img_height - 1);

    int roi_width = abs(roi_x2 - roi_x1) + 1;
    int roi_height = abs(roi_y2 - roi_y1) + 1;
  
    cout << "\tregion of interest: " << roi_x1 << ", " << roi_y1  << ", " << roi_x2 << ", " << roi_y2 << ", scale: " << scale << endl;

    // part, scale, rotation, y, x
    vector<vector<FloatGrid3> > log_part_detections(nParts, 
						    vector<FloatGrid3>(nScales, 
								       FloatGrid3(boost::extents[nRotations][roi_height][roi_width])));
    
    bool bInterpolate = false;
    
    QString qsFilename;
    QString qsVarName;
    
    int scaleidx = 0;

    for (uint rotidx = 0; rotidx < nRotations; ++rotidx) {
      double rotation = rot_from_index(exp_param, rotidx);
      
      cout << "scale: " << scale << ", rotation: " << rotation << endl;
      rotation *= M_PI / 180.0;        
      
      double_vector ax(2);
      double_vector ay(2);
      ax(0) = cos(rotation);
      ax(1) = sin(rotation);
      ay(0) = -sin(rotation);
      ay(1) = cos(rotation);
      
      /** compute features */
      FeatureGrid grid(roi_width, roi_height, ax, ay, window_step, desc_size);
      
      /** take position of roi in the image into account */
      grid.rect.part_pos(0) = roi_x1;
      grid.rect.part_pos(1) = roi_y1;
      
      QRect roi(roi_x1, roi_y1, roi_width, roi_height);
      
      part_detect::computeDescriptorGridRoi(kmaimg, grid, abc_param.feature_type().c_str(), roi);
      
      /** set back to zero so that we backproject from grid into the roi rectangle, take care of the offset later */
      grid.rect.part_pos(0) = 0;
      grid.rect.part_pos(1) = 0;
      
      /** compute part scores */
      for (int pidx = 0; pidx < nParts; ++pidx) {
	
	ScoreGrid score_grid;
	part_detect::computeScoreGrid(abc_param, window_param.part(pidx), 
				      v_abc[pidx], grid_step, scale, grid, score_grid);         
	
	double ext_border = 0;
	double_matrix Ti_ei = hc::get_translation_matrix(-ext_border, -ext_border);
	double_matrix Tig = prod(Ti_ei, score_grid.getTig());
	
	//FloatGrid2 img_score_grid(boost::extents[roi_height][roi_width]);
	
	//TransformationMethod transform_method = TM_BILINEAR;
	TransformationMethod transform_method = TM_DIRECT;
	
	FloatGrid2 img_score_grid(boost::extents[roi_height][roi_width]);
	multi_array_op::transform_grid_fixed_size(score_grid.grid, img_score_grid, Tig,
						  part_detect::NO_CLASS_VALUE, transform_method);
	
	object_detect::clip_scores_fill(img_score_grid);
	/** 
	    TODO: remove detections on the extended border, otherwise pose estimate might be outside of the ROI
	*/
	
	// ...
	
	log_part_detections[pidx][scaleidx][rotidx] = img_score_grid;      
	//       QString qsSaveFilename = qsDebugDir + "/test_grid_pidx" + QString::number(pidx) + "_" + QString::number(rotidx) + ".mat";
	//       matlab_io::mat_save_multi_array(qsSaveFilename , "img_score_grid", img_score_grid);
	
      }// parts
    } // rotations
    
    /** extract maxima of part detections */
    for (int pidx = 0; pidx < nParts; ++pidx) {
      std::vector<PartHyp> part_hyp;
      findLocalMax(exp_param, log_part_detections[pidx][scaleidx], part_hyp, exp_param.roi_save_num_samples());

      /** add roi offset */
      for (int idx = 0; idx < (int)part_hyp.size(); ++idx) {
	/** add roi offset */
	part_hyp[idx].m_x += roi_x1;
	part_hyp[idx].m_y += roi_y1;
      }

      best_part_det.push_back(part_hyp);
    }
    
    /** convert to log scores as expected in computeRootPosteriorRot  */
    for (int pidx = 0; pidx < nParts; ++pidx) {	
      multi_array_op::computeLogGrid(log_part_detections[pidx][scaleidx]);
    }
    
    /** inference */
    FloatGrid3 root_part_posterior;
    bool bIsSparse = true;
    vector<vector<FloatGrid3> > log_part_detections_orig = log_part_detections;
    boost_math::double_matrix rot_params;
    boost_math::double_matrix pos_params;
    boost_math::double_matrix pos_prior_params;
    boost_math::double_vector rootpos_det;
    
    bool bSaveMarginals = false;
    
    std::vector<PartHyp> _best_part_hyp;
    
    /** imgidx is only needed to save part_marginals, which we don't do here anyway */
    object_detect::computeRootPosteriorRot(part_app, log_part_detections, root_part_posterior, 
					   part_app.m_rootpart_idx, joints, flip, bIsSparse, imgidx, 
					   best_part_hyp, bSaveMarginals);
    
    /** add roi offset */
    for (int pidx = 0; pidx < (int)best_part_hyp.size(); ++pidx) {
      //_best_part_hyp.push_back(best_part_hyp[pidx][0]);
      for (int idx = 0; idx < (int)best_part_hyp[pidx].size(); ++idx) {
	best_part_hyp[pidx][idx].m_x += roi_x1;
	best_part_hyp[pidx][idx].m_y += roi_y1;
      }
      _best_part_hyp.push_back(best_part_hyp[pidx][0]);
    }
    
    assert((int)best_part_hyp.size() == nParts);
    
  }

  void findObjectImageRoi(PartApp part_app, int imgidx)
  {
    const ExpParam &exp_param = part_app.m_exp_param;
    const AbcDetectorParam &abc_param = part_app.m_abc_param;
    const PartWindowParam &window_param = part_app.m_window_param;
    const PartConfig &part_conf = part_app.m_part_conf;

    uint nRotations = exp_param.num_rotation_steps();
    int nParts = part_conf.part_size();
    uint nScales = 1;
    int nJoints = part_conf.joint_size();
    bool flip = false;
    
    AnnotationList roi_annolist;

    QString qsPartMarginalsDir = (exp_param.log_dir() + "/" + exp_param.log_subdir() + "/part_marginals_roi").c_str();

    if (!filesys::check_dir(qsPartMarginalsDir))
      filesys::create_dir(qsPartMarginalsDir);    

    assert(part_app.m_exp_param.roi_annolist().size() > 0);
    roi_annolist.load(exp_param.roi_annolist());

    assert((int)roi_annolist.size() > imgidx);
    assert(roi_annolist[imgidx].size() > 0);

    /** visualization, just a test, should be implemented elsewhere */
    QImage _img;
    assert(_img.load(part_app.m_test_annolist[imgidx].imageName().c_str()));
    QImage img = _img.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&img);
    painter.setRenderHints(QPainter::Antialiasing);

    /** load joints */
    vector<Joint> joints(nJoints);
    loadJoints(part_app, joints, flip);

    /* load classifiers */
    vector<AdaBoostClassifier> v_abc;
    for (int pidx = 0; pidx < nParts; ++pidx) {

      /** here we assume that parts without detector always come in the end ? */

      if (pidx != nParts - 1)
	assert(part_app.m_part_conf.part(pidx).is_detect());

      if (part_app.m_part_conf.part(pidx).is_detect()) {
	AdaBoostClassifier abc;
	part_app.loadClassifier(abc, pidx);
	v_abc.push_back(abc);
      }
    }
    
    for (int ridx = 0; ridx < (int)roi_annolist[imgidx].size(); ++ridx) {

      std::vector<std::vector<PartHyp> > best_part_det;
      std::vector<std::vector<PartHyp> > best_part_hyp;

      findObjectRoiHelper(part_app, roi_annolist[imgidx][ridx], roi_annolist[imgidx].imageName().c_str(), 
			  v_abc, joints, 
			  best_part_det, best_part_hyp);

      assert(best_part_det.size() == nParts);
      assert(best_part_hyp.size() == nParts);

      /** save local maxima of part detections */
      QString qsPartDetOutFilename = qsPartMarginalsDir + "/part_det_imgidx" + padZeros(QString::number(imgidx), 4) + 
	"_roi" + padZeros(QString::number(ridx), 4) + ".mat";

      MATFile *f = matlab_io::mat_open(qsPartDetOutFilename, "wz");

      if (f == 0) {
	cout << "error opening " << qsPartDetOutFilename.toStdString() << endl;
      }
      assert(f != 0);
      
      assert(best_part_det.size() == nParts);
      for (int pidx = 0; pidx < nParts; ++pidx) {
	vector<PartHyp> &part_hyp = best_part_det[pidx];
	FloatGrid2 part_hyp_mat(boost::extents[part_hyp.size()][PartHyp::vectSize()]);

	for (int idx = 0; idx < (int)part_hyp.size(); ++idx) {
	  part_hyp_mat[boost::indices[idx][index_range()]] = part_hyp[idx].toVect();		  
	}
	matlab_io::mat_save_multi_array(f, "part" + QString::number(pidx), part_hyp_mat);
      }
      cout << "saving " << qsPartDetOutFilename.toStdString() << endl;
      matlab_io::mat_close(f);

      /** save local maxima of part posteriors */
      QString qsPartPostOutFilename = qsPartMarginalsDir + "/part_post_imgidx" + padZeros(QString::number(imgidx), 4) + 
	"_roi" + padZeros(QString::number(ridx), 4) + ".mat";

      f = matlab_io::mat_open(qsPartPostOutFilename, "wz");
      assert(f != 0);
      
      assert(best_part_hyp.size() == nParts);
      for (int pidx = 0; pidx < nParts; ++pidx) {
	vector<PartHyp> &part_hyp = best_part_hyp[pidx];
	FloatGrid2 part_hyp_mat(boost::extents[part_hyp.size()][PartHyp::vectSize()]);

	for (int idx = 0; idx < (int)part_hyp.size(); ++idx) {
	  part_hyp_mat[boost::indices[idx][index_range()]] = part_hyp[idx].toVect();		  
	}
	matlab_io::mat_save_multi_array(f, "part" + QString::number(pidx), part_hyp_mat);
      }
      cout << "saving " << qsPartPostOutFilename.toStdString() << endl;
      matlab_io::mat_close(f);

      /** visualize */
      for (int pidx = 0; pidx < nParts; ++pidx) {
	PartBBox bbox;

	/** detections should be sorted, first one is the best one */
	best_part_hyp[pidx][0].getPartBBox(window_param.part(pidx), bbox);
	
	int coloridx = 1;
	int pen_width = 2;
	draw_bbox(painter, bbox, coloridx, pen_width);
      }

      /** 
	  save pose estimation results
      */
      FloatGrid1 v = best_part_hyp[0][0].toVect();

      FloatGrid2 best_conf(boost::extents[nParts][v.shape()[0]]);

      for (int pidx = 0; pidx < nParts; ++pidx) {
	best_conf[boost::indices[pidx][index_range()]] = best_part_hyp[pidx][0].toVect();
      }

      QString qsOutFilename = qsPartMarginalsDir + "/pose_est_imgidx" + padZeros(QString::number(imgidx), 4) + 
	"_roi" + padZeros(QString::number(ridx), 4) + ".mat";

      matlab_io::mat_save_multi_array(qsOutFilename, "best_conf", best_conf);

      PartHyp part_hyp;
      part_hyp.fromVect(best_conf[boost::indices[0][index_range()]]);
    } // regions 

    QString qsVisDir = qsPartMarginalsDir + "/seg_vis_images";

    if (!filesys::check_dir(qsVisDir)) 
      filesys::create_dir(qsVisDir);

    QString qsVisFilename = qsVisDir + "/debug_img_" + padZeros(QString::number(imgidx), 4) + ".png";
    cout << "saving " << qsVisFilename.toStdString() << endl;

    img.save(qsVisFilename);
  }

}
