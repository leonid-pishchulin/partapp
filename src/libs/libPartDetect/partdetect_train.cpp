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

#include <vector>
#include <set>

#include <boost/random/normal_distribution.hpp>

#include <libAdaBoost/AdaBoost.h>
#include <libAdaBoost/BoostingData.h>

#include <libMultiArray/multi_array_def.h>
#include <libMultiArray/multi_array_op.hpp>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <libPartApp/partapp_aux.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libMatlabIO/matlab_io.h>
#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include <libMisc/misc.hpp>

#include <libKMA2/kmaimagecontent.h>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartEval/parteval.h>

#include "partdetect.h"

using boost::multi_array_types::index_range;

using boost_math::double_matrix;
using boost_math::double_vector;

typedef boost::mt19937 RndGen;
typedef boost::normal_distribution<double> NormalDoubleDist;
typedef boost::variate_generator<RndGen, NormalDoubleDist> NormalDoubleGen;

using namespace std;

namespace part_detect { 

  /**
     get part bounding box (this is essentially the same as "bbox_from_pos" used for 
     part evaluation, merge them?) 

     MA: what is "part_center"? Is this the same as part position? If yes then this function does not work for parts that have part position not equal to center of bounding box
  */
  void make_bbox(const PartWindowParam &window_params, int pidx, 
		 double part_center_x, double part_center_y, double scale, double rot, 
		 PartBBox &bbox)
  {
    assert(window_params.part_size() > pidx);

    bbox.part_pos(0) = part_center_x;
    bbox.part_pos(1) = part_center_y;

    bbox.part_x_axis(0) = cos(rot);
    bbox.part_x_axis(1) = sin(rot);

    bbox.part_y_axis(0) = -sin(rot);
    bbox.part_y_axis(1) = cos(rot);

    bbox.min_proj_x = -scale*window_params.part(pidx).pos_offset_x();
    bbox.min_proj_y = -scale*window_params.part(pidx).pos_offset_y();

    bbox.max_proj_x = bbox.min_proj_x + scale*window_params.part(pidx).window_size_x();
    bbox.max_proj_y = bbox.min_proj_y + scale*window_params.part(pidx).window_size_y();
  }

  bool is_point_in_rect(PartBBox bbox, double point_x, double point_y)
  {
    double_vector point_pos(2);
    point_pos(0) = point_x;
    point_pos(1) = point_y;
  
    point_pos -= bbox.part_pos;
    assert(abs(ublas::norm_2(bbox.part_x_axis) - 1.0) < 1e-6);
    assert(abs(ublas::norm_2(bbox.part_y_axis) - 1.0) < 1e-6);

    double proj_x = ublas::inner_prod(point_pos, bbox.part_x_axis);
    double proj_y = ublas::inner_prod(point_pos, bbox.part_y_axis);
    
    if (proj_x > bbox.min_proj_x && proj_x < bbox.max_proj_x &&
        proj_y > bbox.min_proj_y && proj_y < bbox.max_proj_y) 
      return true;

    return false;
  }

  bool is_occluded(const AnnoRect &annorect, const PartDef &partdef){
    
    for (int i = 0; i < partdef.part_pos_size(); ++i) {
      uint id = partdef.part_pos(i);
      const AnnoPoint *p = annorect.get_annopoint_by_id(id);
      assert(p != NULL);
      
      /* make sure we get the right annopoint */
      assert(p->id == (int)id);
      
      if (!(p->is_visible)){
	return true;
      }
      
    }
    return false;
  }
  
  /**

     load detection scores from disc and extract the required number of false positives

   */
  void bootstrap_get_rects(const PartApp &part_app, int imgidx, int pidx, 
                           int num_rects, double min_score, 
                           vector<PartBBox> &rects, vector<double> &rects_scale,
                           bool bIgnorePartRects, bool bDrawRects, int pidx_window_param)
  {
    cout << "bootstrap_get_rects" << endl;

    double scale_tolerance_threshold = 0.21;
    double rot_tolerance_threshold = 31;
    
    float overlap_threshold = 0.3;
    if (part_app.m_exp_param.has_bootstrap_threshold())
      overlap_threshold = part_app.m_exp_param.bootstrap_threshold();
    
    rects.clear();
    rects_scale.clear();

    QString qsBootstrapDir = (part_app.m_exp_param.log_dir() + "/" + 
                              part_app.m_exp_param.log_subdir() + "/bootstrap_scoregrid").c_str();

    assert(filesys::check_dir(qsBootstrapDir));

    /* load exp_param, used while bootstrapping */
    QString qsExpParamBootstrap = qsBootstrapDir + "/exp_param_bootstrap.txt";
    ExpParam exp_param_bootstrap;

    assert(filesys::check_file(qsExpParamBootstrap));
    parse_message_from_text_file(qsExpParamBootstrap, exp_param_bootstrap);

    vector<PartBBox> all_rects;
    vector<double> all_rects_scale;

    QString qsFilename = qsBootstrapDir + "/bootstrap_annolist.al";
    assert(filesys::check_file(qsFilename));

    QString qsDebugDir = (part_app.m_exp_param.log_dir() + "/" + 
                          part_app.m_exp_param.log_subdir() + "/debug").c_str();

    if (bDrawRects) {
      if (!filesys::check_dir(qsDebugDir))
        filesys::create_dir(qsDebugDir);
    }

    AnnotationList bootstrap_annolist;
    bootstrap_annolist.load(qsFilename.toStdString());
    assert(bootstrap_annolist.size() > 0);

    assert(imgidx >= 0 && imgidx < (int)bootstrap_annolist.size());

    bool flip = false;
    
    qsFilename = qsBootstrapDir + "/imgidx" + QString::number(imgidx) + 
      "-pidx" + QString::number(pidx_window_param) + "-o" + QString::number((int)flip) + "-scoregrid.mat";

    vector<vector<FloatGrid2> > score_grid;

    /* load scoremaps */

    cout << "loading " << qsFilename.toStdString() << endl;

    MATFile *f = matlab_io::mat_open(qsFilename, "r");
    assert(f != 0);
    matlab_io::mat_load_multi_array_vec2(f, "cell_scoregrid", score_grid);
    matlab_io::mat_close(f);

    f = matlab_io::mat_open(qsFilename, "r");
    FloatGrid4 transform_Ti2 = matlab_io::mat_load_multi_array<FloatGrid4>(f, QString("transform_Ti2"));
    matlab_io::mat_close(f);

    f = matlab_io::mat_open(qsFilename, "r");
    FloatGrid4 transform_T2g = matlab_io::mat_load_multi_array<FloatGrid4>(f, QString("transform_T2g"));
    matlab_io::mat_close(f);

    /* initilize grid dimensions */
    assert(score_grid.size() == exp_param_bootstrap.num_scale_steps());
    assert(score_grid[0].size() == exp_param_bootstrap.num_rotation_steps());
    int nScales = score_grid.size();
    int nRotations = score_grid[0].size();

    cout << "nScales: " << nScales << endl;
    cout << "nRotations: " << nRotations << endl;

    /* initialize images for visualization of false positives at different scales */
    QImage _img;
    assert(_img.load(bootstrap_annolist[imgidx].imageName().c_str()));

    QImage img = _img.convertToFormat(QImage::Format_RGB32);
    //assert(img.save("./debug/original.png"));

    vector<QImage> img_scale;
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
      img_scale.push_back(img);

    /** 
        add all parts from the same appearance group as the current part to the list 
        of parts which should not be treated as false positives
    */
    vector<double> gt_scale;
    vector<double> gt_rot;
    vector<PartBBox> gt_bbox;
    double grow_factor = 1.1;

    vector<int> gt_pidx;
    gt_pidx.push_back(pidx);

    int agidx = get_app_group_idx(part_app.m_part_conf, part_app.m_part_conf.part(pidx).part_id());
    if (agidx != -1) {
      cout << "appearance group: " << agidx << endl;
      assert(agidx >= 0 && agidx < part_app.m_part_conf.app_group_size());

      for (int idx = 0; idx < part_app.m_part_conf.app_group(agidx).part_id_size(); ++idx) {
        int pid = part_app.m_part_conf.app_group(agidx).part_id(idx);
        int _pidx = pidx_from_pid(part_app.m_part_conf, pid);
        assert(_pidx >= 0 && _pidx < part_app.m_part_conf.part_size());

        if (pidx != _pidx) {
          gt_pidx.push_back(_pidx);
          cout << "\tadding part " << _pidx << " to the ground truth " << endl;
        }
      }
    }

    for (int rectidx = 0; rectidx < (int)bootstrap_annolist[imgidx].size(); ++rectidx) {

      int rect_height = bootstrap_annolist[imgidx][rectidx].bottom() - bootstrap_annolist[imgidx][rectidx].top();

      /* compute relative scale */
      double scale = rect_height / (double)part_app.m_window_param.train_object_height();

      cout << "rectidx: " << rectidx << ", scale: " << scale << endl;

      for (int pidxidx = 0; pidxidx < (int)gt_pidx.size(); ++pidxidx) {
          PartBBox bbox;
          get_part_bbox(bootstrap_annolist[imgidx][rectidx], part_app.m_part_conf.part(gt_pidx[pidxidx]), bbox);
      
          bbox.min_proj_x *= grow_factor;
          bbox.max_proj_x *= grow_factor;
          bbox.min_proj_y *= grow_factor;
          bbox.max_proj_y *= grow_factor;
      
          double rotation = atan2(bbox.part_x_axis(1), bbox.part_x_axis(0));
          rotation = rotation/M_PI*180.0;

          gt_scale.push_back(scale);
          gt_rot.push_back(rotation);
          gt_bbox.push_back(bbox);     
      }
    }
  
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx)
      for (int rotidx = 0; rotidx < nRotations; ++rotidx) {


        //double scale = scale_from_index(part_app.m_exp_param, scaleidx);
        //double rotation = rot_from_index(part_app.m_exp_param, rotidx);

        /* scale is wrt discretization used while bootstrapping, not the one used in experiment */
        double scale = scale_from_index(exp_param_bootstrap, scaleidx);
        double rotation = rot_from_index(exp_param_bootstrap, rotidx);


        int grid_width = score_grid[scaleidx][rotidx].shape()[1];
        int grid_height = score_grid[scaleidx][rotidx].shape()[0];

        double_matrix T2g;
        multi_array_op::array_to_matrix(transform_T2g[boost::indices[scaleidx][rotidx][index_range()][index_range()]], T2g);

        double_matrix Ti2;
        multi_array_op::array_to_matrix(transform_Ti2[boost::indices[scaleidx][rotidx][index_range()][index_range()]], Ti2);
      
        double_matrix Tig(3, 3);
        Tig = prod(Ti2, T2g);

        for (int gx = 0; gx < grid_width; ++gx)
          for (int gy = 0; gy < grid_height; ++gy) {
            if (score_grid[scaleidx][rotidx][gy][gx] == NO_CLASS_VALUE)
              continue;

            double_vector vg = hc::get_point(gx, gy);
            double_vector vi = prod(Tig, vg);
          
            //const ExpParam &exp_param = part_app.m_exp_param;

            if (score_grid[scaleidx][rotidx][gy][gx] > min_score) {
              bool rectok = true;

	      if (bIgnorePartRects && not getNumPartTypes(part_app.m_window_param, pidx) ==1) {
                  /* test that rect in not on the groundtruth */
                  for (int gtidx = 0; gtidx < (int)gt_scale.size(); ++gtidx) {
		    if (is_point_in_rect(gt_bbox[gtidx], vi(0), vi(1)) && 
                        abs(gt_scale[gtidx] - scale) <= scale_tolerance_threshold && 
                        abs(gt_rot[gtidx] - rotation) <= rot_tolerance_threshold) {
                      rectok = false;
		      break;
                    }
                  }
	      }
	      else if (bIgnorePartRects && getNumPartTypes(part_app.m_window_param, pidx)>1) {
                  /* test that rect in not on the groundtruth */
                  for (int gtidx = 0; gtidx < (int)gt_scale.size(); ++gtidx) {
		    
		    PartBBox bbox;
		    make_bbox(part_app.m_window_param, pidx_window_param, 
			      vi(0), vi(1), 
			      scale_from_index(exp_param_bootstrap, scaleidx), 
			      rot_from_index(exp_param_bootstrap, rotidx)*M_PI/180.0,
			      bbox);
		    
		    PartBBox gt = gt_bbox[gtidx];
		    gt.min_proj_y += part_app.m_part_conf.part(pidx).ext_y_neg();
		    gt.max_proj_y -= part_app.m_part_conf.part(pidx).ext_y_pos();
		    gt.min_proj_x += part_app.m_part_conf.part(pidx).ext_x_neg();
		    gt.max_proj_x -= part_app.m_part_conf.part(pidx).ext_x_pos();
		    
		    bbox.min_proj_y += part_app.m_part_conf.part(pidx).ext_y_neg();
		    bbox.max_proj_y -= part_app.m_part_conf.part(pidx).ext_y_pos();
		    bbox.min_proj_x += part_app.m_part_conf.part(pidx).ext_x_neg();
		    bbox.max_proj_x -= part_app.m_part_conf.part(pidx).ext_x_pos();

		    bool bIsGTmatch = is_gt_bbox_match(gt, bbox, overlap_threshold);
		    //cout << "is_gt_bbox_match: " << bIsGTmatch << endl;
		    
                    if (bIsGTmatch) {
                      rectok = false;
		      break;
                    }
                  }
	      }
	      
              PartBBox bbox;
	      
	      make_bbox(part_app.m_window_param, pidx_window_param, 
                        vi(0), vi(1), 
                        scale_from_index(exp_param_bootstrap, scaleidx), 
                        rot_from_index(exp_param_bootstrap, rotidx)*M_PI/180.0,
                        bbox);
	      
              if (bDrawRects) {
                QPainter painter(&img_scale[scaleidx]);
                painter.setPen(Qt::yellow);
		draw_bbox(painter, bbox, !rectok);
		draw_bbox(painter, gt_bbox[0], 2);

              }
	      
              if (rectok) {
                all_rects.push_back(bbox);
                all_rects_scale.push_back(scale);
              }
            }

          }// grid points
      }// scales&rotations
     
    cout << "found " << all_rects.size() << " rectangles with score > " << min_score << endl;

    /* choose random subset of found rectangles */
    if ((int)all_rects.size() > num_rects) {
      vector<int> randperm;
      for (int idx = 0; idx < (int)all_rects.size(); ++idx)
        randperm.push_back(idx);

      std::random_shuffle(randperm.begin(), randperm.end());

      for (int idx = 0; idx < num_rects; ++idx) {
        rects.push_back(all_rects[randperm[idx]]);
        rects_scale.push_back(all_rects_scale[randperm[idx]]);
      }
    }

    /* save visualization*/

    if (bDrawRects) {
      for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {
//         QString qsScale = QString::number(scaleidx);
//         if (qsScale.length() == 1)
//           qsScale = "0" + qsScale;

        //QString qsFilename = "./debug/get_rects_test_" + QString::number(imgidx) + "_scale" + qsScale + ".png";

        QString qsFilename = qsDebugDir + "/bootstrap_fp_imgidx_" + padZeros(QString::number(imgidx), 4) + 
          "_pidx" + padZeros(QString::number(pidx_window_param), 2) + 
          "_scale" + padZeros(QString::number(scaleidx), 2) + ".png";

        cout << "saving " << qsFilename.toStdString() << endl;
        assert(img_scale[scaleidx].save(qsFilename));
      }
    }
    cout << "done" << endl;
  }
  

  /**

  here we extract regions of training images which contain the object (no rescaling, only crop)
  and save them in separate directory, corresponding annolist is also created

  */
  void prepare_bootstrap_dataset(const PartApp &part_app, const AnnotationList &annolist, 
                                 int firstidx, int lastidx)
  {
    AnnotationList new_annolist;

    QString qsBootstrapDir = (part_app.m_exp_param.log_dir() + "/" + 
                              part_app.m_exp_param.log_subdir() + "/bootstrap_scoregrid").c_str();

    cout << "saving scores to " << qsBootstrapDir.toStdString() << endl;

    if (!filesys::check_dir(qsBootstrapDir))
      assert(filesys::create_dir(qsBootstrapDir));

    int crop_offset_x = 150;
    int crop_offset_y = 150;
    
    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      kma::ImageContent *kmaimg = kma::load_convert_gray_image(annolist[imgidx].imageName().c_str());

      int img_width = kmaimg->x();
      int img_height = kmaimg->y();

      for (int ridx = 0; ridx < (int)annolist[imgidx].size(); ++ridx) {
        int obj_x = (int)(0.5*(annolist[imgidx][ridx].left() + annolist[imgidx][ridx].right()));
        int obj_y = (int)(0.5*(annolist[imgidx][ridx].top() + annolist[imgidx][ridx].bottom()));

        int minx = max(obj_x - crop_offset_x, 0);
        int maxx = min(obj_x + crop_offset_x, img_width-1);

        int miny = max(obj_y - crop_offset_y, 0);
        int maxy = min(obj_y + crop_offset_y, img_height-1);

        int new_img_width = maxx - minx + 1;
        int new_img_height = maxy - miny + 1;

        kma::ImageContent *newimg = new kma::ImageContent(new_img_height, new_img_width);

        /* 
           this initialization is necessary, for some reason not all positions are filled during interpolation 
           i suspect this has something to do with even/odd number of pixels in the target image 

           MA: this should be fixed by now in my libKMA
        */
        for (int ix = 0; ix < (int)newimg->x(); ++ix) 
          for (int iy = 0; iy < (int)newimg->y(); ++iy)
            newimg->fel[iy][ix] = 0;

        newimg->crop(kmaimg, (int)(0.5*(minx + maxx)), (int)(0.5*(miny + maxy)));
        QString qsFilename2 = qsBootstrapDir + "/img" + padZeros(QString::number(imgidx), 4) + ".png";

        cout << "saving " << qsFilename2.toStdString() << endl;
        newimg->writePNG(qsFilename2.toStdString().c_str());

        AnnoRect r = annolist[imgidx][ridx];
          r.setCoords(r.left() - minx, r.top() - miny, r.right() - minx, r.bottom() - miny);
          r.setObjPos((int)(0.5*(r.left() + r.right())), (int)(0.5*(r.top() + r.bottom())));

          for (int idx = 0; idx < (int)r.m_vAnnoPoints.size(); ++idx) {
            r.m_vAnnoPoints[idx].x -= minx;
            r.m_vAnnoPoints[idx].y -= miny;
          }

        Annotation a(qsFilename2.toStdString());
            a.addAnnoRect(r);
        new_annolist.addAnnotation(a);      

        delete newimg;
      }// rectangles

      delete kmaimg;    
    }// images


    QString qsFilename = qsBootstrapDir + "/bootstrap_annolist.al";
    new_annolist.save(qsFilename.toStdString());
  }

  void bootstrap_partdetect(PartApp &part_app, int firstidx, int lastidx)
  {
    QString qsBootstrapDir = (part_app.m_exp_param.log_dir() + "/" + 
                              part_app.m_exp_param.log_subdir() + "/bootstrap_scoregrid").c_str();

    //assert(filesys::check_dir(qsBootstrapDir));
    filesys::check_dir(qsBootstrapDir);

    cout << "loading scores from " << qsBootstrapDir.toStdString() << endl;

    QString qsFilename = qsBootstrapDir + "/bootstrap_annolist.al";
    AnnotationList bootstrap_annolist;
    bootstrap_annolist.load(qsFilename.toStdString());

    AbcDetectorParam abc_param = part_app.m_abc_param;

    /* load classifiers */
    vector<AdaBoostClassifier> v_abc;
    int nParts = part_app.m_part_conf.part_size();

    for (int pidx = 0; pidx < nParts; ++pidx) {
      if (pidx != nParts - 1)
        assert(part_app.m_part_conf.part(pidx).is_detect());

      if (part_app.m_part_conf.part(pidx).is_detect()) {
	if (!part_app.m_part_conf.part(pidx).has_mult_types()) {
	  AdaBoostClassifier abc;

	  /* here we should always load non-bootstrapped classifier: part_app.loadClassifier(abc, pidx, 0); */
	  part_app.loadClassifier(abc, pidx, 0);
	  //part_app.loadClassifier(abc, pidx, 1);
	  
	  //part_app.loadClassifier(abc, pidx);
	  v_abc.push_back(abc);
	}
	else{
	  int nTypes = getNumPartTypes(part_app.m_window_param, pidx);
	  //cout << "pidx = " << pidx << ", ntypes: " << nTypes << endl;
	  for(uint tidx = 0; tidx < nTypes; ++tidx){
	    AdaBoostClassifier abc;
	    int boostrap_type = 0;
	    part_app.loadClassifier(abc, pidx, boostrap_type, tidx);
	    v_abc.push_back(abc);
	  }
	}
      }
    }

    /* compute windows as sparsely as possible */
    abc_param.set_window_desc_step_ratio(1.0); 

    bool bSaveImageScoreGrid = false;
    bool bAddImageBorder = true;
    bool flip = false;

    /*
       enforce scale range to collect false positives at
       scales smaller and larger then the training scale 
    */

    ExpParam exp_param_bootstrap = part_app.m_exp_param;
    exp_param_bootstrap.set_min_object_scale(0.45);
    exp_param_bootstrap.set_max_object_scale(1.55);
    exp_param_bootstrap.set_num_scale_steps(11);
    
    // out of time and memory reasons:
    // if having multiple part types never search over multiple scales
    
    for (int pidx = 0; pidx < nParts; pidx++){
      int nTypes = getNumPartTypes(part_app.m_window_param, pidx);
      if (nTypes > 1){
	exp_param_bootstrap.set_min_object_scale(1.0);
	exp_param_bootstrap.set_max_object_scale(1.0);
	exp_param_bootstrap.set_num_scale_steps(1);
	break;
      }
    }
    /* store exp_param, used while bootstrapping */
    QString qsExpParamBootstrap = qsBootstrapDir + "/exp_param_bootstrap.txt";
    print_message_to_text_file(qsExpParamBootstrap, exp_param_bootstrap);


    /* same function as used for part detection at the test stage */  
    //    partdetect_dense(part_app.m_exp_param, abc_param, part_app.m_window_param, 
    vector<vector<FloatGrid3> > part_detections;
    partdetect_dense(exp_param_bootstrap, abc_param, part_app.m_window_param, 
                     bootstrap_annolist, v_abc, 
                     qsBootstrapDir, 
                     firstidx, lastidx, flip, 
                     bSaveImageScoreGrid, bAddImageBorder,
		     part_detections);


  }


  kma::ImageContent* crop_part_region(kma::ImageContent *image, const PartBBox &bbox, int desc_size) {

    /** MA: Should not we add 2*desc_size here? Also, what happens if part center != center of part bounding box ? (case X) */
    /** MA: All of this is not critical, this function is used only for visualization, right?                      */

    /** MA: try to handle the (case X) properly */
    double box_cx = 0.5*(bbox.max_proj_x + bbox.min_proj_x);
    double box_cy = 0.5*(bbox.max_proj_y + bbox.min_proj_y);
    boost_math::double_vector world_c = bbox.part_pos + box_cx*bbox.part_x_axis + box_cy*bbox.part_y_axis;
    
    int region_width = (int)(bbox.max_proj_x - bbox.min_proj_x + 1) + desc_size;
    int region_height = (int)(bbox.max_proj_y - bbox.min_proj_y + 1) + desc_size;

    kma::ImageContent *region = new kma::ImageContent(region_height, region_width);

    region->interpolate(image, world_c(0), world_c(1), bbox.part_x_axis(0), bbox.part_y_axis(0), 
                        bbox.part_x_axis(1), bbox.part_y_axis(1));

//     region->interpolate(image, bbox.part_pos(0), bbox.part_pos(1), bbox.part_x_axis(0), bbox.part_y_axis(0), 
//                         bbox.part_x_axis(1), bbox.part_y_axis(1));

    /** clip to [0, 255] since "interpolate" forgets to fill last row/column if image size is odd 
        
        MA: this should be fixed by now
     */
    for (int ix = 0; ix < region_width; ++ix) 
      for (int iy = 0; iy < region_height; ++iy){
        if (region->fel[iy][ix] < 0)
          region->fel[iy][ix] = 0;
        else if (region->fel[iy][ix] > 255)
          region->fel[iy][ix] = 255;
      }

    return region;              
  }


  namespace ExamplesCounting {

    /* number of positive/negative examples that should be available for training */
    int num_train_pos;
    int num_train_neg;

    /* number of positive examples in the training data,
       this number may vary for different parts (i.e. if not every image has all parts annotated)
    */
    int num_avail_pos;

    /* number of jittered positive examples */
    int num_jitter_pos;

    /* number of jittered examples per positive example */
    int num_jitter_per_pos;

    /* number of negative images */
    int num_avail_negative_images;

    /* number of bootstrapping examples required */
    int num_bootstrap_req;

    /* number of images suitable for bootstrapping 
       (current part must be annotated in all rectangles of the image) */
    int num_avail_bootstrap_images;
    int num_bootstrap_per_image;

    /* number of negative rectangles that should be randomly sampled in background 
       (this is known after collection of bootstrapping examples) */
    int num_neg_random;
    int num_neg_random_per_image;

    /* counting variables for number of added examples */
    int added_bootstrap;
    int added_bootstrap_curimg;

    int added_pos;
    int added_pos_jitter;
    
    int added_neg;
    int added_neg_curimg;

    vector<FeatureVector> all_jitter;
    vector<QString> all_jitter_image_names;
    vector<kma::ImageContent *> all_jitter_images;

    vector<FeatureVector> all_negatives;
    vector<QString> all_negatives_image_names;
    vector<kma::ImageContent *> all_negatives_images;

    vector<FeatureVector> all_bootstrap;
    vector<QString> all_bootstrap_image_names;
    vector<kma::ImageContent *> all_bootstrap_images;
    
    void init(const AbcDetectorParam &abc_param, 
	      const PartDef &part_def, 
	      const AnnotationList &reallist,
	      const AnnotationList &reshapedlist,
	      const AnnotationList &neglist,
	      const AnnotationList &bootstrap_annolist,
	      const int part_image_offset,
	      const bool bAddImageBorder) {

      /** clean up from the previous traing run */
      all_jitter.clear();
      all_jitter_image_names.clear();
      all_jitter_images.clear();

      all_negatives.clear();
      all_negatives_image_names.clear();
      all_negatives_images.clear();

      all_bootstrap.clear();
      all_bootstrap_image_names.clear();
      all_bootstrap_images.clear();

      /** init */

      num_train_pos = abc_param.num_train_pos() + abc_param.num_train_reshaped(); 

      if (abc_param.has_num_train_neg())
        num_train_neg = abc_param.num_train_neg();
      else 
        num_train_neg = num_train_pos + abc_param.num_train_reshaped();

      /** 
          find out how many positive rectangles have annotation for current part
      */
      num_avail_pos = 0; 
      for (uint imgidx = 0; imgidx < reallist.size(); ++imgidx) 
        for (uint ridx = 0; ridx < reallist[imgidx].size(); ++ridx)
          if (annorect_has_part(reallist[imgidx][ridx], part_def)) 
	    //if (annorect_has_part(reallist[imgidx][ridx], part_def)&& !is_occluded(reallist[imgidx][ridx], part_def))
	      ++num_avail_pos;
      
      assert(num_avail_pos > 0);

      num_jitter_pos = 0;
      num_jitter_per_pos = 0;

      if (abc_param.do_jitter()) {
        num_jitter_pos = num_train_pos - num_avail_pos - abc_param.num_train_reshaped();
        num_jitter_per_pos = (int)ceil(num_jitter_pos / (double)num_avail_pos);
      }
      
      /** 
          find out how many negative images have annotation for current part
          discard images wich have annotated rectangles and current part is not annotated
      */
      num_avail_negative_images = neglist.size();
      for (uint imgidx = 0; imgidx < neglist.size(); ++imgidx){ 
        for (uint ridx = 0; ridx < neglist[imgidx].size(); ++ridx) 
          if (!annorect_has_part(neglist[imgidx][ridx], part_def)){
            --ExamplesCounting::num_avail_negative_images;
            break;
          }
      }
	
      /** 
        find out how many images can be used for bootstrapping
      */
      num_bootstrap_req = 0;
      num_avail_bootstrap_images = 0;
      num_bootstrap_per_image = 0;

      if (bootstrap_annolist.size() > 0) {
        num_bootstrap_req = (int)(abc_param.bootstrap_fraction() * num_train_neg);

        num_avail_bootstrap_images = bootstrap_annolist.size();

        for (int imgidx = 0; imgidx < (int)bootstrap_annolist.size(); ++imgidx) {
          assert(bootstrap_annolist[imgidx].size() < 2);

          if (bootstrap_annolist[imgidx].size() == 1) 
            if (!annorect_has_part(bootstrap_annolist[imgidx][0], part_def))
              --num_avail_bootstrap_images;
        }

        assert(num_avail_bootstrap_images > 0);
        num_bootstrap_per_image =  (int)ceil(num_bootstrap_req / (double)num_avail_bootstrap_images);
      }

      num_neg_random = 0; // depends on added_bootstrap
      num_neg_random_per_image = 0;

      added_bootstrap = 0;
      added_bootstrap_curimg = 0;

      added_pos = 0;
      added_pos_jitter = 0;

      added_neg = 0;
      added_neg_curimg = 0;
    }

    void done_collect_bootstrap() {
      num_neg_random = num_train_neg - added_bootstrap;
      assert(num_neg_random >= 0);

      assert(num_avail_negative_images > 0);
      num_neg_random_per_image = (int)ceil(num_neg_random / (double)num_avail_negative_images);
    }

    void print(ostream &out) {
      out << "ExamplesCounting::print()" 
           << "\n\tnum_train_pos: " << num_train_pos 
           << "\n\tnum_train_neg: " << num_train_neg 
           << "\n\tnum_avail_pos: "<< num_avail_pos 
           << "\n\tnum_jitter_pos: " << num_jitter_pos 
           << "\n\tnum_jitter_per_pos: " << num_jitter_per_pos
           << "\n\tnum_avail_negative_images: " << num_avail_negative_images
           << "\n\tnum_bootstrap_req: " << num_bootstrap_req
           << "\n\tnum_avail_bootstrap_images: " << num_avail_bootstrap_images
           << "\n\tnum_bootstrap_per_image: " << num_bootstrap_per_image
           << "\n\tnum_neg_random: " << num_neg_random
           << "\n\tnum_neg_random_per_image: " << num_neg_random_per_image
           << "\n\tadded_bootstrap: " << added_bootstrap
           << "\n\tadded_pos: " << added_pos
           << "\n\tadded_pos_jitter: " << added_pos_jitter
           << "\n\tadded_neg: " << added_neg << endl;
    }

  }

  void sample_reshaped_points(const AnnotationList &reshapedlist,
			      AnnotationList &sampled_reshapedlist, 
			      const PartConfig &part_conf, 
			      const PartWindowParam window_param,
			      const AbcDetectorParam &abc_param, int pidx, int pidx_window_param){
    
    vector<uint> validSamples;

    for (uint imgidx = 0; imgidx < reshapedlist.size(); ++imgidx) {
      for (uint ridx = 0; ridx < reshapedlist[imgidx].size(); ++ridx) {
	
	if (!annorect_has_part(reshapedlist[imgidx][ridx], part_conf.part(pidx)))
	  continue;
	
	kma::ImageContent *input_image = kma::load_convert_gray_image(reshapedlist[imgidx].imageName().c_str());
	
	PartBBox bbox;
	get_part_bbox(reshapedlist[imgidx][ridx], part_conf.part(pidx), bbox);
	
	double_vector endpoint_top;
	double_vector endpoint_bottom;
	double seg_len;
       	get_bbox_endpoints(bbox, endpoint_top, endpoint_bottom, seg_len);
	
	double factor = 0.3;
	/*
	if (seg_len < factor * window_param.part(pidx).window_size_y())
	  continue;
	*/
	if (seg_len < factor * window_param.part(pidx_window_param).window_size_y())
	  continue;
	
	float offset = 20;
	
	// exclude all parts close to the border
	if (not((bbox.part_pos(0) + bbox.min_proj_x < offset) || 
		(bbox.part_pos(0) + bbox.max_proj_x > input_image->x() - offset) ||
		(bbox.part_pos(1) + bbox.min_proj_y < offset) || 
		(bbox.part_pos(1) + bbox.max_proj_y > input_image->y() - offset))){
	  validSamples.push_back(imgidx);
	}
      }
    }
    
    // sample from the list of reshaped images
    std::random_shuffle(validSamples.begin(), validSamples.end());
    
    uint nSamples = (abc_param.num_train_reshaped() < validSamples.size() ? abc_param.num_train_reshaped() : validSamples.size());

    for (uint aidx = 0; aidx < nSamples; ++aidx) {
      sampled_reshapedlist.addAnnotation(reshapedlist[validSamples[aidx]]);
    }
  }
  
  /**
     train AdaBoost classifer for specific part optionally including jittered positive samples 
     and hard negative samples (bootstrapping). 

     Bootstrapping samples are obtained from a set of detections produced by initial classifer
     trained without adding hard examples.
  */
  void abc_train_class(PartApp &part_app, int pidx, bool bBootstrap, int tidx)
  {
    cout << "abc_train_class" << endl;
    cout << "feature type: " << part_app.m_abc_param.feature_type() << endl;
    cout << "bBootstrap: " << bBootstrap << endl;

    const AbcDetectorParam &abc_param = part_app.m_abc_param;
    const PartConfig &part_conf = part_app.m_part_conf;
    const ExpParam &exp_param = part_app.m_exp_param;
    const PartWindowParam &window_param = part_app.m_window_param;
    AnnotationList poslist;
    const AnnotationList &reallist = part_app.m_train_annolist;
    const AnnotationList &reshapedlist = part_app.m_train_reshaped_annolist;
    //const AnnotationList &neglist = part_app.m_train_annolist;
    const AnnotationList &neglist = (part_app.m_neg_annolist.size() > 0 ? part_app.m_neg_annolist : part_app.m_train_annolist);

    AnnotationList bootstrap_annolist;
    if (bBootstrap) {
      QString qsBootstrapDir = (part_app.m_exp_param.log_dir() + "/" + 
                                part_app.m_exp_param.log_subdir() + "/bootstrap_scoregrid").c_str();

      assert(filesys::check_dir(qsBootstrapDir));
      cout << "using boostraping data from " << qsBootstrapDir.toStdString() << endl;

      QString qsFilename = qsBootstrapDir + "/bootstrap_annolist.al";
      assert(filesys::check_file(qsFilename));

      bootstrap_annolist.load(qsFilename.toStdString());
      assert(bootstrap_annolist.size() > 0);
    }
    
    // init random number generator
    srand(42u);    
    
    // when using reshaped images
    assert(abc_param.num_train_reshaped() <= reshapedlist.size());
		       
    bool bSavePartRegions = true;

    /** add image border so that features of the part rectangle could be computed for any position (only affects negatives for now) */
    bool bAddImageBorder = true;  
    
    int pidx_window_param = getPartById(part_app.m_window_param, pidx, tidx);
    // assume parts without types come as first
    if (!part_app.m_part_conf.part(pidx).has_mult_types())
      assert(pidx_window_param == pidx);
        
    int part_image_offset = (int)ceil(sqrt(square(window_param.part(pidx_window_param).window_size_x()) + 
                                           square(window_param.part(pidx_window_param).window_size_y()))/2) + (int)(1.1*abc_param.desc_size()) + 1;
    /**
       init different counters for numbers of training/test examples
     */
    ExamplesCounting::init(part_app.m_abc_param, part_app.m_part_conf.part(pidx),
                           reallist, reshapedlist, neglist, bootstrap_annolist, 
			   part_image_offset, bAddImageBorder);

    /** create directories to store training images (only for inspection, features are computed on the fly) */

    QString qsPosTrainDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
                             "/pos_train").c_str();

    qsPosTrainDir += "/pidx" + QString::number(pidx);

    QString qsNegTrainDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
                             "/neg_train").c_str();

    qsNegTrainDir += "/pidx" + QString::number(pidx);

    QString qsJitterTrainDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
                                "/jitter_train").c_str();

    qsJitterTrainDir += "/pidx" + QString::number(pidx);

    QString qsBootstrapTrainDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
                                   "/bootstrap_train").c_str();
    qsBootstrapTrainDir += "/pidx" + QString::number(pidx);

    if (part_app.m_part_conf.part(pidx).has_mult_types()){
      qsPosTrainDir += "_type" + QString::number(tidx);
      qsNegTrainDir += "_type" + QString::number(tidx);
      qsJitterTrainDir += "_type" + QString::number(tidx);
      qsBootstrapTrainDir += "_type" + QString::number(tidx);
    }

    if (bSavePartRegions) {
      filesys::create_dir(qsPosTrainDir);
      filesys::create_dir(qsNegTrainDir);

      if (abc_param.do_jitter())
        filesys::create_dir(qsJitterTrainDir);

      if (bBootstrap)
        filesys::create_dir(qsBootstrapTrainDir);
    }
  
    /** train classifier - begin */
    AdaBoostClassifier abc;
    vector<FeatureVector> trainingdata;

    /** min score for false positives used as bootstrapping examples */
    double min_boostrap_score = 0.1;

    /** define now much jitter to add to the data */
    double jitter_offsetx_sigma = 2;
    double jitter_offsety_sigma = 2;
    double jitter_scale_sigma = 0.05;
    double jitter_rot_sigma = 2*M_PI/360.0;

    /** define random number generators for jittering data */
    uint jitter_rnd_seed = 42u;
    NormalDoubleGen jitter_rot_gen(RndGen(jitter_rnd_seed++), NormalDoubleDist(0, jitter_rot_sigma));
    NormalDoubleGen jitter_scale_gen(RndGen(jitter_rnd_seed++), NormalDoubleDist(1, jitter_scale_sigma));
    NormalDoubleGen jitter_offsetx_gen(RndGen(jitter_rnd_seed++), NormalDoubleDist(0, jitter_offsetx_sigma));
    NormalDoubleGen jitter_offsety_gen(RndGen(jitter_rnd_seed++), NormalDoubleDist(0, jitter_offsety_sigma));

    ExamplesCounting::print(cout);

    /**                                             */
    /** collect negative points after bootstrapping */
    /**                                             */
    if (bBootstrap) {

      for (int imgidx = 0; imgidx < (int)bootstrap_annolist.size(); ++imgidx) {

        /* skip image if the part is not annotated */
        if (bootstrap_annolist[imgidx].size() == 1) 
          if (!annorect_has_part(bootstrap_annolist[imgidx][0], part_conf.part(pidx)))
            continue;
            
        ExamplesCounting::added_bootstrap_curimg = 0;

        kma::ImageContent *kmaimg = kma::load_convert_gray_image(bootstrap_annolist[imgidx].imageName().c_str());
    
        vector<PartBBox> rects;
        vector<double> rects_scale;

        bool bIgnorePartRects = true;
        bool bDrawRects = false;

        /* try to get a little more images then needed (in case if some rectangles will be too close to the border) */

        bootstrap_get_rects(part_app, imgidx, pidx, 
                            3*ExamplesCounting::num_bootstrap_per_image, min_boostrap_score,
                            rects, rects_scale,               
                            bIgnorePartRects, bDrawRects, pidx_window_param);
	//getchar();
        for (int rectidx = 0; rectidx < (int)rects.size(); ++rectidx) {
          PartBBox adjusted_rect;
          vector<float> all_features;
	  
	  bool bres = compute_part_bbox_features_scale(abc_param, window_param, 
                                                       rects[rectidx], kmaimg, pidx_window_param, rects_scale[rectidx], 
                                                       all_features, adjusted_rect);
	  //cout << "bres: " << bres << endl;
          if (bres) {
            if (bSavePartRegions) {
              kma::ImageContent *region = crop_part_region(kmaimg, adjusted_rect, (int)(rects_scale[rectidx]*abc_param.desc_size()));
              QString qsRegionFilename = qsBootstrapTrainDir + 
                "/bootstrap_imgidx" + padZeros(QString::number(imgidx), 5) + 
                "_ridx" + QString::number(rectidx) + ".png"; 

              ExamplesCounting::all_bootstrap_image_names.push_back(qsRegionFilename);
              ExamplesCounting::all_bootstrap_images.push_back(region);
            }

            FeatureVector fv(all_features.size());
            fv.setData(all_features, 0);
            cout << fv.at(0) << endl;
            
            ExamplesCounting::all_bootstrap.push_back(fv);
          }

        }// bootstrap rectangles 

        delete kmaimg;

      }// images

    }// if bBootstrap

    /* add required number of bootstrapping examples */
    if (bBootstrap) {
      vector<uint> randperm;
      for (int idx = 0; idx < (int)ExamplesCounting::all_bootstrap.size(); ++idx) 
        randperm.push_back(idx);

      std::random_shuffle(randperm.begin(), randperm.end());
      uint max_idx = std::min((uint)ExamplesCounting::num_bootstrap_req, (uint)ExamplesCounting::all_bootstrap.size());
      
      for (uint idx = 0; idx < max_idx; ++idx) {
        assert(idx < randperm.size());
        assert(randperm[idx] < ExamplesCounting::all_bootstrap.size());

        trainingdata.push_back(ExamplesCounting::all_bootstrap[randperm[idx]]);
        ++ExamplesCounting::added_bootstrap;

        if (bSavePartRegions) {
          assert(randperm[idx] < ExamplesCounting::all_bootstrap_images.size());
          assert(randperm[idx] < ExamplesCounting::all_bootstrap_image_names.size());

          QString qsRegionFilename = ExamplesCounting::all_bootstrap_image_names[randperm[idx]];
          cout << "saving " + qsRegionFilename.toStdString() << endl;
          ExamplesCounting::all_bootstrap_images[randperm[idx]]->writePNG(qsRegionFilename.toStdString().c_str());
        }
      }
      
      for (uint idx = 0; idx < ExamplesCounting::all_bootstrap_images.size(); ++idx)
        delete ExamplesCounting::all_bootstrap_images[idx];

      ExamplesCounting::all_bootstrap_images.clear();
      ExamplesCounting::all_bootstrap_image_names.clear();
    }

    ExamplesCounting::done_collect_bootstrap();
    ExamplesCounting::print(cout);

    AnnotationList sampled_reshapedlist;
    sample_reshaped_points(reshapedlist, sampled_reshapedlist, part_conf, window_param, abc_param, pidx, pidx_window_param);
    
    for (uint aidx = 0; aidx < (int)reallist.size(); ++aidx)
      poslist.addAnnotation(reallist[aidx]);
    for (uint aidx = 0; aidx < sampled_reshapedlist.size(); ++aidx) 
      poslist.addAnnotation(sampled_reshapedlist[aidx]);
    
    /**                                             */
    /** collect positive training points            */
    /**                                             */
    for (uint imgidx = 0; imgidx < poslist.size(); ++imgidx) {
      cout << "computing positive features, image: " << imgidx << endl;
      
      // load image
      kma::ImageContent *input_image;
      int added_offset = 0;

      if (!bAddImageBorder) {
        input_image = kma::load_convert_gray_image(poslist[imgidx].imageName().c_str());
      }
      else {
        kma::ImageContent *_input_image = kma::load_convert_gray_image(poslist[imgidx].imageName().c_str());
        added_offset = part_image_offset;
	int corrected_offset;
        input_image = add_image_border2(_input_image, added_offset, corrected_offset);
	added_offset = corrected_offset;
	delete _input_image;
      }

      for (uint ridx = 0; ridx < poslist[imgidx].size(); ++ridx) {

        if (!annorect_has_part(poslist[imgidx][ridx], part_conf.part(pidx)))
          continue;
	
	// compute part bounding box
        PartBBox bbox;
        if (!get_part_bbox(poslist[imgidx][ridx], part_conf.part(pidx), bbox))
	  continue;
	  
	bbox.part_pos(0) += added_offset;
        bbox.part_pos(1) += added_offset;
      
	vector<float> all_features;
        // compute features
        PartBBox adjusted_rect;

        bool bres = compute_part_bbox_features(abc_param, window_param,
                                               bbox,
                                               input_image, pidx_window_param, all_features, 
                                               adjusted_rect);
	if (bres) {
          if (bSavePartRegions) {
            kma::ImageContent *region = crop_part_region(input_image, adjusted_rect, abc_param.desc_size());
            QString qsRegionFilename = qsPosTrainDir + "/pos_imgidx" + padZeros(QString::number(imgidx), 5) + 
              "_ridx" + QString::number(ridx) + ".png";

            cout << "saving " + qsRegionFilename.toStdString() << endl;
            region->writePNG(qsRegionFilename.toStdString().c_str());
	    // DEBUG
	    /*
	    if (poslist[imgidx][ridx].m_nObjectId == 100000){
	      QString qsMedoidFilename = qsPosTrainDir + "/pos_imgidx" + padZeros(QString::number(imgidx), 5) + 
		"_ridx" + QString::number(ridx) + "_medoid.png";

	      cout << "saving " + qsMedoidFilename.toStdString() << endl;
	      region->writePNG(qsMedoidFilename.toStdString().c_str());
	    }
	    */
	    // DEBUG
            delete region;
          }
          FeatureVector fv(all_features.size());
          fv.setData(all_features, 1);

          trainingdata.push_back(fv);
          cout << trainingdata.back().at(0) << endl;
      
          ++ExamplesCounting::added_pos;          
        }
        else {
          cout << "warning: positive bounding box outside of the image" << endl;
        }
        /** added jittered versions of the part rectangle */
	// jitter only real samples (dataType == 0)
        for (int jitidx = 0; jitidx < ExamplesCounting::num_jitter_per_pos && 
               ExamplesCounting::added_pos_jitter < ExamplesCounting::num_jitter_pos && poslist[imgidx].dataType() == 0; ++jitidx) {

          double jit_scale = jitter_scale_gen();
          double jit_rot = jitter_rot_gen();
          double jit_offsetx = jitter_offsetx_gen();
          double jit_offsety = jitter_offsety_gen();
          
          cout << "jitidx: " << jitidx << 
            ", scale: " << jit_scale << 
            ", rot: " << jit_rot << 
            ", x: " << jit_offsetx << 
            ", y: " << jit_offsety << endl;
            
          PartBBox jit_bbox = bbox;

          double_matrix R = boost_math::get_rotation_matrix(jit_rot);
          double_vector part_x_axis = prod(R, jit_bbox.part_x_axis);
          double_vector part_y_axis = prod(R, jit_bbox.part_y_axis);

          jit_bbox.part_x_axis = part_x_axis;
          jit_bbox.part_y_axis = part_y_axis;

          jit_bbox.part_pos(0) += jit_offsetx;
          jit_bbox.part_pos(1) += jit_offsety;
            
          PartBBox jit_adjusted_rect;
          vector<float> jit_all_features;
	  /*
	  bool bres = compute_part_bbox_features_scale(abc_param, window_param,
                                                       jit_bbox,
                                                       input_image, pidx, jit_scale,
                                                       jit_all_features,
                                                       jit_adjusted_rect);          
	  */
          bool bres = compute_part_bbox_features_scale(abc_param, window_param,
                                                       jit_bbox,
                                                       input_image, pidx_window_param, jit_scale,
                                                       jit_all_features,
                                                       jit_adjusted_rect);          
          if (bres) {
            if (bSavePartRegions) {
              kma::ImageContent *region = crop_part_region(input_image, jit_adjusted_rect, (int)(jit_scale*abc_param.desc_size()));
              QString qsRegionFilename = qsJitterTrainDir + "/pos_imgidx" +  padZeros(QString::number(imgidx), 5) + 
                "_ridx" + QString::number(ridx) + 
                "_jitidx" + QString::number(jitidx) + ".png";

              ExamplesCounting::all_jitter_image_names.push_back(qsRegionFilename);
              ExamplesCounting::all_jitter_images.push_back(region);
	      
            }

            FeatureVector fv(jit_all_features.size());
            fv.setData(jit_all_features, 1);
            cout << fv.at(0) << endl;

            ExamplesCounting::all_jitter.push_back(fv);
            
          }
        }// for jitter

      }// annorect

      delete input_image;

    }// positive images

    int nSkippedNegatives = 0;

    for (uint imgidx = 0; imgidx < neglist.size(); ++imgidx) {
      ExamplesCounting::added_neg_curimg = 0;

      bool bSkip = false;
      for (uint ridx = 0; ridx < neglist[imgidx].size(); ++ridx) 
        if (!annorect_has_part(neglist[imgidx][ridx], part_conf.part(pidx))) {
          ++nSkippedNegatives;
          bSkip = true;
          break;
        }

      if (bSkip) 
        continue;

      // load image
      kma::ImageContent *input_image;
      int added_offset = 0;

      if (!bAddImageBorder) {
        input_image = kma::load_convert_gray_image(neglist[imgidx].imageName().c_str());
      }
      else {
        kma::ImageContent *_input_image = kma::load_convert_gray_image(neglist[imgidx].imageName().c_str());
        added_offset = part_image_offset;
	int corrected_offset;
        input_image = add_image_border2(_input_image, added_offset, corrected_offset);
	added_offset = corrected_offset;
	part_image_offset = corrected_offset;	
	delete _input_image;
      }
    
      cout << "extracting negatives from image " << imgidx << endl;

      // random numbers for sampling of negative rectangles
      boost::mt19937 rng(42u * imgidx);
      boost::uniform_real<> dist_01(0.0, 1.0);
      boost::variate_generator<boost::mt19937, boost::uniform_real<> > gen_01(rng, dist_01);

      int min_pos_x = part_image_offset; 
      int max_pos_x = input_image->x() - part_image_offset - 1;

      int min_pos_y = part_image_offset;
      int max_pos_y = input_image->y() - part_image_offset - 1;

      // skip the image if can't extract negative poselets
      if (part_conf.part(pidx).has_mult_types()){
	if (not((min_pos_x < max_pos_x) && 
		(min_pos_y < max_pos_y))){
	  ++nSkippedNegatives;
	  continue;
	}
      }
      else{
	assert(min_pos_x < max_pos_x);
	assert(min_pos_y < max_pos_y);
      }
      
      //int neg_count = 0;
      //while ( neg_count < nNegativePerImage) {
      while (ExamplesCounting::added_neg_curimg < ExamplesCounting::num_neg_random_per_image) {

        PartBBox sample_rect;
	
        double scale;
        double rot;
	sample_random_partrect(gen_01,
                               1.0, 1.0,
                               exp_param.min_part_rotation()/180.0*M_PI, exp_param.max_part_rotation()/180.0*M_PI,
                               min_pos_x, max_pos_x,
                               min_pos_y, max_pos_y,
                               window_param.part(pidx_window_param).window_size_x(), 
                               window_param.part(pidx_window_param).window_size_y(), 
                               sample_rect, scale, rot);

        bool bsampleok = true;
        // test that bounding box is on the background
        for (uint ridx = 0; ridx < neglist[imgidx].size(); ++ridx) {

          if (sample_rect.part_pos(0) >= (neglist[imgidx][ridx].left() + added_offset) && 
              sample_rect.part_pos(0) <= (neglist[imgidx][ridx].right() + added_offset)  &&
              sample_rect.part_pos(1) >= (neglist[imgidx][ridx].top() + added_offset) && 
              sample_rect.part_pos(1) <= (neglist[imgidx][ridx].bottom() + added_offset)) {
            bsampleok = false;
            break;
          }
        }// annorects
	
	// add negatives
        if (bsampleok) {
          vector<float> all_features;
          // compute features
          PartBBox adjusted_rect;

	  bool bres = compute_part_bbox_features(abc_param, window_param,
                                                 sample_rect, 
                                                 input_image, pidx_window_param, all_features, 
                                                 adjusted_rect);
        
          if (bres) {
            if (bSavePartRegions) {
              kma::ImageContent *region = crop_part_region(input_image, adjusted_rect, abc_param.desc_size());
              QString qsRegionFilename = qsNegTrainDir + "/neg_imgidx" + padZeros(QString::number(imgidx), 5) + "_" +
                padZeros(QString::number(ExamplesCounting::added_neg_curimg), 3) + ".png";

              ExamplesCounting::all_negatives_image_names.push_back(qsRegionFilename);
              ExamplesCounting::all_negatives_images.push_back(region);

              //cout << "saving " + qsRegionFilename.toStdString() << endl;
              //region->writePNG(qsRegionFilename.toStdString().c_str());
              //delete region;
            }

            FeatureVector fv(all_features.size());
            fv.setData(all_features, 0);

            assert(fv.numDims() > 0);
            cout << fv.at(0) << endl;

            ExamplesCounting::all_negatives.push_back(fv);

            //++ExamplesCounting::added_neg;
            ++ExamplesCounting::added_neg_curimg;

          }
          else {
	    cout << "part_image_offset: " << part_image_offset << endl;
            cout << "sample_rect.part_pos: " << sample_rect.part_pos(0) << " " << sample_rect.part_pos(1) << endl;
	    cout << "sample_rect.min_proj: " << sample_rect.min_proj_x << " " << sample_rect.min_proj_y << endl;
	    cout << "sample_rect.max_proj: " << sample_rect.max_proj_x << " " << sample_rect.max_proj_y << endl;
            cout << "image_width: " << input_image->x() << ", image_height: " << input_image->y() << endl;
            assert(false && "random rectangle outside of the image");
          }    

        }// if sample ok
        
      }// sample negatives


      delete input_image;

      //if (ExamplesCounting::added_neg >= ExamplesCounting::num_neg_random)
      //  break;
    
    }// negative images

    /* add required number of jittered positives */
    if (abc_param.do_jitter()) {
      //assert(ExamplesCounting::all_jitter.size() >= ExamplesCounting::num_jitter_pos);

      vector<uint> randperm;
      for (int idx = 0; idx < (int)ExamplesCounting::all_jitter.size(); ++idx)
        randperm.push_back(idx);

      std::random_shuffle(randperm.begin(), randperm.end());

      uint max_idx = std::min((uint)ExamplesCounting::num_jitter_pos, (uint)ExamplesCounting::all_jitter.size());

      for (uint idx = 0; idx < max_idx; ++idx) {
        assert(idx < randperm.size());
        assert(randperm[idx] < ExamplesCounting::all_jitter.size());
        
        trainingdata.push_back(ExamplesCounting::all_jitter[randperm[idx]]);
        ++ExamplesCounting::added_pos_jitter;

        if (bSavePartRegions) {
          assert(randperm[idx] < ExamplesCounting::all_jitter_images.size());
          assert(randperm[idx] < ExamplesCounting::all_jitter_image_names.size());

          QString qsRegionFilename = ExamplesCounting::all_jitter_image_names[randperm[idx]];
          cout << "saving " + qsRegionFilename.toStdString() << endl;
          ExamplesCounting::all_jitter_images[randperm[idx]]->writePNG(qsRegionFilename.toStdString().c_str());
        }
      }

      for (uint idx = 0; idx < ExamplesCounting::all_jitter_images.size(); ++idx) 
        delete ExamplesCounting::all_jitter_images[idx];
        
      ExamplesCounting::all_jitter_images.clear();
      ExamplesCounting::all_jitter_image_names.clear();
    }

    /* add required number of negatives */
    {
      assert(ExamplesCounting::all_negatives.size() == 
             (neglist.size() - nSkippedNegatives) * ExamplesCounting::num_neg_random_per_image);
      
      assert((int)ExamplesCounting::all_negatives.size() >= ExamplesCounting::num_neg_random);
      vector<uint> randperm;
      for (int idx = 0; idx < (int)ExamplesCounting::all_negatives.size(); ++idx)
        randperm.push_back(idx);

      std::random_shuffle(randperm.begin(), randperm.end());
      for (uint idx = 0; idx < (uint)ExamplesCounting::num_neg_random; ++idx) {
        assert(idx < randperm.size());
        assert(randperm[idx] < ExamplesCounting::all_negatives.size());

        trainingdata.push_back(ExamplesCounting::all_negatives[randperm[idx]]);
        ++ExamplesCounting::added_neg;

        if (bSavePartRegions) {
          assert(randperm[idx] < ExamplesCounting::all_negatives_images.size());
          assert(randperm[idx] < ExamplesCounting::all_negatives_image_names.size());

          QString qsRegionFilename = ExamplesCounting::all_negatives_image_names[randperm[idx]];
          cout << "saving " + qsRegionFilename.toStdString() << endl;

          ExamplesCounting::all_negatives_images[randperm[idx]]->writePNG(qsRegionFilename.toStdString().c_str());
        }
      }     

      for (uint idx = 0; idx < ExamplesCounting::all_negatives_images.size(); ++idx) 
        delete ExamplesCounting::all_negatives_images[idx];

      ExamplesCounting::all_negatives_images.clear();
      ExamplesCounting::all_negatives_image_names.clear();
    }

    /* test */
    int nD = trainingdata[0].numDims();
    for (uint fidx = 0; fidx < trainingdata.size(); ++fidx) {
      if (trainingdata[fidx].numDims() != nD){
	cout << fidx << endl;
	cout << trainingdata[fidx].numDims() << endl;
	cout << nD << endl;
      }
      assert(trainingdata[fidx].numDims() == nD);
    }
    ExamplesCounting::print(cout);

    /** save training log */
    QString qsLogFile = part_app.m_exp_param.class_dir().c_str();
    qsLogFile += "/part" + QString::number(pidx);
    QString qsTypeSuff = "";
    if (tidx > -1)
      qsTypeSuff = QString("_type") + QString::number(tidx);
    qsLogFile += qsTypeSuff;
    if (bBootstrap)
      qsLogFile += "-bootstrap";
    qsLogFile += ".trainlog";
    
    std::fstream fstr_out(qsLogFile.toStdString().c_str(), std::ios::out | std::ios::trunc);
    cout << "saving log to: " << qsLogFile.toStdString() << endl;
    ExamplesCounting::print(fstr_out);

    cout << "total number of training points: " << trainingdata.size() << endl;
    cout << "point dimension: " << nD << endl;

    // init boosting struct
    Data boostingData(nD, 2);
    boostingData.addSampleVector(trainingdata);

    cout << boostingData.getC() << endl;
    cout << boostingData.getN() << endl;
    cout << boostingData.getD() << endl;

    // boosting 
    cout << "training classifier for "  << abc_param.boosting_rounds() << " rounds ..." << endl;

    //AdaBoostClassifier abc;
    abc.trainClassifier(&boostingData, abc_param.boosting_rounds(), &cout);

    // compute test set error
    double threshold = 0;
    float correct_rate = 0;
    for (uint ptidx = 0; ptidx < trainingdata.size(); ++ptidx) {
      float margin = abc.evaluateFeaturePoint(trainingdata[ptidx], true);
    
      int predicted_class = (margin <= threshold) ? 0:1;
    
      cout << "true label: " << trainingdata[ptidx].getTargetClass() << ", margin: " << margin << endl;
      correct_rate += (int)(predicted_class == trainingdata[ptidx].getTargetClass());
    }

    correct_rate /= trainingdata.size();
    cout << "misclassification rate:" << 1 - correct_rate << endl;

    /** train classifier - end */

    int bootstrap_type = 0;
    if (bBootstrap)
      bootstrap_type = 1;

    part_app.saveClassifier(abc, pidx, bootstrap_type, tidx);

    
  }

  /**
     compute features on the part bounding box 

     abc_param - size of descriptors, step between descriptors and size of part window
     bbox - positions of the descriptors
   
     input_image
     pidx
     all_features

     features are normalized using part orientation

     improvement over the previous version:
     - smooth whole image instead of each descriptor patches
     - normalize with part rotation and map to descriptor patch in single operation
  */

  bool compute_part_bbox_features_scale(const AbcDetectorParam &abc_param, const PartWindowParam &window_param, 
                                        const PartBBox &bbox, 
                                        kma::ImageContent *input_image, int pidx, double scale, 
                                        vector<float> &all_features,                                 
                                        PartBBox &adjusted_rect)
  {
    adjusted_rect = bbox;

    /* first center average box on the part bounding box */
    float cx = 0.5*(adjusted_rect.min_proj_x + adjusted_rect.max_proj_x);
    float cy = 0.5*(adjusted_rect.min_proj_y + adjusted_rect.max_proj_y);

    adjusted_rect.min_proj_x = cx - 0.5*scale*window_param.part(pidx).window_size_x();
    adjusted_rect.min_proj_y = cy - 0.5*scale*window_param.part(pidx).window_size_y();

    adjusted_rect.max_proj_x = adjusted_rect.min_proj_x + scale*window_param.part(pidx).window_size_x() - 1;
    adjusted_rect.max_proj_y = adjusted_rect.min_proj_y + scale*window_param.part(pidx).window_size_y() - 1;
    
    FeatureGrid grid(adjusted_rect, scale*abc_param.desc_step(), boost_math::round(scale*abc_param.desc_size()));
    
    /* make sure feature counts are consistent */
    int grid_x_count, grid_y_count;
    get_window_feature_counts(abc_param, window_param.part(pidx), grid_x_count, grid_y_count);
    grid.nx = grid_x_count;
    grid.ny = grid_y_count;
  
    grid.desc.clear();
    grid.desc.resize(grid.ny, vector<vector<float> >(grid.nx, vector<float>()));
    
    //assert(grid.nx == grid_x_count && grid.ny == grid_y_count);

    /* compute descriptors */
    computeDescriptorGrid(input_image, grid, abc_param.feature_type().c_str());

    /* build feature vector */
    bool bres = grid.concatenate(0, 0, grid.nx, grid.ny, 1, all_features);
    return bres;
  }

  bool compute_part_bbox_features(const AbcDetectorParam &abc_param, const PartWindowParam &window_param, 
                                  const PartBBox &bbox, 
                                  kma::ImageContent *input_image, int pidx, vector<float> &all_features, 
                                  PartBBox &adjusted_rect)
  {
    double scale = 1.0;
    return compute_part_bbox_features_scale(abc_param, window_param, bbox, input_image, pidx, scale, 
                                            all_features, adjusted_rect);

    int grid_x_count, grid_y_count;
    get_window_feature_counts(abc_param, window_param.part(pidx), grid_x_count, grid_y_count);
    assert((int)all_features.size() == grid_x_count*grid_y_count);
  }

}// namespace 
