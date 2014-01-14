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

#ifndef _PART_DETECT_H_
#define _PART_DETECT_H_

#include <vector>
// boost::random (needed to generate random rectangles for negative set)
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

#include <libAnnotation/annotationlist.h>

#include <libPartDetect/PartConfig.pb.h>
#include <libPartDetect/AbcDetectorParam.pb.h>
#include <libPartDetect/PartWindowParam.pb.h>

#include <libPartDetect/partdef.h>
#include <libPartDetect/FeatureGrid.h>

#include <libPartApp/partapp.h>

#include <libKMA2/kmaimagecontent.h>

#include <libPartApp/partapp_aux.hpp>
#include <libPictStruct/objectdetect.h>

namespace part_detect {

  const int NO_CLASS_VALUE = 0;

  /****************************************
  
   partdetect_test.cpp

  ****************************************/
  
  /**
     this function is used for bootstrapping and in partdetect
  */
  void partdetect_dense(const ExpParam &exp_param, const AbcDetectorParam &abc_param, 
			const PartWindowParam &window_param, 
			const AnnotationList &test_annolist, std::vector<AdaBoostClassifier> &v_abc,
			QString qsScoreGridDir, 
			int firstidx, int lastidx, bool flip, 
			bool bSaveImageScoreGrid, bool bAddImageBorder, 
			std::vector<std::vector<FloatGrid3> > &part_detections, 
			bool bSaveScoreGrid = true);
  
  void partdetect_dense_mix(const ExpParam &exp_param, const AbcDetectorParam &abc_param, 
			    const PartConfig &partconf, const PartWindowParam &window_param, 
			    const AnnotationList &test_annolist, std::vector<AdaBoostClassifier> &v_abc,
			    QString qsScoreGridDir, int firstidx, int lastidx, bool flip, 
			    bool bSaveImageScoreGrid, bool bAddImageBorder);

  void partdetect(const PartApp &part_app, int firstidx, int lastidx, bool flip, bool bSaveImageScoreGrid, 
		  std::vector<std::vector<FloatGrid3> > &part_detections, const AnnotationList &annolist,
		  QString qsScoreGridDir,
		  bool bSaveScoreGrid = true);

  void computeScoreGrid(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param,
			const AdaBoostClassifier &abc, double grid_step, double part_scale,  
			FeatureGrid &feature_grid, ScoreGrid &score_grid, bool bSqueeze = true, bool bNormalizeBoost = true);

  void squeezeScoreGrid(ScoreGrid &score_grid);


  /****************************************
  
   partdetect_train.cpp

  ****************************************/

  void make_bbox(const PartWindowParam &window_params, int pidx, 
                        double part_center_x, double part_center_y, double scale, double rot, 
                        PartBBox &bbox);

  bool is_point_in_rect(PartBBox bbox, double point_x, double point_y);

  void bootstrap_get_rects(const PartApp &part_app, int imgidx, int pidx, 
			   int num_rects, double min_score, 
			   std::vector<PartBBox> &rects, std::vector<double> &rects_scale,
			   bool bIgnorePartRects, bool bDrawRects, int pidx_window_param);

  void bootstrap_partdetect(PartApp &part_app, int firstidx, int lastidx);

  void prepare_bootstrap_dataset(const PartApp &part_app, const AnnotationList &annolist, 
				 int firstidx, int lastidx);
  

  /**
     train AdaBoost part classifier
  */
  
  //void abc_train_class(PartApp &part_app, int pidx, bool bBootstrap);
  void abc_train_class(PartApp &part_app, int pidx, bool bBootstrap, int tidx = -1);


  /**
     compute features for part bounding box
  */
  bool compute_part_bbox_features(const AbcDetectorParam &abcparam, const PartWindowParam &window_param, 
				  const PartBBox &bbox, 
				  kma::ImageContent *input_image, int pidx, std::vector<float> &all_features, 
				  PartBBox &adjusted_rect);


  bool compute_part_bbox_features_scale(const AbcDetectorParam &abcparam, const PartWindowParam &window_param, 
					const PartBBox &bbox, 
					kma::ImageContent *input_image, int pidx, double scale, 
					std::vector<float> &all_features,                                 
					PartBBox &adjusted_rect);
  kma::ImageContent* crop_part_region(kma::ImageContent *image, const PartBBox &bbox, int desc_size);

  /****************************************
  
   partdetect_aux.cpp

  ****************************************/

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC);
  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC, const FloatGrid2 &detection_mask);
  void computeDescriptorGridRoi(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC, const QRect& roi);

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC, 
			     const FloatGrid2 &detection_mask, const QRect& roi);

  void getDescCoord(FeatureGrid &grid, FloatGrid2 &imgposGrid);
  
  int get_app_group_idx(const PartConfig &part_conf, int pid);

  int pidx_from_pid(const PartConfig &part_conf, int pid);


  /** 
      determine how many features fit into detection window for given object part     
      (number of features depends of window size and distance between features)
  */
  void get_window_feature_counts(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param, 
				 int &grid_x_count, int &grid_y_count);

  void sample_random_partrect(boost::variate_generator<boost::mt19937, boost::uniform_real<> > &gen_01,
			      double min_scale, double max_scale,
			      double min_rot, double max_rot, 
			      double min_pos_x, double max_pos_x,
			      double min_pos_y, double max_pos_y,
			      int rect_width, int rect_height, 
			      PartBBox &partrect, double &scale, double &rot);
  
  /****************************************
  
   partdetect_icps.cpp

  ****************************************/
     
  /**
     compute average size of part bounding box
  */
  void compute_part_type_window_param(const AnnotationList &annolist, const PartConfig &partconf, 
				      PartWindowParam &windowparam, const ExpParam &exp_param, QString qsClassDir);  
  void get_part_type_clusters(const AnnotationList &annolist, const PartConfig &partconf, const ExpParam &exp_param,
			      std::vector<AnnotationList> &clusterlist, int pidx);
  int getPartById(const PartWindowParam &windowparam, int pidx, int tidx);
  void loadPartTypeData(QString qsClassDir, const PartWindowParam &windowparam, const PartConfig &partconf, int pidx, std::vector<AnnotationList> &annolistClusters, QString qsListName = "train");
  void saveCombResponces(const PartApp &part_app, int imgidx, QString qsDetRespDir, QString qsScoreGridDir, 
			 QString qsImgName, int root_pos_x, int root_pos_y, float root_rot, 
			 std::vector<std::vector<FloatGrid3> > &part_detections, bool bLoadScoreGrid);
  int getNumPartTypes(const PartWindowParam &windowparam, int pidx);
  
  int partdetect_dpm(const PartApp &part_app, int imgidx, QString qsModelDPMDir, QString qsUnaryDPMDir);
  int partdetect_dpm_all(const PartApp &part_app, int imgidx);
  void runMatlabCode(QString qsCommandLine);


}// namespace 

#endif 
