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

#ifndef _OBJECT_DETECT_H_
#define _OBJECT_DETECT_H_

#include <vector>

#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include <libAnnotation/annotation.h>
#include <libAnnotation/annotationlist.h>

#include <libPartApp/partapp.h>

#include <libPartDetect/PartConfig.pb.h>

#include <libBoostMath/boost_math.h>

#include <libMultiArray/multi_array_def.h>

#include <libPictStruct/HypothesisList.pb.h>

#include <libKMA2/kmaimagecontent.h>

namespace object_detect {
  
  enum ScoreProbMapType {
    SPMT_NONE = 0,
    SPMT_RATIO = 1,
    SPMT_RATIO_WITH_PRIOR = 2,
    SPMT_SPATIAL = 3,
    SPMT_SPATIAL_PROD = 4
  };
  
  typedef HypothesisList::ObjectHypothesis ObjectHypothesis;

  int JointTypeIdFromString(QString qsType);
  
  struct Joint {
    enum {POS_GAUSSIAN = 1, ROT_GAUSSIAN = 2};

    Joint():rot_mean(0), rot_sigma(0), detC(0) {}

    Joint(int _type, int _child_idx, int _parent_idx, 
	  boost_math::double_vector _offset_c, 
	  boost_math::double_vector _offset_p, 
	  boost_math::double_matrix _C);

    Joint(int _type, int _child_idx, int _parent_idx, 
	  boost_math::double_vector _offset_c, 
	  boost_math::double_vector _offset_p, 
	  boost_math::double_matrix _C, 
	  double _rot_mean, 
	  double _rot_sigma);

    int type;
    int mix_comp_id;
    
    int child_idx; // id of child part (not index, i.e. this is indepent of the order in which parts are stored)
    int parent_idx; // id of parent part

    boost_math::double_vector offset_c; // parent_pos - child_pos (in pixels)
    boost_math::double_vector offset_p; // child_pos - parent_pos (in pixels)
    boost_math::double_matrix C; 

    double rot_mean;
    double rot_sigma;

    double detC;
    boost_math::double_matrix invC;
  };

  struct PartHyp {
  PartHyp():m_scaleidx(-1), m_rotidx(-1), m_x(-1), m_y(-1), m_score(-1e6), m_imgidx(-1) {}
    
  PartHyp(const ExpParam &exp_param, int scaleidx, int rotidx, int x, int y, float score): m_scaleidx(scaleidx), 
      m_rotidx(rotidx), m_x(x), m_y(y), m_score(score) 
    {
      m_scale = scale_from_index(exp_param, scaleidx);

      m_rot = rot_from_index(exp_param, rotidx); // rotation in degrees
    }

  PartHyp(const ExpParam &exp_param, int scaleidx, int rotidx, int x, int y, float score, int imgidx): m_scaleidx(scaleidx), 
      m_rotidx(rotidx), m_x(x), m_y(y), m_score(score), m_imgidx(imgidx) 
    {
      m_scale = scale_from_index(exp_param, scaleidx);

      m_rot = rot_from_index(exp_param, rotidx); // rotation in degrees
    }

    void getPartBBox(const PartWindowParam::PartParam &part_param, PartBBox &bbox) 
    { 
      assert(m_scaleidx >= 0 && m_rotidx >= 0); 

      bbox_from_pos(part_param, m_scale, m_rot / 180 * M_PI, m_x, m_y, bbox);
      //bbox_from_pos(exp_param, part_param, m_scaleidx, m_rotidx, m_x, m_y, bbox); 
    }

    void getPartBBox2(const ExpParam &exp_param, const PartWindowParam::PartParam &part_param, PartBBox &bbox) 
    { 
      assert(m_scaleidx >= 0 && m_rotidx >= 0); 

      bbox_from_pos(exp_param, part_param, m_scaleidx, m_rotidx, m_x, m_y, bbox); 
    }
    
    AnnoRect toAnnoRect(const PartWindowParam::PartParam &part_param) const
    {
      PartBBox bbox;
      bbox_from_pos(part_param, m_scale, m_rot / 180 * M_PI, m_x, m_y, bbox);
      AnnoRect annoRect(bbox.min_proj_x + bbox.part_pos(0),
			bbox.min_proj_y + bbox.part_pos(1),
			bbox.max_proj_x + bbox.part_pos(0),
			bbox.max_proj_y + bbox.part_pos(1));
      return annoRect;
    }
    
    static uint vectSize() 
    {
      return 7;
    }

    //FloatGrid1 toVect(const ExpParam &exp_param) const
    FloatGrid1 toVect() const
    {
      FloatGrid1 r(boost::extents[vectSize()]);

      int idx = 0;
      r[idx++] = m_scaleidx;
      r[idx++] = m_scale;
      //r[idx++] = scale_from_index(exp_param, m_scaleidx);

      r[idx++] = m_rotidx;
      r[idx++] = m_rot;
      //r[idx++] = rot_from_index(exp_param, m_rotidx);

      r[idx++] = m_x;
      r[idx++] = m_y;

      r[idx++] = m_score;

      assert(idx == (int)vectSize());
      return r;
    }

    template <typename Array1d>
    void fromVect(const Array1d &r)
    {
      assert(Array1d::dimensionality == 1);
      assert(r.shape()[0] == vectSize());

      //double scale;
      //double rot;

      int idx = 0;

      m_scaleidx = r[idx++];
      m_scale = r[idx++];
      m_rotidx = r[idx++];
      m_rot = r[idx++];

      m_x = r[idx++];
      m_y = r[idx++];
      m_score = r[idx++];

      assert(idx == (int)vectSize());
    }

    // Leonid: needed for collecting true/false positives from different images
    int m_imgidx;
    
    int m_scaleidx;
    int m_rotidx;
    int m_x, m_y;

    float m_score;

    float m_scale;
    float m_rot;   
  };



  /****************************************
  
   objectdetect_learnparam.cpp

  ****************************************/
  void save_joint(const PartApp &part_app, Joint &joint, int tidx = -1);
  void load_joint(const PartApp &part_app, int jidx, Joint &joint, int tidx = -1);

  void learn_conf_param(const PartApp &part_app, const AnnotationList &train_annolist);
  void learn_conf_param_pred_data(const PartApp &part_app);

  /****************************************
  
   objectdetect_aux.cpp

  ****************************************/

  void loadJoints(const PartApp &part_app, std::vector<Joint> &joints, bool flip, int imgidx = -1, bool bIsTest = true);

  void findLocalMax(const FloatGrid3 &log_prob_grid, std::vector<boost_math::double_vector> &local_max, int max_hypothesis_number);

  void findLocalMax(const ExpParam &exp_param, const FloatGrid3 &log_prob_grid, 
                    HypothesisList &hypothesis_list,
                    int max_hypothesis_number);

  void findLocalMax(const ExpParam &exp_param, const FloatGrid3 &log_prob_grid,  
   		    std::vector<PartHyp> &part_hyp,  
   		    int max_hypothesis_number); 

  int JointTypeFromString(QString qsType);

  int MapTypeFromString(QString qsScoreProbMapType);

  QString MapTypeToString(int maptype) ;
    
  QString getObjectHypFilename(int imgidx, bool flip, int scoreProbMapType);

  void findObjectDataset(const PartApp &part_app, int firstidx, int lastidx, int scoreProbMapType);
  void findObjectDatasetMix(const std::vector<PartApp> &partapp_components, int firstidx, int lastidx);
  
  void nms_recursive(const HypothesisList hypothesis_list, 
                     std::vector<bool> &nms, 
                     double train_object_width, 
                     double train_object_height);

  void saveRecoResults(const PartApp &part_app, int scoreProbMapType);

  /****************************************
  
   objectdetect_findpos.cpp

  ****************************************/

  void findObjectImagePosJoints(const PartApp &part_app, int imgidx, bool flip, HypothesisList &hypothesis_list,  
 		       int scoreProbMapType); 

  /****************************************
  
   objectdetect_findrot.cpp

  ****************************************/

  void get_incoming_joints(const std::vector<Joint> &joints, int curidx, std::vector<int> &all_children, std::vector<int> &all_joints);
  
  void computeRootPosteriorRot(const PartApp part_app, 
                               std::vector<std::vector<FloatGrid3> > &log_part_detections, 
                               FloatGrid3 &root_part_posterior, int rootpart_idx, std::vector<Joint> joints, bool flip, 
                               bool bIsSparse, int imgidx);

  /** best_part_hyp is a set of local maxima of the marginal of each part */
  void computeRootPosteriorRot(const PartApp part_app, 
                               std::vector<std::vector<FloatGrid3> > &log_part_detections, 
                               FloatGrid3 &root_part_posterior, int rootpart_idx, std::vector<Joint> joints, bool flip, 
                               bool bIsSparse, int imgidx, 
			       std::vector<std::vector<PartHyp> >& best_part_hyp, 
			       bool bSaveMarginals);


  void computeRotJointMarginal(const ExpParam &exp_param, 
                               FloatGrid3 &log_prob_child, FloatGrid3 &log_prob_parent, 
                               const boost_math::double_vector &_offset_c_10, const boost_math::double_vector &_offset_p_01, 
                               const boost_math::double_matrix &C, 
                               double rot_mean, double rot_sigma,
                               double scale, bool bIsSparse);
  
  void findObjectImageRotJoints(const PartApp &part_app, int imgidx, 
				bool flip, HypothesisList &hypothesis_list, 
				int scoreProbMapType,
				QString qsPartMarginalsDir,
				QString qsScoreGridDir, QString qsImgName, 
				std::vector<std::vector<FloatGrid3> > &log_part_detections,
				bool bLoadScoreGrid = true);

  double get_runtime();
  
  /****************************************
  
   objectdetect_roi.cpp

  ****************************************/

  void findObjectImageRoi(PartApp part_app, int imgidx);

  void findObjectRoiHelper(PartApp part_app, AnnoRect &rect, QString qsImageName,
  			   const std::vector<AdaBoostClassifier> &v_abc,
  			   const std::vector<Joint> joints,
  			   std::vector<std::vector<PartHyp> > &best_part_det,
  			   std::vector<std::vector<PartHyp> > &best_part_hyp,
			   QString qsScoreGridDir = "",
			   int imgidx = -1, bool bRandConf = false);
  
  /****************************************
  
   objectdetect_icps.cpp

  ****************************************/
  void getRotParams(const PartApp &part_app, int imgidx, boost_math::double_matrix &rot_params, bool bTest);
  void getRotScoreGrid(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_rot_scores, boost_math::double_matrix &rot_params);
  void getRootPosDet(const PartApp &part_app, int imgidx, int rootpart_idx, boost_math::double_vector &rootpos_det, bool bTest);
  void getPosParams(const PartApp &part_app, int imgidx, boost_math::double_matrix &pos_params, int rootpart_idx, bool bTest);
  void getPosScoreGrid(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_pos_scores, int imgidx, boost_math::double_matrix &pos_params, int rootpart_idx, boost_math::double_vector &rootpos_det);
  void getTorsoPosPriorParams(const PartApp &part_app, int img_height, int img_width, boost_math::double_matrix &pos_prior_params);
  void setPosScoreGrid(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, int imgidx, int rootpart_idx);
  void setTorsoPosPrior(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, boost_math::double_matrix &pos_prior_params, int rootpart_idx);
  void setPartMixComp(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, int imgidx, QString qsImgName);
    
  void loadDPMScoreGrid(QString qsDPMdir, int imgidx, std::vector<FloatGrid2> &dpmPriorGrid, bool bIsCell = true);
  void addDPMScore(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, std::vector<FloatGrid2> log_dpmPriorGrid, int pidx, float dpm_weight);
  void addLoadDPMScore(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, int imgidx, float dpm_weight, int nRotationsDPM, QString qsDPMparentDir, bool bUsePartIdx, int pidx_only = -1);
  
  void savePredictedPartConf(const PartApp &part_app, int firstidx, int lastidx, bool bIsTest);
  void addExtraUnary(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, 
		     const std::vector<std::vector<FloatGrid3> > &log_extra_unary_scores,
		     float weight);
  
  void computeTorsoPosPriorParams(const PartApp &part_app);

  void computePoseLL(const PartApp &part_app, int firstidx, int lastidx);
  int trainLDA(const PartApp &part_app, int idxMode, int idxFactor = -1);
  int predictFactors(const PartApp &part_app, int idxMode, int imgidx);
  
}
#endif
