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

#ifndef _PART_APP_H_
#define _PART_APP_H_

#include <vector>
#include <QString>

#include <libAdaBoost/AdaBoost.h>
#include <libAnnotation/annotationlist.h>

#include <libMultiArray/multi_array_def.h>

#include <libPartDetect/AbcDetectorParam.pb.h>
#include <libPartDetect/PartConfig.pb.h>
#include <libPartDetect/PartWindowParam.pb.h>
#include <libPartDetect/partdef.h>

#include <libPictStruct/objectdetect_aux.hpp>
//#include <libSVMstruct/maxmargin_learning.h>

#include "ExpParam.pb.h"

QString complete_relative_path(QString qsInputFile, QString qsReferenceFile);

/**
   this function is implemented in libPartApp since it is used in both parteval.cpp and scoreparams_test.cpp
   and i did not find a better place for it
 */

void bbox_from_pos(const ExpParam &exp_param, const PartWindowParam::PartParam &part_param, 
                   int scaleidx, int rotidx, int ix, int iy, 
                   PartBBox &bbox);

void bbox_from_pos(const PartWindowParam::PartParam &part_param, 
                   double scale, double rot, int ix, int iy, 
                   PartBBox &bbox);



/** 
    PartApp is a container for options/datasets used in the experiment 
    (better name would be something like ExpDef)

    here we make a library out of it since it is used in 
      libPartDetect 
      libPictStruct
      apps/partapp

    all of which are potentially independent
*/

class PartApp {
  
 public:
 
  /* name of the expopt file (typically provided from the command line)*/
  QString m_qsExpParam;
  
  /* specify whether partapp is allowed to update the classifiers and joint parameters */
  bool m_bExternalClassDir;
  
  /* specify whether partapp is allowed to update the scoregrids */
  bool m_bExternalScoregridDir;
  
  int m_rootpart_idx;
  
  /* specify whether partapp is allowded to update the samples dir */
  bool m_bExternalSamplesDir;

  /* multi-component support */
  //std::vector<int> m_compidx; // for each component: idx of the component in the training data (currently silhouette id)
  std::vector<int> m_validation_idx;
  std::vector<int> m_validation_all_idx;
  //std::vector<int> m_num_parttypes;
  
  int m_compidx;
  
  ExpParam m_exp_param;
  PartConfig m_part_conf;
  AbcDetectorParam m_abc_param;
  PartWindowParam m_window_param;
  
  AnnotationList m_train_annolist;
  AnnotationList m_train_reshaped_annolist;
  AnnotationList m_validation_annolist;
  AnnotationList m_test_annolist;
  AnnotationList m_neg_annolist;
  AnnotationList m_bootstrap_annolist;

 PartApp(int compidx = 0) : m_bExternalClassDir(false), m_rootpart_idx(-1), m_bExternalSamplesDir(false) {m_compidx = compidx;}
  
  void init(QString qsExpParam);

  /**
     new and somewhat adhoc: support for multiple components, each component corresponding to 
     a different pictorial structures model

     return PartApp that is single-component
     
     the returned PartApp 
      - has log_dir inside the log_dir of the original PartApp
      - shares AdaBoost/ShapeContext parameters and body configuration
      - what to do with external class_dir ?
   */
  static void init_multicomponent(QString qsExpParam, std::vector<PartApp> &partapp_components);

  void init_setpath(QString qsExpParam, bool is_multicomp);
  void init_loaddata();
  void init_partdims();
  
  //int getCompId(){return m_compidx;}
  //void setCompId(int compidx){m_compidx = compidx;}
 
  QString getClassFilename(int pidx, int bootstrap_type, int tidx = -1) const;

  /* bootstrap_type: 
       0 - no bootstrapping 
       1 - bootstrapping 
       2 - first try to load classifier with bootstrapping, if unavailable load the classifier without bootstrapping
   */

  void loadClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type = 2, int tidx = -1) const;
  void saveClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type = 2, int tidx = -1) const;

  void getScoreGridFileName(int imgidx, int pidx, bool flip,
			    QString &qsFilename, QString &qsVarName, QString qsScoreGridDir) const;

  void loadScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
		     bool flip, bool bInterpolate, QString qsScoreGridDir, QString qImgName) const;
  void loadScoreGridRotRange(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
			     bool flip, bool bInterpolate, QString qsScoreGridDir, QString qsImgName, std::vector<int> &rotidx_transform) const;
  
  void saveScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx,  
		     bool flip) const; 

  FloatGrid3 loadPartMarginal(int imgidx, int pidx, int scaleidx, bool flip) const;

};


#endif
