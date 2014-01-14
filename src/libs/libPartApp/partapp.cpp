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

#include <iostream>
#include <algorithm>

#include <QDir>

#include <libAnnotation/annotationlist.h>

#include <libFilesystemAux/filesystem_aux.h>
#include <libPartDetect/partdetect.h>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libMatlabIO/matlab_io.h>
#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_transform.hpp>

//#include <libPictStruct/objectdetect.h>

#include "partapp.h"
#include "partapp_aux.hpp"

using namespace std;

using boost::multi_array_types::index_range;

/** 
    MA: as far as I can tell this function produces completely wrong result for parts where part position is not the same as center of the bounding box
*/
void bbox_from_pos(const ExpParam &exp_param, const PartWindowParam::PartParam &part_param, 
                   int scaleidx, int rotidx, int ix, int iy, 
                   PartBBox &bbox)
{
  double rot = rot_from_index(exp_param, rotidx) / 180.0 * M_PI;
  double scale = scale_from_index(exp_param, scaleidx);

  bbox_from_pos(part_param, scale, rot, ix, iy, bbox);
 
}

void bbox_from_pos(const PartWindowParam::PartParam &part_param, 
                   double scale, double rot, int ix, int iy, 
                   PartBBox &bbox)
{
  bbox.part_pos(0) = ix;
  bbox.part_pos(1) = iy;
  
  bbox.part_x_axis(0) = cos(rot);
  bbox.part_x_axis(1) = sin(rot);

  bbox.part_y_axis(0) = -bbox.part_x_axis(1);
  bbox.part_y_axis(1) = bbox.part_x_axis(0);

  double rect_width = scale*part_param.window_size_x();
  double rect_height = scale*part_param.window_size_y();
  //cout << "bbox_from_pos, scale: " << scale << ", rect_width: " << rect_width << ", rect_height: " << rect_height << endl;

  bbox.min_proj_x = -scale*part_param.pos_offset_x();
  bbox.min_proj_y = -scale*part_param.pos_offset_y();

  bbox.max_proj_x = bbox.min_proj_x + rect_width;
  bbox.max_proj_y = bbox.min_proj_y + rect_height;
}

void appendAnnoList(AnnotationList &annolist1, const AnnotationList &annolist2)
{
  for (int aidx = 0; aidx < (int)annolist2.size(); ++aidx) 
    annolist1.addAnnotation(annolist2[aidx]);
}

void convertFullPath(QString qsAnnoListFile, AnnotationList &annolist) {
  assert(filesys::check_file(qsAnnoListFile) && "annotation file not found");

  QString qsAnnoListPath;
  QString qsAnnoListName;
  filesys::split_filename(qsAnnoListFile, qsAnnoListPath, qsAnnoListName);

  for (int imgidx = 0; imgidx < (int)annolist.size(); ++imgidx) {
    if (!filesys::check_file(annolist[imgidx].imageName().c_str())) {

      QString qsFilename = qsAnnoListPath + "/" + annolist[imgidx].imageName().c_str();

      if (!filesys::check_file(qsFilename)) {
        cout << "file not found: " << qsFilename.toStdString() << endl;
        assert(false);
      }

      annolist[imgidx].setImageName(qsFilename.toStdString());
    }

  }
}

QString complete_relative_path(QString qsInputFile, QString qsReferenceFile)
{
  qsInputFile = qsInputFile.trimmed();

  QDir dir(qsInputFile);

  QString qsRes;

  if (dir.isRelative()) {
    QString qsPath1, qsName1;
    QString qsPath2, qsName2;

    filesys::split_filename(qsInputFile, qsPath1, qsName1);
    filesys::split_filename(qsReferenceFile, qsPath2, qsName2);

    /** corrections from Marcin */
    if (qsPath1 == ".")
      qsRes = qsPath2 + "/" + qsName1;
    else 
      qsRes = qsPath2 + qsPath1.mid(1) + "/" + qsName1;

  }
  else {
    qsRes = qsInputFile;
  }

  return qsRes;
}

void PartApp::init(QString qsExpParam)
{
  m_qsExpParam = qsExpParam;
  parse_message_from_text_file(qsExpParam, m_exp_param);

  assert(!m_exp_param.is_multicomponent());

  init_setpath(qsExpParam, false);
  init_loaddata();
  init_partdims();

}

void PartApp::init_multicomponent(QString qsExpParam, std::vector<PartApp> &partapp_components)
{
  ExpParam exp_param;
  parse_message_from_text_file(qsExpParam, exp_param);
  assert(exp_param.is_multicomponent());

  partapp_components.clear();

  /** loop over components */
  vector<int> train_compidx;
  AnnotationList train_annolist_all;
  vector<int> validation_all_idx;
  vector<int> neg_all_idx;
  
  for (uint compidx = 0; compidx < exp_param.num_components(); ++compidx) {
    
    PartApp part_app(compidx);
    
    part_app.m_qsExpParam = qsExpParam;
    part_app.m_exp_param = exp_param;

    part_app.m_exp_param.set_is_multicomponent(false);

    if (!part_app.m_exp_param.has_log_subdir()) {
      QString qsExpParamPath;
      QString qsExpParamName;
      QString qsExpParamExt;
      filesys::split_filename_ext(qsExpParam, qsExpParamPath, qsExpParamName, qsExpParamExt);
      part_app.m_exp_param.set_log_subdir(qsExpParamName.toStdString());
    }

    part_app.m_exp_param.set_log_subdir(part_app.m_exp_param.log_subdir() + "/log_dir_comp" + 
					(padZeros(QString::number(compidx), 3)).toStdString()); 

    /** take care of external class dir (should be provided for each component) */
    if (exp_param.comp_class_dir_size() > 0) {
      assert(exp_param.comp_class_dir_size() == part_app.m_exp_param.num_components());
      
      part_app.m_exp_param.set_class_dir(exp_param.comp_class_dir(compidx));
      part_app.m_exp_param.clear_comp_class_dir();
    }

    /** standard init, but with different log_dir */
    part_app.init_setpath(qsExpParam, true);

    /** loading datasets takes a while, do it only for the first component */
    if (compidx == 0) {
      part_app.init_loaddata();

      train_annolist_all = part_app.m_train_annolist;
      /** figure out the number of components, so far component ID is encoded via silhouette ID */
      for (uint imgidx = 0; imgidx < train_annolist_all.size(); ++imgidx) {
	for (uint ridx = 0; ridx < train_annolist_all[imgidx].size(); ++ridx) { 

	  /** ignore annotations without annopoints */
	  if (train_annolist_all[imgidx][ridx].m_vAnnoPoints.size() == 0)
	    continue;

	  if (train_annolist_all[imgidx][ridx].silhouetteID() < 0) {
	    cout << "missing id: " << imgidx << ", ridx: " << ridx << endl;
	    assert(false);
	  }

	  train_compidx.push_back(train_annolist_all[imgidx][ridx].silhouetteID());
	}
      }
  
      sort(train_compidx.begin(), train_compidx.end());
      vector<int>::iterator it = unique(train_compidx.begin(), train_compidx.end());
      train_compidx.resize(it - train_compidx.begin());
  
      cout << "found training data for components: " << endl;
  
      for (uint idx = 0; idx < train_compidx.size(); ++idx)
	cout << train_compidx[idx] << endl;

      /** only need training data if no external class_dir is provided */
      // Leonid: why?
      //if (exp_param.comp_class_dir_size() == 0) {
      if (true) {
	assert(train_compidx.size() == exp_param.num_components());
      }
      else {
	/** 
	    since the order of classifiers in comp_class_dir is arbitrary 
	    there is no way to match "silhouette id" to component id

	    -> ignore training data
	*/

	train_compidx.clear();
	train_compidx.resize(exp_param.num_components(), -1);
      }
  
      cout << "number of components: " << train_compidx.size() << endl;    
    }
    else {
      assert(partapp_components.size() > 0);

      part_app.m_train_annolist = train_annolist_all;
      part_app.m_train_reshaped_annolist = partapp_components[0].m_train_reshaped_annolist;
      part_app.m_validation_annolist = partapp_components[0].m_validation_annolist;
      part_app.m_test_annolist = partapp_components[0].m_test_annolist;
      part_app.m_neg_annolist = partapp_components[0].m_neg_annolist;
    }


    /**
       remove training data from other components 
    */
    assert(compidx < train_compidx.size());

    AnnotationList train_annolist = train_annolist_all;
    part_app.m_train_annolist.clear();

    for (uint imgidx = 0; imgidx < train_annolist.size(); ++imgidx) {

      Annotation a = train_annolist[imgidx];
      a.clear();

      for (uint ridx = 0; ridx < train_annolist[imgidx].size(); ++ridx) {
	if (train_annolist[imgidx][ridx].silhouetteID() == train_compidx[compidx])
	  a.addAnnoRect(train_annolist[imgidx][ridx]);
      }
      
      if (a.size() > 0){
	part_app.m_train_annolist.addAnnotation(a);

      }
      
    }
    cout << "component " << compidx << ", num training images: " << part_app.m_train_annolist.size() << endl;
    
    part_app.init_partdims();

    partapp_components.push_back(part_app);
    
  } // components 
}

void PartApp::init_setpath(QString qsExpParam, bool is_multicomp)
{
  assert(m_exp_param.part_conf().length() > 0);
  assert(m_exp_param.abc_param().length() > 0);
  assert(m_exp_param.log_dir().length() > 0);
  assert(m_exp_param.train_dataset_size() > 0);

  /* load part configuration and detector parameters */
  QString qsPartConf = complete_relative_path(m_exp_param.part_conf().c_str(), qsExpParam);
  QString qsAbcParam = complete_relative_path(m_exp_param.abc_param().c_str(), qsExpParam);

  cout << "part configuration: " << qsPartConf.toStdString() << endl;
  cout << "classifier parameters: " << qsAbcParam.toStdString() << endl;

  parse_message_from_text_file(qsPartConf, m_part_conf);
  parse_message_from_text_file(qsAbcParam, m_abc_param);
  
  assert(m_part_conf.part_size() > 0 && "missing part definitions");

  if (!m_exp_param.has_log_subdir()) {
    QString qsExpParamPath;
    QString qsExpParamName;
    QString qsExpParamExt;
    filesys::split_filename_ext(qsExpParam, qsExpParamPath, qsExpParamName, qsExpParamExt);
    m_exp_param.set_log_subdir(qsExpParamName.toStdString());    
  }

  if (!m_exp_param.has_class_dir()) {
    m_bExternalClassDir = false;

    m_exp_param.set_class_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/class");
    cout << "set class_dir to: " << m_exp_param.class_dir() << endl;
  }
  else {
    m_bExternalClassDir = true;

    cout << "class_dir: " << m_exp_param.class_dir() << endl;
  }

  /** initialize scoregrid_dir to default value */
  if (!m_exp_param.has_scoregrid_dir()) {
    m_bExternalScoregridDir = false;
    m_exp_param.set_scoregrid_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/test_scoregrid");
    cout << "set scoregrid_dir to: " << m_exp_param.scoregrid_dir() << endl;
  }
  else {
    cout << "scoregrid_dir: " << m_exp_param.scoregrid_dir() << endl;
    m_bExternalScoregridDir = true;
  }

  /** initialize scoregrid_pos_dir to default value */
  if (!m_exp_param.has_scoregrid_train_dir()) {
    m_exp_param.set_scoregrid_train_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/train_scoregrid");
    cout << "set scoregrid_train_dir to: " << m_exp_param.scoregrid_train_dir() << endl;
  }
  else {
    cout << "scoregrid_train_dir: " << m_exp_param.scoregrid_train_dir() << endl;
  }

  if (!m_exp_param.has_part_marginals_dir()) {
    m_exp_param.set_part_marginals_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/part_marginals");
    cout << "set part_marginals_dir to: " << m_exp_param.part_marginals_dir() << endl;
  }
  else {
    cout << "part_marginals_dir: " << m_exp_param.part_marginals_dir() << endl;
  }
  
  if (!m_exp_param.has_test_dpm_unary_dir()) {
    m_exp_param.set_test_dpm_unary_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/test_dpm_unary");
    cout << "set test_dpm_unary_dir to: " << m_exp_param.test_dpm_unary_dir() << endl;
  }
  else {
    cout << "test_dpm_unary_dir: " << m_exp_param.test_dpm_unary_dir() << endl;
  }
  
  if (!m_exp_param.has_dpm_model_dir()) {
    m_exp_param.set_dpm_model_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/dpm_model");
    cout << "set dpm_model_dir to: " << m_exp_param.dpm_model_dir() << endl;
  }
  else {
    cout << "dpm_model_dir: " << m_exp_param.dpm_model_dir() << endl;
  }
  
  if (!m_exp_param.has_test_dpm_torso_dir()) {
    m_exp_param.set_test_dpm_torso_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/test_dpm_unary/torso");
    cout << "set test_dpm_torso_dir to: " << m_exp_param.test_dpm_torso_dir() << endl;
  }
  else {
    cout << "test_dpm_torso_dir: " << m_exp_param.test_dpm_torso_dir() << endl;
  }

  QString qsExpParamName = "";
  if (!is_multicomp)
    qsExpParamName = m_exp_param.log_subdir().c_str();
  else{
    QString qsExpParamPath;
    QString qsExpParamExt;
    filesys::split_filename_ext(m_exp_param.log_subdir().c_str(), qsExpParamPath, qsExpParamName, qsExpParamExt);
    filesys::split_filename_ext(qsExpParamPath.toStdString().c_str(), qsExpParamPath, qsExpParamName, qsExpParamExt);
  }
  
  if (!m_exp_param.has_mix_dir()) {
    m_exp_param.set_mix_dir(m_exp_param.log_dir() + "/" + qsExpParamName.toStdString().c_str() + "/part_marginals");
    cout << "set mix_dir to: " << m_exp_param.mix_dir() << endl;
  }
  else {
    cout << "mix_dir: " << m_exp_param.mix_dir() << endl;
  }
  
  if (!m_exp_param.has_pred_data_dir()) {
    m_exp_param.set_pred_data_dir(m_exp_param.log_dir() + "/" + qsExpParamName.toStdString().c_str() + "/pred_data");
    cout << "set pred_data_dir to: " << m_exp_param.pred_data_dir() << endl;
  }
  else {
    cout << "pred_data_dir: " << m_exp_param.pred_data_dir() << endl;
  }
  
  if (!m_exp_param.has_pred_data_test_dir()) {
    m_exp_param.set_pred_data_test_dir(m_exp_param.log_dir() + "/" + qsExpParamName.toStdString().c_str() + "/pred_data_test");
    cout << "set pred_data_test_dir to: " << m_exp_param.pred_data_test_dir() << endl;
  }
  else {
    cout << "pred_data_test_dir: " << m_exp_param.pred_data_test_dir() << endl;
  }
  
  if (!m_exp_param.has_poselet_resp_val_dir()) {
    m_exp_param.set_poselet_resp_val_dir(m_exp_param.log_dir() + "/" + qsExpParamName.toStdString().c_str() + "/resp_train");
    cout << "set poselet_resp_val_dir to: " << m_exp_param.poselet_resp_val_dir() << endl;
  }
  else {
    cout << "poselet_resp_val_dir: " << m_exp_param.poselet_resp_val_dir() << endl;
  }
  
  if (!m_exp_param.has_poselet_resp_test_dir()) {
    m_exp_param.set_poselet_resp_test_dir(m_exp_param.log_dir() + "/" + qsExpParamName.toStdString().c_str() + "/resp_test");
    cout << "set poselet_resp_test_dir to: " << m_exp_param.poselet_resp_test_dir() << endl;
  }
  else {
    cout << "poselet_resp_test_dir: " << m_exp_param.poselet_resp_test_dir() << endl;
  }
  
  if (!m_exp_param.has_spatial_dir()) {
    m_exp_param.set_spatial_dir(m_exp_param.log_dir() + "/" + qsExpParamName.toStdString().c_str() + "/spatial");
    cout << "set spatial_dir to: " << m_exp_param.spatial_dir() << endl;
  }
  else {
    cout << "spatial_dir: " << m_exp_param.spatial_dir() << endl;
  }

  /** initialize partprob_dir to default value */
//   if (!m_exp_param.has_partprob_dir()) {
//     m_exp_param.set_partprob_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/test_partprob");
//     cout << "set partprob_dir to: " << m_exp_param.partprob_dir() << endl;
//   }
//   else {
//     cout << "partprob_dir: " << m_exp_param.partprob_dir() << endl;
//   }

  if (!m_exp_param.has_dai_samples_dir()) {
    m_bExternalSamplesDir = false;

    std::string sSamplesDir;

    if (m_exp_param.dai_samples_type() == "part_post")
      sSamplesDir = "part_marginals_samples";
    else if (m_exp_param.dai_samples_type() == "part_det")
      sSamplesDir = "test_scoregrid_samples";
    else 
      assert(false && "unknown part samples type");

    m_exp_param.set_dai_samples_dir(m_exp_param.log_dir() + "/" + 
				    m_exp_param.log_subdir() + "/" + sSamplesDir);

    cout << "set dai_samples_dir to: " << m_exp_param.dai_samples_dir() << endl;
  }
  else {
    m_bExternalSamplesDir = true;
    cout << "dai_samples_dir: " << m_exp_param.dai_samples_dir() << endl;
  }

  if (!filesys::check_dir(m_exp_param.log_dir().c_str())) {
    assert(filesys::create_dir(m_exp_param.log_dir().c_str()));
  }

  if (!filesys::check_dir(m_exp_param.class_dir().c_str())) {
    filesys::create_dir(m_exp_param.class_dir().c_str());
  }
  
  if (!filesys::check_dir(m_exp_param.spatial_dir().c_str())) {
    filesys::create_dir(m_exp_param.spatial_dir().c_str());
  }
}

void PartApp::init_loaddata()
{
  /**
     load training, validation and test data 
  */

  if (m_exp_param.train_dataset_size() > 0) {

    /* concatenate training annotation files */
    for (int listidx = 0; listidx < m_exp_param.train_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.train_dataset(listidx).c_str()));

      cout << "\nloading training data from " << m_exp_param.train_dataset(listidx) << endl;

      AnnotationList annolist;
      annolist.load(m_exp_param.train_dataset(listidx));

      appendAnnoList(m_train_annolist, annolist);
    }

    //assert(m_train_annolist.size() > 0);
  }
  cout << "loaded " << m_train_annolist.size() << " real training images" << endl;
  
  if (m_exp_param.train_dataset_reshaped_size() > 0) {
    
    /* concatenate training annotation files */
    for (int listidx = 0; listidx < m_exp_param.train_dataset_reshaped_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.train_dataset_reshaped(listidx).c_str()));
      
      cout << "\nloading training data from " << m_exp_param.train_dataset_reshaped(listidx) << endl;
      
      AnnotationList annolist;
      annolist.load(m_exp_param.train_dataset_reshaped(listidx), 1);
      
      appendAnnoList(m_train_reshaped_annolist, annolist);
    }
    
    //assert(m_train_annolist.size() > 0);
  }
  cout << "loaded " << m_train_reshaped_annolist.size() << " reshaped training images" << endl;
  
  if (m_exp_param.validation_dataset_size() > 0) {

    for (int listidx = 0; listidx < m_exp_param.validation_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.validation_dataset(listidx).c_str()));

      cout << "\nloading validataion data from " << m_exp_param.validation_dataset(listidx) << endl;

      AnnotationList annolist;
      annolist.load(m_exp_param.validation_dataset(listidx));

      appendAnnoList(m_validation_annolist, annolist);
    }

    //assert(m_validation_annolist.size() > 0);
  }

  if (m_exp_param.test_dataset_size() > 0) {

    /* concatenate test annotation files */
    for (int listidx = 0; listidx < m_exp_param.test_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.test_dataset(listidx).c_str()));      

      cout << "\nloading test data from " << m_exp_param.test_dataset(listidx) << endl;      

      AnnotationList annolist;
      annolist.load(m_exp_param.test_dataset(listidx));
      
      /* expand path, in case it is given relative to home directory */
      convertFullPath(m_exp_param.test_dataset(listidx).c_str(), annolist);
      
      appendAnnoList(m_test_annolist, annolist);
    }

    assert(m_test_annolist.size() > 0);
  }

  if (m_exp_param.neg_dataset_size() > 0) {

    /* concatenate neg annotation files */
    for (int listidx = 0; listidx < m_exp_param.neg_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.neg_dataset(listidx).c_str()));      

      cout << "\nloading negative data from " << m_exp_param.neg_dataset(listidx) << endl;      

      AnnotationList annolist;
      annolist.load(m_exp_param.neg_dataset(listidx));
      
      /* expand path, in case it is given relative to home directory */
      convertFullPath(m_exp_param.neg_dataset(listidx).c_str(), annolist);
      
      appendAnnoList(m_neg_annolist, annolist);
    }

    assert(m_neg_annolist.size() > 0);
  }
  
  if (m_exp_param.bootstrap_dataset_size() > 0) {

    /* concatenate neg annotation files */
    for (int listidx = 0; listidx < m_exp_param.bootstrap_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.bootstrap_dataset(listidx).c_str()));      

      cout << "\nloading bootstrap data from " << m_exp_param.bootstrap_dataset(listidx) << endl;      

      AnnotationList annolist;
      annolist.load(m_exp_param.bootstrap_dataset(listidx));
      
      /* expand path, in case it is given relative to home directory */
      convertFullPath(m_exp_param.bootstrap_dataset(listidx).c_str(), annolist);
      
      appendAnnoList(m_bootstrap_annolist, annolist);
    }

    assert(m_bootstrap_annolist.size() > 0);
  }
}

void PartApp::init_partdims()
{
  QString qsWindowParamFile = (m_exp_param.class_dir() + "/window_param.txt").c_str();
  cout << "qsWindowParamFile: " << qsWindowParamFile.toStdString() << endl;

  /* load part window dimensions */
  
  if (filesys::check_file(qsWindowParamFile)) {
    cout << "\nloading window parameters from " << qsWindowParamFile.toStdString() << endl;
    parse_message_from_text_file(qsWindowParamFile, m_window_param);
  }
  else {
    cout << "WARNING: window parameters file not found" << endl;
  }
  
  /* compute part window dimensions if needed */
  //if (m_window_param.part_size() != m_part_conf.part_size()) {
  if (m_window_param.part_size() < m_part_conf.part_size()) {
    cout << "\nrecomputing part window size ..." << endl;
  
    //part_detect::compute_part_window_param(m_train_annolist, m_part_conf, m_window_param);
    part_detect::compute_part_type_window_param(m_train_annolist, m_part_conf, 
						m_window_param,	m_exp_param, QString(m_exp_param.class_dir().c_str()));
    //assert(m_window_param.part_size() == m_part_conf.part_size());
    cout << m_window_param.part_size() << endl;
    cout << m_part_conf.part_size() << endl;
    assert(m_window_param.part_size() >= m_part_conf.part_size());
    
    cout << "saving " << qsWindowParamFile.toStdString() << endl;
    print_message_to_text_file(qsWindowParamFile, m_window_param);
  }

  /**
     find the root part
   */
  for (int pidx = 0; pidx < m_part_conf.part_size(); ++pidx) {
    if (m_part_conf.part(pidx).is_root()) {
      assert(m_rootpart_idx == -1);
      m_rootpart_idx = pidx;
      break;
    }
  }
  assert(m_rootpart_idx >=0 && "missing root part");

  /**  
    save part parameters to enable visualization in Matlab
  */

  //if (!m_bExternalClassDir) {
  {

    /** 
	MA: all of this would not be necessary if there existed an implementation of Protocol Buffers for Matlab
     */

    QString qsParamsMatDir = (m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/params_mat").c_str();
    
    if (!filesys::check_dir(qsParamsMatDir))
      filesys::create_dir(qsParamsMatDir);

    cout << "saving Matlab visualization parameters in:  " << qsParamsMatDir.toStdString() << endl;

    /* here we skip parts which are not detected */
    
    bool bSaveMatlabParams = true;
    
    if (bSaveMatlabParams) {
      int nParts = 0;
      for (int pidx = 0; pidx < m_part_conf.part_size(); ++pidx)
	if (m_part_conf.part(pidx).is_detect())
	  ++nParts;

      matlab_io::mat_save_double(qsParamsMatDir + "/num_parts.mat", "num_parts", nParts);

      boost_math::double_matrix part_dims(nParts, 2);
      for (int pidx = 0; pidx < m_part_conf.part_size(); ++pidx)
	if (m_part_conf.part(pidx).is_detect()) {
	  assert(m_window_param.part_size() > pidx);
	  part_dims(pidx, 0) = m_window_param.part(pidx).window_size_x();
	  part_dims(pidx, 1) = m_window_param.part(pidx).window_size_y();
	}

      matlab_io::mat_save_double_matrix(qsParamsMatDir + "/part_dims.mat", "part_dims", part_dims);

      MATFile *f = matlab_io::mat_open(qsParamsMatDir + "/rotation_params.mat", "wz");
      assert(f != 0);
      matlab_io::mat_save_double(f, "min_part_rotation", m_exp_param.min_part_rotation());
      matlab_io::mat_save_double(f, "max_part_rotation", m_exp_param.max_part_rotation());
      matlab_io::mat_save_double(f, "num_rotation_steps", m_exp_param.num_rotation_steps());
      matlab_io::mat_close(f);

      f = matlab_io::mat_open(qsParamsMatDir + "/scale_params.mat", "wz");
      assert(f != 0);
      matlab_io::mat_save_double(f, "min_object_scale", m_exp_param.min_object_scale());
      matlab_io::mat_save_double(f, "max_object_scale", m_exp_param.max_object_scale());
      matlab_io::mat_save_double(f, "num_scale_steps", m_exp_param.num_scale_steps());
      matlab_io::mat_close(f);

      /**
	 not sure what happens if root part is a "dummy" part without detector, 

	 above we saved parameters only for detectable parts
      */

      matlab_io::mat_save_double(qsParamsMatDir + "/rootpart_idx.mat", "rootpart_idx", m_rootpart_idx);
    }
  }  
}

QString PartApp::getClassFilename(int pidx, int bootstrap_type, int tidx) const
{
  QString qsClassFilename;

  QString qsTypeSuff = "";
  if (tidx > -1)
    qsTypeSuff = QString("_type") + QString::number(tidx);
  
  if (bootstrap_type == 0) {
    qsClassFilename = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + qsTypeSuff + ".class";
  }
  else if (bootstrap_type == 1) {
    qsClassFilename = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + qsTypeSuff + "-bootstrap.class";
  }
  else {
    QString qsClassFilenameBootstrap = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + qsTypeSuff + "-bootstrap.class";
    QString qsClassFilenameNormal = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + qsTypeSuff + ".class";

    //if (m_abc_param.num_train_bootstrap() > 0) {

    //if (m_abc_param.bootstrap_fraction() > 0) {

    /** 
        "num_train_bootstrap" is deprecated, 
        at some point we should look at "bootstrap_fraction" only

        for now check both "bootstrap_fraction" and "num_train_bootstrap"
        in order to be able to load old bootstrapped classfiers
    */

    //if (m_abc_param.bootstrap_fraction() > 0 || m_abc_param.num_train_bootstrap() > 0) {

    if (m_abc_param.bootstrap_fraction() > 0) {
      if (filesys::check_file(qsClassFilenameBootstrap)) {
        qsClassFilename = qsClassFilenameBootstrap;
      }
      else {
        cout << "warning: bootstrap classifier not found!!!" << endl;
        qsClassFilename = qsClassFilenameNormal;
      }
    }
    else {
      qsClassFilename = qsClassFilenameNormal;
    }
  }
  
  return qsClassFilename;
}

void PartApp::loadClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type, int tidx) const
{
  assert(m_exp_param.class_dir().length() > 0);
  assert(filesys::check_dir(m_exp_param.class_dir().c_str()));

  QString qsClassFilename = getClassFilename(pidx, bootstrap_type, tidx);

  cout << "loading classifier from " << qsClassFilename.toStdString() << endl;
  assert(filesys::check_file(qsClassFilename));

  abc.loadClassifier(qsClassFilename.toStdString());
}

void PartApp::saveClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type, int tidx) const
{
  assert(!m_bExternalClassDir && "can not update external parameters");
  
  assert(m_exp_param.class_dir().length() > 0);
  assert(filesys::check_dir(m_exp_param.class_dir().c_str()));

  //QString qsClassFilename = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + ".class";
  QString qsClassFilename = getClassFilename(pidx, bootstrap_type, tidx);

  cout << "saving classifier to " << qsClassFilename.toStdString() << endl;
  abc.saveClassifier(qsClassFilename.toStdString());
}


void PartApp::getScoreGridFileName(int imgidx, int pidx, bool flip,
                                   QString &qsFilename, QString &qsVarName, QString qsScoreGridDir) const
{
  qsFilename = qsScoreGridDir + "/imgidx" + QString::number(imgidx) + 
    "-pidx" + QString::number(pidx) + "-o" + QString::number((int)flip) + "-scoregrid.mat";
  //qsVarName = "cell_img_scoregrid";
  qsVarName = "cell_scoregrid";
}

/**
   load marginals for given part and scale

   dimensions of grid3: rotation, y, x

 */
FloatGrid3 PartApp::loadPartMarginal(int imgidx, int pidx, int scaleidx, bool flip) const 
{
  QString qsPartMarginalsDir = (m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/part_marginals").c_str();
  assert(filesys::check_dir(qsPartMarginalsDir));
  
  QString qsFilename = qsPartMarginalsDir + "/log_part_posterior_final" + 
    "_imgidx" + QString::number(imgidx) +
    "_scaleidx" + QString::number(scaleidx) +
    "_o" + QString::number((int)flip) + 
    "_pidx" + QString::number(pidx) + ".mat";

  cout << "loading " << qsFilename.toStdString() << endl;

  MATFile *f = matlab_io::mat_open(qsFilename, "r");
  assert(f != 0);

  /* make sure we use copy constructor here */ 
  FloatGrid3 _grid3 = matlab_io::mat_load_multi_array<FloatGrid3>(f, "log_prob_grid");
  matlab_io::mat_close(f);

  return _grid3;
}

void PartApp::loadScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
                            bool flip, bool bInterpolate, QString qsScoreGridDir, QString qsImgName) const
{
  cout << "PartApp::loadScoreGrid" << endl;

  QString qsFilename;
  QString qsVarName;
  getScoreGridFileName(imgidx, pidx, flip, qsFilename, qsVarName, qsScoreGridDir);
  
  cout << "loading scoregrid from " << qsFilename.toStdString() << endl;
  MATFile *f = matlab_io::mat_open(qsFilename, "r");
  assert(f != 0);
  matlab_io::mat_load_multi_array_vec2(f, qsVarName, score_grid);
  if (f != 0)
    matlab_io::mat_close(f);


  /** load transformation matrices */

  f = matlab_io::mat_open(qsFilename, "r"); assert(f != 0);
  FloatGrid4 transform_Ti2 = matlab_io::mat_load_multi_array<FloatGrid4>(f, "transform_Ti2");
  matlab_io::mat_close(f);

  f = matlab_io::mat_open(qsFilename, "r"); assert(f != 0);
  FloatGrid4 transform_T2g = matlab_io::mat_load_multi_array<FloatGrid4>(f, "transform_T2g");
  matlab_io::mat_close(f);

  /** find the image size, yes we need to load an image just for that :) */
  kma::ImageContent *kmaimg = kma::load_convert_gray_image(qsImgName.toStdString().c_str());

  assert(kmaimg != 0);
  int img_width = kmaimg->x();
  int img_height = kmaimg->y();
  delete kmaimg;

  /** map from the grid to image coordinates */

  int nScales = m_exp_param.num_scale_steps();
  
  int nRotations = m_exp_param.num_rotation_steps();

  assert((int)score_grid.size() == nScales);
  assert((int)score_grid[0].size() == nRotations);
    
  std::vector<std::vector<FloatGrid2> > score_grid_image(nScales, std::vector<FloatGrid2>());

  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      double_matrix T2g;
      double_matrix Ti2;

      multi_array_op::array_to_matrix(transform_T2g[boost::indices[scaleidx][rotidx][index_range()][index_range()]],
                                      T2g);
        
      multi_array_op::array_to_matrix(transform_Ti2[boost::indices[scaleidx][rotidx][index_range()][index_range()]],
                                      Ti2);

      double_matrix Tig = prod(Ti2, T2g);
        

      FloatGrid2 grid(boost::extents[img_height][img_width]);
      if (bInterpolate)
        multi_array_op::transform_grid_fixed_size(score_grid[scaleidx][rotidx], grid, Tig, 
                                                  part_detect::NO_CLASS_VALUE, TM_BILINEAR);
      else
        multi_array_op::transform_grid_fixed_size(score_grid[scaleidx][rotidx], grid, Tig, 
                                                  part_detect::NO_CLASS_VALUE, TM_DIRECT);

      score_grid_image[scaleidx].push_back(grid);
    }

  score_grid.clear();
  score_grid = score_grid_image;
}

void PartApp::loadScoreGridRotRange(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
				    bool flip, bool bInterpolate, QString qsScoreGridDir, QString qsImgName, std::vector<int> &rotidx_transform) const
{
  cout << "PartApp::loadScoreGridRotRange" << endl;

  QString qsFilename;
  QString qsVarName;
  getScoreGridFileName(imgidx, pidx, flip, qsFilename, qsVarName, qsScoreGridDir);
  
  cout << "loading scoregrid from " << qsFilename.toStdString() << endl;
  MATFile *f = matlab_io::mat_open(qsFilename, "r");
  assert(f != 0);
  matlab_io::mat_load_multi_array_vec2(f, qsVarName, score_grid);
  if (f != 0)
    matlab_io::mat_close(f);
  
  //cout << "********************************" << endl;
  //cout << score_grid[0].size() << endl;
  //cout << "********************************" << endl;
  
  /** load transformation matrices */
  f = matlab_io::mat_open(qsFilename, "r"); assert(f != 0);
  FloatGrid4 transform_Ti2 = matlab_io::mat_load_multi_array<FloatGrid4>(f, "transform_Ti2");
  matlab_io::mat_close(f);

  f = matlab_io::mat_open(qsFilename, "r"); assert(f != 0);
  FloatGrid4 transform_T2g = matlab_io::mat_load_multi_array<FloatGrid4>(f, "transform_T2g");
  matlab_io::mat_close(f);

  /** find the image size, yes we need to load an image just for that :) */
  kma::ImageContent *kmaimg = kma::load_convert_gray_image(qsImgName.toStdString().c_str());

  assert(kmaimg != 0);
  int img_width = kmaimg->x();
  int img_height = kmaimg->y();
  delete kmaimg;

  /** map from the grid to image coordinates */

  int nScales = m_exp_param.num_scale_steps();
  
  //int nRotations = 1;
  //int nRotations = m_exp_param.num_rotation_steps();

  assert((int)score_grid.size() == nScales);
  //assert((int)score_grid[0].size() == nRotations);
    
  std::vector<std::vector<FloatGrid2> > score_grid_image(nScales, std::vector<FloatGrid2>());

  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {
    for (int rotidx = 0; rotidx < rotidx_transform.size();++rotidx) {
      double_matrix T2g;
      double_matrix Ti2;
      int ridx = rotidx_transform[rotidx];
      //cout << "rotidx_transform: " << ridx << endl;

      multi_array_op::array_to_matrix(transform_T2g[boost::indices[scaleidx][ridx][index_range()][index_range()]], T2g);
      multi_array_op::array_to_matrix(transform_Ti2[boost::indices[scaleidx][ridx][index_range()][index_range()]], Ti2);

      double_matrix Tig = prod(Ti2, T2g);
      FloatGrid2 grid(boost::extents[img_height][img_width]);
      if (bInterpolate)
	multi_array_op::transform_grid_fixed_size(score_grid[scaleidx][rotidx], grid, Tig, 
						  part_detect::NO_CLASS_VALUE, TM_BILINEAR);
      else
	multi_array_op::transform_grid_fixed_size(score_grid[scaleidx][rotidx], grid, Tig, 
						  part_detect::NO_CLASS_VALUE, TM_DIRECT);
      score_grid_image[scaleidx].push_back(grid);
    }
  }
  
  score_grid.clear();
  score_grid = score_grid_image;
}

void PartApp::saveScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
                            bool flip) const
{
  cout << "PartApp::savedScoreGrid" << endl;
  QString qsFilename;
  QString qsVarName;
  getScoreGridFileName(imgidx, pidx, flip, qsFilename, qsVarName, "test_scoregrid");

  cout << "saving scoregrid to " << qsFilename.toStdString() << endl;
  MATFile *f = matlab_io::mat_open(qsFilename, "wz");
  assert(f != 0);
  matlab_io::mat_save_multi_array_vec2(f, qsVarName, score_grid);
  if (f != 0)
    matlab_io::mat_close(f);

}

