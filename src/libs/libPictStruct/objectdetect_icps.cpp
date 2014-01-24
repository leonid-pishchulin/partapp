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

#include <libPartApp/partapp_aux.hpp>
#include <libFilesystemAux/filesystem_aux.h>
#include <libMultiArray/multi_array_op.hpp>
#include <libPartDetect/partdetect.h>
#include <libDiscPS/disc_ps.h>
#include <libDiscPS/factors.h>
#include <libPrediction/matlab_runtime.h>
#include <QProcess>

#include "objectdetect.h"

using namespace std;

namespace object_detect {

  void getTorsoPosPriorParams(const PartApp &part_app, int img_height, int img_width, boost_math::double_matrix &pos_prior_params){
    
    cout << "\ngetTorsoPosPriorParams()" << endl;
    
    QString qsFilename = (part_app.m_exp_param.class_dir() + "/torso_pos_prior.mat").c_str();    
    matlab_io::mat_load_double_matrix(qsFilename, "params", pos_prior_params);
    
    cout << pos_prior_params(0,0) << endl;
    cout << pos_prior_params(0,1) << endl;
    cout << pos_prior_params(0,2) << endl;
    cout << pos_prior_params(0,3) << endl;
    
    cout << endl;
  }

  void computeTorsoPosPriorParams(const PartApp &part_app){
    
    cout << "\ncomputeTorsoPosPriorParams()" << endl;
    
    if (part_app.m_bExternalClassDir) {
      cout << "WARNING! No pos prior is computed: can not update external parameters" << endl;
      return;
    }
      
    float offset_x = 0.0, offset_y = 0.0;
    uint id1 = part_app.m_part_conf.part(part_app.m_rootpart_idx).part_pos(0);
    uint id2 = part_app.m_part_conf.part(part_app.m_rootpart_idx).part_pos(1);
    
    boost_math::double_matrix torso_offset(part_app.m_train_annolist.size(),2);

    int nEx = -1;
    
    for (int imgidx = 0; imgidx < part_app.m_train_annolist.size();++imgidx){
      if (annorect_has_part(part_app.m_train_annolist[imgidx][0], part_app.m_part_conf.part(part_app.m_rootpart_idx))){
	kma::ImageContent *kmaimg = kma::load_convert_gray_image(part_app.m_train_annolist[imgidx].imageName().c_str());
	int img_height = kmaimg->y();
	int img_width = kmaimg->x();
	
	float x0 = (1.0*img_width/2);
	float y0 = (1.0*img_height/2);
	
	const AnnoPoint *p1 = part_app.m_train_annolist[imgidx][0].get_annopoint_by_id(id1);
	const AnnoPoint *p2 = part_app.m_train_annolist[imgidx][0].get_annopoint_by_id(id2);
	
	assert(p1 != NULL);
	assert(p2 != NULL);
	
	float x_torso = (p1->x + p2->x)/2;
	float y_torso = (p1->y + p2->y)/2;
	
	nEx++;
	torso_offset(nEx,0) = x0 - x_torso;
	torso_offset(nEx,1) = y0 - y_torso;      
	offset_x += torso_offset(nEx,0);
	offset_y += torso_offset(nEx,1);
	
	delete kmaimg;
      }
    }

    float mu_x = offset_x/nEx;
    float mu_y = offset_y/nEx;
    float var_x = 0;
    float var_y = 0;
    
    for (int imgidx = 0; imgidx < nEx;++imgidx){
      
      var_x += square(torso_offset(imgidx,0) - mu_x);
      var_y += square(torso_offset(imgidx,1) - mu_y);
    }
    
    var_x = var_x/nEx;
    var_y = var_y/nEx;
    
    boost_math::double_matrix params(1,4);
    params(0,0) = mu_x;
    params(0,1) = mu_y;
    params(0,2) = var_x;
    params(0,3) = var_y;
    
    /*
    float mu_x2 = 0.37;   
    float mu_y2 = 12.8;
    float sigma_x2 = 9.4;
    float sigma_y2 = 16.6;
    */
    
    cout << "pos prior params" << endl;
    
    cout << "nEx: " << nEx << endl;
    cout << "mu_x: " << mu_x << endl;
    cout << "mu_y: " << mu_y << endl;
    cout << "var_x: " << var_x << endl;
    cout << "var_y: " << var_y << endl;
    
    QString qsFilename = (part_app.m_exp_param.class_dir() + "/torso_pos_prior.mat").c_str();
    cout << "saving " << qsFilename.toStdString() << endl;
    MATFile *f = matlab_io::mat_open(qsFilename, "wz");
    assert(f != 0);
    
    matlab_io::mat_save_double_matrix(f, "params", params);
    
    cout << endl;
  }
    
  void setTorsoPosPrior(const PartApp &part_app, vector<vector<FloatGrid3> > &log_part_detections, boost_math::double_matrix &pos_prior_params, int rootpart_idx){
    
    cout << "\nsetTorsoPosPriorLSP()" << endl;
    
    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    
    float mu_x = pos_prior_params(0,0);   
    float mu_y = pos_prior_params(0,1);
    float var_x = pos_prior_params(0,2);
    float var_y = pos_prior_params(0,3);
    
    float img_c_x = 0.5*img_width;
    float img_c_y = 0.5*img_height;
    
    float weight = part_app.m_exp_param.torso_pos_prior_weight();
    cout << "torso_pos_prior_weight: " << weight << endl;
    
    float sigma_weight  = 1.0;
    
    float var_weight = square(sigma_weight);
    var_x *= var_weight;
    var_y *= var_weight;
    
    cout << "var_x: " << var_x << "; var_y: " << var_y << endl;
    
    boost_math::double_matrix pred_pos(img_height, img_width);
    for (int iy = 0; iy < img_height; ++iy) 
      for (int ix = 0; ix < img_width; ++ix){
	
	float ix_rel = img_c_x - ix;
	float iy_rel = img_c_y - iy;

	float score = 0.0;
	float score_x = exp(-0.5*pow(ix_rel-mu_x,2)/var_x);
	float score_y = exp(-0.5*pow(iy_rel-mu_y,2)/var_y);
	score = weight*score_x*score_y;
	
	if (score < 1e-4)
	  score = 1e-4;
	score = log(score);
	
	pred_pos(iy,ix) = score;
      }
    
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
      for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
	for (int iy = 0; iy < img_height; ++iy) 
	  for (int ix = 0; ix < img_width; ++ix){
	    log_part_detections[rootpart_idx][scaleidx][rotidx][iy][ix] += pred_pos(iy,ix);
	  }
      }
  }
  
  void getRotParams(const PartApp &part_app, int imgidx, boost_math::double_matrix &rot_params, bool bTest){
    
    cout << "\ngetRotParams()" << endl;
    
    int nParts = part_app.m_part_conf.part_size();
    QString qsPredDataDir = part_app.m_exp_param.pred_data_test_dir().c_str();
    rot_params.resize(nParts,3);
    
    QString qsListname = "test";
    if (!bTest)
      qsListname = "train";
    
    QString qsFilename = qsPredDataDir + "/" + qsListname + "list_params_rot_imgidx_" + QString::number(imgidx) + ".mat";
    cout << "loading " << qsFilename.toStdString().c_str() << endl;
    boost_math::double_matrix rot_params_all;
    matlab_io::mat_load_double_matrix(qsFilename, "rot_" + qsListname, rot_params_all);
    
    qsFilename = qsPredDataDir + "/" + qsListname + "list_pred_rot_imgidx_" + QString::number(imgidx) + ".mat";
    cout << "loading " << qsFilename.toStdString().c_str() << endl;
    boost_math::double_matrix rot_clus_all;
    matlab_io::mat_load_double_matrix(qsFilename, "clusidx_" + qsListname, rot_clus_all);
    
    for (int pidx = 0; pidx < nParts; ++pidx) {
      
      if (part_app.m_part_conf.part(pidx).is_detect()) {
		
	rot_params(pidx,0) = rot_params_all(pidx,0);
	rot_params(pidx,1) = square(rot_params_all(pidx,1));
	rot_params(pidx,2) = rot_clus_all(pidx,0);
	cout << "imgidx: " << imgidx << "; pidx: " << pidx << "; mix_compidx: " << rot_params(pidx,2) << endl;
      }
    }
    
  }
  
  void getRotScoreGrid(const PartApp &part_app, vector<vector<FloatGrid3> > &log_rot_scores, boost_math::double_matrix &rot_params){
    
    cout << "\ngetRotScoreGrid()" << endl;
    QString qsPredDataDir = part_app.m_exp_param.pred_data_dir().c_str();
    
    int img_width = log_rot_scores[0][0].shape()[2];
    int img_height = log_rot_scores[0][0].shape()[1];
    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    
    for (int pidx = 0; pidx < nParts; ++pidx) {
      
      if (part_app.m_part_conf.part(pidx).is_detect()) {
	
	boost_math::double_vector pred_rot(nRotations);
	
	float mu = rot_params(pidx,0);
	float var = rot_params(pidx,1);
	
	cout << "pidx: " << pidx << ", mu: " << mu << ", var: " << var << endl;
	
	float max_score = -1000;
	
	for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
	  
	  float rot = rot_from_index(part_app.m_exp_param, rotidx)/180*M_PI;
	  float score = 0.0;
	  
	  score = exp(-0.5*square(rot-mu)/var);
	  if (score < 1e-4)
	    score = 1e-4;
	  score = log(score);
	  
	  pred_rot[rotidx] = score;
	  
	  if (max_score < score)
	    max_score = score; 
	}
	
	cout << "max_score: " << max_score << endl;
	
	for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	  for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
	    for (int iy = 0; iy < img_height; ++iy) 
	      for (int ix = 0; ix < img_width; ++ix){
		log_rot_scores[pidx][scaleidx][rotidx][iy][ix] = pred_rot[rotidx];
	      }
	  }
	
      }// is_detect
    }// pidx
    
  }
  
  void getRootPosDet(const PartApp &part_app, int imgidx, int rootpart_idx, boost_math::double_vector &rootpos_det, bool bTest){
    
    cout << "\ngetRootPosDet()" << endl;
    int rootidx_det = -1;
    if (part_app.m_exp_param.has_rootidx_det())
      rootidx_det = part_app.m_exp_param.rootidx_det();
    
      int root_pos_x = -1, root_pos_y = -1;
      
      if (part_app.m_exp_param.has_use_gt_torso() && part_app.m_exp_param.use_gt_torso()){
	cout << "\nWARNING! Using GT torso position!\n" << endl;
	PartBBox root_bbox;
	get_part_bbox(part_app.m_test_annolist[imgidx][0], part_app.m_part_conf.part(rootpart_idx), root_bbox);

	root_pos_x = root_bbox.part_pos(0);
	root_pos_y = root_bbox.part_pos(1);
      }
      else{
	cout << "rootidx_det: " << rootidx_det << endl;
	
	QString qsLogDir = "";
	if (bTest){
	  assert(part_app.m_exp_param.has_torso_det_test_dir());
	  qsLogDir = part_app.m_exp_param.torso_det_test_dir().c_str();
	}
	else{
	  assert(part_app.m_exp_param.has_torso_det_train_dir());
	  qsLogDir = part_app.m_exp_param.torso_det_train_dir().c_str();
	}
	
	QString qsFilename = qsLogDir + "/pose_est_imgidx" + padZeros(QString::number(imgidx),4) + ".mat";
	cout << "loading " << qsFilename.toStdString().c_str() << endl;
	boost_math::double_matrix best_conf;
	matlab_io::mat_load_double_matrix(qsFilename, "best_conf", best_conf);
	
	root_pos_x = best_conf(rootidx_det,4);
	root_pos_y = best_conf(rootidx_det,5);
      }
      rootpos_det.resize(2);
      rootpos_det(0) = root_pos_x;
      rootpos_det(1) = root_pos_y;
  }
  
  void getPosParams(const PartApp &part_app, int imgidx, boost_math::double_matrix &pos_params, int rootpart_idx, bool bTest){
    
    cout << "\ngetPosParams()" << endl;
    
    QString qsPredDataDir = part_app.m_exp_param.pred_data_test_dir().c_str();
    int nParts = part_app.m_part_conf.part_size();
    pos_params.resize(nParts, 5);
    
    QString qsListname = "test";
    if (!bTest)
      qsListname = "train";
    
    QString qsFilename = qsPredDataDir + "/" + qsListname + "list_params_pos_imgidx_" + QString::number(imgidx) + ".mat";
    cout << "loading " << qsFilename.toStdString().c_str() << endl;
    boost_math::double_matrix pos_params_all;
    matlab_io::mat_load_double_matrix(qsFilename, "pos_" + qsListname, pos_params_all);
    
    qsFilename = qsPredDataDir + "/" + qsListname + "list_pred_pos_imgidx_" + QString::number(imgidx) + ".mat";
    cout << "loading " << qsFilename.toStdString().c_str() << endl;
    boost_math::double_matrix pos_clus_all;
    matlab_io::mat_load_double_matrix(qsFilename, "clusidx_" + qsListname, pos_clus_all);

    for (int pidx = 0; pidx < nParts; ++pidx) {
      
      if (pidx == rootpart_idx)
      	continue;
               
      if (part_app.m_part_conf.part(pidx).is_detect()) {
	
	pos_params(pidx,0) = pos_params_all(pidx,0);//mu_x;
	pos_params(pidx,1) = pos_params_all(pidx,1);//mu_y;
	pos_params(pidx,2) = square(pos_params_all(pidx,2));//var_x;
	pos_params(pidx,3) = square(pos_params_all(pidx,3));//var_y;
	pos_params(pidx,4) = pos_clus_all(pidx,0);
	
	cout << "imgidx: " << imgidx << "; pidx: " << pidx << "; mix_compidx: " <<  pos_params(pidx,4) << endl;
      }
    }
  }
  
  void getPosScoreGrid(const PartApp &part_app, vector<vector<FloatGrid3> > &log_pos_scores, int imgidx, boost_math::double_matrix &pos_params, int rootpart_idx, boost_math::double_vector &rootpos_det){
    
    cout << "\ngetPosScoreGrid()" << endl;
    
    int img_width = log_pos_scores[0][0].shape()[2];
    int img_height = log_pos_scores[0][0].shape()[1];
    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();

    float var_weight = 1.0;

    for (int pidx = 0; pidx < nParts; ++pidx) {
      
      if (pidx == rootpart_idx)
      	continue;
                  
      if (part_app.m_part_conf.part(pidx).is_detect()) {
	
	boost_math::double_matrix pred_pos(img_height, img_width);

	float mu_x = pos_params(pidx,0);
	float mu_y = pos_params(pidx,1);
	float var_x = pos_params(pidx,2)*var_weight;
	float var_y = pos_params(pidx,3)*var_weight;
	
	cout << "pidx: " << pidx << ", mu_x: " << mu_x << ", mu_y: " << mu_y << ", var_x: " << var_x << ", var_y: " << var_y << endl;
	float max_score = -1000;
	for (int iy = 0; iy < img_height; ++iy) 
	  for (int ix = 0; ix < img_width; ++ix){
	    	    
	    float ix_rel = ix - rootpos_det(0);
	    float iy_rel = iy - rootpos_det(1);
	    
	    float score = 0;
	    
	    float score_x = exp(-0.5*pow(ix_rel-mu_x,2)/var_x);
	    float score_y = exp(-0.5*pow(iy_rel-mu_y,2)/var_y);
	    score = score_x*score_y;
	    if (score < 1e-4)
	      score = 1e-4;
	    score = log(score);
	    
	    pred_pos(iy,ix) = score;
	    if (max_score < score)
	      max_score = score; 
	  }
	
	cout << "max_score " << max_score << endl;
	for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	  for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
	    for (int iy = 0; iy < img_height; ++iy) 
	      for (int ix = 0; ix < img_width; ++ix){
		log_pos_scores[pidx][scaleidx][rotidx][iy][ix] = pred_pos(iy,ix);
	      }
	  }
      }
    }
    
    bool bVis = false;
    if (bVis){
      QString qsResDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/vis_pos_pred_test/").c_str();
      if (!filesys::check_dir(qsResDir)) {
	cout << "creating " << qsResDir.toStdString().c_str() << endl;
	filesys::create_dir(qsResDir);
      }
      QString qsFname = qsResDir + "/pos_pred_imgidx_" + padZeros(QString::number(imgidx), 4) + ".mat";
      for (int pidx = 0; pidx < nParts; ++pidx){ 
	for (int iy = 0; iy < img_height; ++iy) 
	  for (int ix = 0; ix < img_width; ++ix)
	    if (log_pos_scores[pidx][0][0][iy][ix] > log(1e-4))
	      log_pos_scores[0][0][0][iy][ix] = log_pos_scores[pidx][0][0][iy][ix];
	QString qsFname2 = qsResDir + "/pos_pred_imgidx_" + padZeros(QString::number(imgidx), 4) + "_pidx_" + padZeros(QString::number(pidx), 4) + ".mat";
	matlab_io::mat_save_multi_array(qsFname2, "log_pos_scores", log_pos_scores[pidx][0]);
      }
      matlab_io::mat_save_multi_array(qsFname, "log_pos_scores", log_pos_scores[0][0]);
    }
  }

  void addLoadDPMScore(const PartApp &part_app, vector<vector<FloatGrid3> > &log_part_detections, 
		       int imgidx, float dpm_weight, int nRotationsDPM, 
		       QString qsDPMparentDir, bool bUsePartIdx, int pidx_only){

    cout << "\nAddDPMscoreCell()" << endl;
    QString qsPredDataDir = part_app.m_exp_param.pred_data_dir().c_str();

    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];
    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    
    cout << "dpm_weight: " << dpm_weight << endl;
    
    for (int pidx = 0; pidx < nParts; ++pidx) {
      if (pidx_only > -1 && pidx_only < nParts && pidx_only != pidx)      
	continue;
      
      cout << "\nimgidx: " << imgidx << "; pidx: " << pidx << endl;
      
      if (part_app.m_part_conf.part(pidx).is_detect()){
	
	std::vector<FloatGrid2> dpmScoreGrid(nRotationsDPM, FloatGrid2(boost::extents[img_height][img_width]));    

	QString qsDPMdir = qsDPMparentDir;
	if (bUsePartIdx)
	  qsDPMdir += QString("/pidx_") + padZeros(QString::number(pidx), 4);
	loadDPMScoreGrid(qsDPMdir, imgidx, dpmScoreGrid, true);

	for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	  for (int rotidx = 0; rotidx < nRotations; ++rotidx){
	    int rotidx_dpm = (nRotationsDPM == nRotations ? rotidx : 0);
	    for (int iy = 0; iy < img_height; ++iy) 
	      for (int ix = 0; ix < img_width; ++ix){
		float val = dpmScoreGrid[rotidx_dpm][iy][ix];
		log_part_detections[pidx][scaleidx][rotidx][iy][ix] += (val > 1e-4 ? dpm_weight*log(val) : log(1e-4));	
	      }
	  }
      }//is_detect
    }
  }
  
  void addDPMScore(const PartApp &part_app, std::vector<std::vector<FloatGrid3> > &log_part_detections, std::vector<FloatGrid2> log_dpmPriorGrid, int pidx, float dpm_weight){
    
    cout << "\naddDPMScore()" << endl;
    
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    int img_height = log_part_detections[pidx][0].shape()[1];
    int img_width = log_part_detections[pidx][0].shape()[2];
    
    cout << "pidx: " << pidx << endl;
    
    cout << "dpm_weight: " << dpm_weight << endl;
    assert(dpm_weight > 0);
    
    //assert(dpm_prior_grid.size() == nRotations);
    assert(log_dpmPriorGrid[0].shape()[0] == img_height);
    assert(log_dpmPriorGrid[0].shape()[1] == img_width);
               
    bool bRot = false;
    if (log_dpmPriorGrid.size() > 1)
      bRot = true;
    
    cout << "bRot: " << bRot << endl;
    
    // DPM prior
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx)
      for (int ridx = 0; ridx < nRotations; ++ridx) {
	for (int iy = 0; iy < img_height; ++iy)
	  for (int ix = 0; ix < img_width; ++ix) {
	    if (bRot)
	      log_part_detections[pidx][scaleidx][ridx][iy][ix] += dpm_weight*log_dpmPriorGrid[ridx][iy][ix];
	    else
	      log_part_detections[pidx][scaleidx][ridx][iy][ix] += dpm_weight*log_dpmPriorGrid[0][iy][ix];
	  }
      }
    
  }
  
  void addExtraUnary(const PartApp &part_app, 
		     vector<vector<FloatGrid3> > &log_part_detections, 
		     const vector<vector<FloatGrid3> > &log_extra_unary_scores,
		     float weight){
    
    cout << "addExtraUnary()" << endl;
    
    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    int img_height = log_part_detections[0][0].shape()[1];
    int img_width = log_part_detections[0][0].shape()[2];
    
    cout << "weight: " << weight << endl;
    
    for (int pidx = 0; pidx < nParts; ++pidx) 
      for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	for (int ridx = 0; ridx < nRotations; ++ridx)
	  for (int iy = 0; iy < img_height; ++iy)
	    for (int ix = 0; ix < img_width; ++ix) 
	      log_part_detections[pidx][scaleidx][ridx][iy][ix] += weight*log_extra_unary_scores[pidx][scaleidx][ridx][iy][ix];
    
  }
  
  void loadDPMScoreGrid(QString qsDPMdir, int imgidx, std::vector<FloatGrid2> &dpmPriorGrid, bool bIsCell){
    
    cout << "loadDPMScoreGrid()" << endl;
    QString qsScoreGridName = qsDPMdir + "/imgidx_" + padZeros(QString::number(imgidx + 1), 4) + ".mat";
    cout << "load " << qsScoreGridName.toStdString().c_str() << endl;
    assert(filesys::check_file(qsScoreGridName));
        
    if (bIsCell){
      std::vector<std::vector<FloatGrid2> > _dpmPriorGrid;
      
      MATFile *f = matlab_io::mat_open(qsScoreGridName, "r");
      assert(f != 0);
      matlab_io::mat_load_multi_array_vec2(f, "scoregrid", _dpmPriorGrid);
      if (f != 0)
	matlab_io::mat_close(f);
      
      assert(_dpmPriorGrid[0].size() == 1);
      //dpmPriorGrid = _dpmPriorGrid[0];
      assert(dpmPriorGrid.size() == _dpmPriorGrid.size());
      for (int gidx = 0; gidx < _dpmPriorGrid.size(); ++gidx)
	dpmPriorGrid[gidx] = _dpmPriorGrid[gidx][0];
    }
    else{
      FloatGrid2 _dpmPriorGrid = matlab_io::mat_load_multi_array<FloatGrid2>(qsScoreGridName, "scoregrid"); 
      dpmPriorGrid.push_back(_dpmPriorGrid);
    }
  }
  
  int trainLDA(const PartApp &part_app, int idxMode, int idxFactor){
    
    cout << "trainLDA()" << endl;
    // idxMode == 0 - pairwise; 1 - unary rot; 2 - unary pos;
    // idxFactor - joint/part id
    
    QString qsCMD = "run_trainClass.sh "  + QString(MATLAB_RUNTIME);
 
    int nClus = 1;
    if (idxMode == 0) // pairwise
      nClus = part_app.m_part_conf.joint(0).num_joint_types();
    else if (idxMode == 1 || idxMode == 2) // unaries
      nClus = part_app.m_part_conf.part(0).num_pred_part_types();
    else
      assert(0);
    
    assert(part_app.m_exp_param.validation_dataset_size() == 1);
    
    QString qsCommandLine = qsCMD + " " + QString::number(idxMode) + " " + QString::number(nClus) + " " +
      QString::number(idxFactor) + " " + (part_app.m_exp_param.pred_data_dir() + " " + 
					  part_app.m_exp_param.poselet_resp_val_dir() + " " + 
					  part_app.m_exp_param.torso_det_train_dir() + " " +
					  part_app.m_exp_param.validation_dataset(0)).c_str();
    
    part_detect::runMatlabCode(qsCommandLine);

    return 1;
    
  }
  
  int predictFactors(const PartApp &part_app,  int idxMode, int imgidx){
    
    cout << "predictFactors()" << endl;
    // idxMode == 0 - pairwise; 1 - unary rot; 2 - unary pos;
    
    QString qsCMD = "run_predictFactorsImg.sh " + QString(MATLAB_RUNTIME);
    
    QString qsCommandLine = qsCMD + " " + QString::number(idxMode) + " " + 
      QString::number(imgidx) + " " + (part_app.m_exp_param.pred_data_dir() + " " + 
				       part_app.m_exp_param.pred_data_test_dir() + " " + 
				       part_app.m_exp_param.poselet_resp_test_dir() + " " + 
				       part_app.m_exp_param.torso_det_test_dir()).c_str();
    
    part_detect::runMatlabCode(qsCommandLine);
    
    return 1;
    
  }
  
  void savePredictedPartConf(const PartApp &part_app, int firstidx, int lastidx, bool bIsTest)
  {
    int nJoints = part_app.m_part_conf.joint_size();
    bool flip = false;

    vector<Joint> joints(nJoints);
    vector<boost_math::double_matrix> jointsTypesList(nJoints);
    
    QString qsListName = "";
    QString qsVisDirName = "";
    if (bIsTest){
      qsListName = "test";
      qsVisDirName = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/vis_pred_test/").c_str();
    }
    else{
      qsListName = "train";
      qsVisDirName = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/vis_pred_train/").c_str();
    }
      
    if (!filesys::check_dir(qsVisDirName)) {
      cout << "creating " << qsVisDirName.toStdString().c_str() << endl;
      filesys::create_dir(qsVisDirName);
    }
    
    QString qsPredDataDir = part_app.m_exp_param.pred_data_test_dir().c_str();
    int nJointTypes = part_app.m_part_conf.joint(0).num_joint_types();      
    
    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx){ 
      
      cout << "imgidx: " << imgidx << endl;
      QString qsImgVisDir = qsVisDirName + "/im" + padZeros(QString::number(imgidx), 4);
      if (!filesys::check_dir(qsImgVisDir)) {
	cout << "creating " << qsImgVisDir.toStdString().c_str() << endl;
	filesys::create_dir(qsImgVisDir);
      }

      QString qsFilename = qsPredDataDir + "/" + qsListName + "list_pred_pwise_imgidx_" + QString::number(imgidx) + ".mat";
	boost_math::double_vector clus_all;
	matlab_io::mat_load_double_vector(qsFilename, "clusidx_" + qsListName, clus_all);

      for (int jidx = 0; jidx < nJoints; ++jidx) {
	
	int tidx = clus_all(jidx);
	
	int child_idx = part_app.m_part_conf.joint(jidx).child_idx();
	int parent_idx = part_app.m_part_conf.joint(jidx).parent_idx();
	QString qsFilename = QString("joint_") +
	  QString::number(child_idx) + "_" + 
	  QString::number(parent_idx) + "_tidx_" + QString::number(tidx) + ".mat";
	QString qsFullFilenameFrom = (part_app.m_exp_param.spatial_dir() + "/" + qsFilename.toStdString()).c_str();
	QString qsFullFilenameTo = qsImgVisDir + "/" + qsFilename;
	
	cout << "copy " << qsFullFilenameFrom.toStdString().c_str() << endl;
	
	if (not filesys::copy_file(qsFullFilenameFrom, qsFullFilenameTo))
	  assert(false && "error while copying the file");
      }
    }
  }
  
  void computePoseLL2(const PartApp &part_app, int firstidx, int lastidx){
    
    uint nJoints = part_app.m_part_conf.joint_size();
    //vector<Joint> joints(nJoints);
    //loadJoints(part_app, joints, false);
    
    double parent_scale = 1.0;
    double child_scale = 1.0;
    boost_math::double_vector L_all(1);
    //boost_math::double_vector L_all(part_app.m_test_annolist.size());
    //for (uint imgidx = 0; imgidx < part_app.m_test_annolist.size(); ++imgidx) {
    //  L_all[imgidx] = -1000;
    //}
    
    QString qsResDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/poseLL_test/").c_str();
    if (!filesys::check_dir(qsResDir)) {
      cout << "creating " << qsResDir.toStdString().c_str() << endl;
      filesys::create_dir(qsResDir);
    }
    
    for (uint imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      
      vector<Joint> joints(nJoints);
      loadJoints(part_app, joints, false, imgidx);
      
      double L = 0.0;
      int nFactors = 0;
      for (int jidx = 0; jidx < nJoints; jidx++){
	
	if (not(jidx == 2 || jidx == 4 || jidx == 5 || jidx == 7 || jidx == 10 || 
		jidx == 13 || jidx == 15 || jidx == 16 || jidx == 18))
	  continue;
	
	int parent_idx = joints[jidx].parent_idx;
	int child_idx = joints[jidx].child_idx;
	
	int ridx = 0;
	
	if (annorect_has_part(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(parent_idx))){
	  // compute part bounding box
	  PartBBox bbox_parent;
	  if (get_part_bbox(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(parent_idx), bbox_parent)){
	    if (annorect_has_part(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(child_idx))){
	      // compute part bounding box
	      PartBBox bbox_child;
	      if (get_part_bbox(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(child_idx), bbox_child)){
		boost_math::double_vector joint_pos_offset;
		double joint_rot_offset, rot_step_size;
		
		double p_joint = disc_ps::eval_joint_factor(joints[jidx], bbox_parent, bbox_child, parent_scale, child_scale, joint_pos_offset, joint_rot_offset, rot_step_size);
		if (isnan(p_joint))
		  cout << "nan; jidx: " << jidx << endl;
		
		L += log(p_joint);
		nFactors++;
		
	      }
	    }//parent_idx
	  }
	}//childx
	
      }//jidx
      L /= nFactors;
      //L_all[imgidx] = L;
      L_all[0] = L;
      cout << "imgidx: " << imgidx << "; L: " << L << endl;
            
      QString qsFname = qsResDir + "/poseLL_imgidx_" + padZeros(QString::number(imgidx), 4) + ".mat";
      cout << "saving " << qsFname.toStdString() << endl;
      MATFile *f = matlab_io::mat_open(qsFname, "wz");
      assert(f != 0);
      matlab_io::mat_save_double_vector(f, "poseLL", L_all);
      
    }//imgidx
    //QString qsFname = "/BS/leonid-people-3d/work/data/new_dataset/annolist_merge_1304_all_test/test/h200/singlePerson/pose22LL.mat";
    //cout << "saving " << qsFname.toStdString() << endl;
    //MATFile *f = matlab_io::mat_open(qsFname, "wz");
    //assert(f != 0);
    //matlab_io::mat_save_double_vector(f, "poseLL", L_all);
  }
  
  void computePoseLL(const PartApp &part_app, int firstidx, int lastidx){
    
    uint nJoints = part_app.m_part_conf.joint_size();
    vector<Joint> joints(nJoints);
    loadJoints(part_app, joints, false);
    
    double parent_scale = 1.0;
    double child_scale = 1.0;
    //boost_math::double_vector L_all(1);
    boost_math::double_vector L_all(part_app.m_test_annolist.size());
    
    for (uint imgidx = 0; imgidx < part_app.m_test_annolist.size(); ++imgidx) {
      L_all[imgidx] = -1000;
    }
    
    QString qsResDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/poseLL_test_upper_body/").c_str();
    if (!filesys::check_dir(qsResDir)) {
      cout << "creating " << qsResDir.toStdString().c_str() << endl;
      filesys::create_dir(qsResDir);
    }
    
    for (uint imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      
      //vector<Joint> joints(nJoints);
      //loadJoints(part_app, joints, false, imgidx);
      
      double L = 0.0;
      int nFactors = 0;
      for (int jidx = 0; jidx < nJoints; jidx++){
	
	//if (not(jidx == 13 || jidx == 15 || jidx == 16 || jidx == 18 || jidx == 4 || jidx == 5)) 
	if (not(jidx == 2 || jidx == 4  || jidx == 5  || jidx == 7 
		|| jidx == 10 || jidx == 13 || jidx == 15 || jidx == 16 || jidx == 18))
	  continue;
	
	int parent_idx = joints[jidx].parent_idx;
	int child_idx = joints[jidx].child_idx;
	
	int ridx = 0;
	
	if (annorect_has_part(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(parent_idx))){
	  // compute part bounding box
	  PartBBox bbox_parent;
	  if (get_part_bbox(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(parent_idx), bbox_parent)){
	    if (annorect_has_part(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(child_idx))){
	      // compute part bounding box
	      PartBBox bbox_child;
	      if (get_part_bbox(part_app.m_test_annolist[imgidx][ridx], part_app.m_part_conf.part(child_idx), bbox_child)){
		boost_math::double_vector joint_pos_offset;
		double joint_rot_offset, rot_step_size;
		
		double p_joint = disc_ps::eval_joint_factor(joints[jidx], bbox_parent, bbox_child, parent_scale, child_scale, joint_pos_offset, joint_rot_offset, rot_step_size);
		if (isnan(p_joint))
		  cout << "nan; jidx: " << jidx << endl;
		
		L += log(p_joint);
		nFactors++;
		
	      }
	    }//parent_idx
	  }
	}//childx
	
      }//jidx
      L /= nFactors;
      L_all[imgidx] = L;
      //L_all[0] = L;
      cout << "imgidx: " << imgidx << "; L: " << L << endl;
    }//imgidx
    //QString qsFname = "/BS/leonid-people-3d/work/data/new_dataset/annolist_merge_1304_all_test/test/h200/singlePerson/pose22LL_ubody.mat";
    //QString qsFname = "/BS/leonid-people-3d/work/data/lsp_dataset/images/png/h200/pose22LL.mat";
    QString qsFname = "/home/leonid/data/ramanan/ramanan_test_h200_color/pose22LL.mat";

    cout << "saving " << qsFname.toStdString() << endl;
    getchar();
    MATFile *f = matlab_io::mat_open(qsFname, "wz");
    assert(f != 0);
    matlab_io::mat_save_double_vector(f, "poseLL", L_all);
  }

}
