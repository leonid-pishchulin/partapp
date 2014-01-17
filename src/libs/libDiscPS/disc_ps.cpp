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

#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <QFileInfo>
#include <QDateTime>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <dai/alldai.h>

#include <libMultiArray/multi_array_op.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libMatlabIO/matlab_io.hpp>

#include <libPartApp/partapp_aux.hpp>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libPictStruct/objectdetect.h>

//DEBUG
#include <libPartEval/parteval.h>
//DEBUG

#include "disc_ps.h"
#include "unique_vect.h"
#include "factors.h"

#include "disc_sample.hpp"

#include "FactorDefs.pb.h"

using namespace std;
using namespace boost::lambda;
using boost::multi_array_types::index_range;

namespace disc_ps {

  using boost_math::double_vector;
  using object_detect::Joint;


  /** 
      helper functions for conversion between node index and part/subject index

      these are used in pictorial structures model which includes parts from multiple people 
   */
  uint nidx_from_pidx(uint nParts, uint didx, uint pidx)
  {
    assert(pidx < nParts);
    return nParts*didx + pidx;    
  }

  uint pidx_from_nidx(uint nParts, uint nidx)
  {
    return nidx % nParts;
  }

  uint didx_from_nidx(uint nParts, uint nidx)
  {
    return nidx / nParts;
  }

  QString get_samples_dir(const ExpParam &exp_param, uint imgidx) 
  {
    return QString::fromStdString(exp_param.dai_samples_dir() + "/samples_imgidx") + padZeros(QString::number(imgidx), 4);
  }

  QString get_samples_filename(const ExpParam &exp_param, uint imgidx, uint pidx)
  {
    return get_samples_dir(exp_param, imgidx) + "/samples_pidx" + QString::number(pidx) + ".mat";
  }

  QString get_samples_post_dir(const ExpParam &exp_param)
  {
    if (QString::fromStdString(exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) { 
      return QString::fromStdString(exp_param.log_dir() + "/" + exp_param.log_subdir() + 
				    "/part_marginals_samples_post");
    }
    else if (QString::fromStdString(exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
      return QString::fromStdString(exp_param.log_dir() + "/" + exp_param.log_subdir() + "/test_scoregrid_samples_post");
    }
    else {
      assert(false && "unknown part samples type");
      return "";
    }
  }

  QString get_samples_post_filename(const ExpParam &exp_param, uint imgidx) {
    return get_samples_post_dir(exp_param) + "/samples_imgidx" + padZeros(QString::number(imgidx), 4) + "_post.mat";
  }
  
  QString get_factors_dir(const ExpParam &exp_param, uint imgidx) 
  {
    if (QString::fromStdString(exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) { 

      QString qsSamplesDir = QString::fromStdString(exp_param.dai_samples_dir());
      int n = qsSamplesDir.length() - QString("part_marginals_samples").length();

      return qsSamplesDir.left(n) + "/part_marginals_samples_factors/imgidx" + QString::number(imgidx);

//       return QString::fromStdString(exp_param.log_dir() + "/" + exp_param.log_subdir() + 
// 				    "/part_marginals_samples_factors/imgidx")  + QString::number(imgidx);

    }
    else if (QString::fromStdString(exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
      QString qsSamplesDir = QString::fromStdString(exp_param.dai_samples_dir());
      int n = qsSamplesDir.length() - QString("test_scoregrid_samples").length();
      
      return qsSamplesDir.left(n) + "/test_scoregrid_samples_factors/imgidx" + QString::number(imgidx);

//      return QString::fromStdString(exp_param.log_dir() + "/" + exp_param.log_subdir() + "/test_scoregrid_samples_factors/imgidx") + QString::number(imgidx);
    }
    else {
      assert(false && "unknown part samples type");
      return "";
    }
  }

  QString get_factor_filename(const ExpParam &exp_param, uint imgidx, QString qsFactorType, uint fidx) {
    assert(qsFactorType == "abc" || qsFactorType == "spatial");

    return get_factors_dir(exp_param, imgidx) + "/" + qsFactorType + "_" + QString::number(fidx) + ".mat";
  }

  void partSample(const PartApp &part_app, int firstidx, int lastidx)
  {
    cout << "disc_ps::partSample" << endl << endl;

    /** make sure we don't overwrite samples from other experiment */
    assert(!part_app.m_bExternalSamplesDir && 
	   "using external samples dir, new samples should be generated using original experiment file");

    int nParts = part_app.m_part_conf.part_size();

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {

      QString qsPrecomputedSamplesDir = get_samples_dir(part_app.m_exp_param, imgidx);

      cout << "saving samples to " << qsPrecomputedSamplesDir.toStdString() << endl;

      if (!filesys::check_dir(qsPrecomputedSamplesDir)) 
	assert(filesys::create_dir(qsPrecomputedSamplesDir));
      
      for (int pidx = 0; pidx < nParts; ++pidx) {

	vector<int> vect_scale_idx;
        vector<int> vect_rot_idx; 
        vector<int> vect_iy; 
        vector<int> vect_ix;
	vector<double> vect_scores;

	/** for now sample at single scale only and no flips */
	assert(part_app.m_exp_param.flip_orientation() == false);

	bool flip = false;

	
	int scaleidx = 0;
	bool bInterpolate = false;
	
	std::vector<std::vector<FloatGrid2> > score_grid;
	QString qsScoreGridDir = part_app.m_exp_param.scoregrid_dir().c_str();
	part_app.loadScoreGrid(score_grid, imgidx, pidx, flip, bInterpolate, qsScoreGridDir, 
			       part_app.m_test_annolist[imgidx].imageName().c_str());
	
	assert(score_grid.size() == 1);
	assert(score_grid[0].size() == part_app.m_exp_param.num_rotation_steps());
	
	double det_score_threshold = 0.0;
	
	/** extract samples either from posteriors or from part detections */

	if (part_app.m_exp_param.dai_samples_type() == "part_det") {
	  assert(part_app.m_exp_param.num_scale_steps() == 1);

	  for (uint ridx = 0; ridx < score_grid[0].size(); ++ridx) {
	    uint img_height = score_grid[0][ridx].shape()[0];
	    uint img_width = score_grid[0][ridx].shape()[1];

	    for (uint ix = 0; ix < img_width; ++ix) 
	      for (uint iy = 0; iy < img_height; ++iy) {
		if (score_grid[0][ridx][iy][ix] > det_score_threshold) {
		  vect_rot_idx.push_back(ridx);
		  vect_iy.push_back(iy);
		  vect_ix.push_back(ix);
		  vect_scale_idx.push_back(scaleidx);
		  vect_scores.push_back(score_grid[scaleidx][ridx][iy][ix]);
		}
	      }
	  }
	  
	  cout << "part " << pidx << "," << "  samples with scores > " << det_score_threshold  << ": " << vect_rot_idx.size() << endl;

	  if (vect_scores.size() > (uint)part_app.m_exp_param.dai_num_samples()) {
	    /** pick the best */

	    vector<std::pair<double, int> > all_scores;
	    for (uint idx = 0; idx < vect_scores.size(); ++idx) 
	      all_scores.push_back(std::pair<double, int>(vect_scores[idx], idx));

	    /** sort in descending order */
	    std::sort(all_scores.begin(), all_scores.end(), bind(&std::pair<double,int>::first, _1) > bind(&std::pair<double,int>::first, _2));
	    assert(all_scores[0].first > all_scores[1].first);

	    vector<int> _vect_rot_idx; 
	    vector<int> _vect_iy; 
	    vector<int> _vect_ix;
	    vector<int> _vect_scale_idx;
	    vector<double> _vect_scores;
	    
	    for (int idx = 0; idx < part_app.m_exp_param.dai_num_samples(); ++idx) {
	      int idx2 = all_scores[idx].second;
	      
	      _vect_rot_idx.push_back(vect_rot_idx[idx2]);
	      _vect_iy.push_back(vect_iy[idx2]);
	      _vect_ix.push_back(vect_ix[idx2]);
	      _vect_scale_idx.push_back(vect_scale_idx[idx2]);
	      _vect_scores.push_back(vect_scores[idx2]);
	    }

	    vect_rot_idx = _vect_rot_idx;
	    vect_ix = _vect_ix;
	    vect_iy = _vect_iy;
	    vect_scale_idx = _vect_scale_idx;
	    vect_scores = _vect_scores;
	  }
  
	  assert(vect_rot_idx.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_ix.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_iy.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_scale_idx.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_scores.size() <= (uint)part_app.m_exp_param.dai_num_samples());

	  assert(vect_ix.size() == vect_rot_idx.size() && 
		 vect_ix.size() == vect_iy.size() && 
		 vect_ix.size() == vect_scale_idx.size() &&
		 vect_ix.size() == vect_scores.size()
		 );


	}
	else if (part_app.m_exp_param.dai_samples_type() == "part_post") {

	  assert(part_app.m_exp_param.dai_bbox_prior() == false);

	  //assert(part_app.m_exp_param.num_scale_steps() == 1);
	  //int scaleidx = 0;

	  assert(part_app.m_exp_param.num_scale_steps() > 0);

	  // 	  FloatGrid3 _part_posterior = part_app.loadPartMarginal(imgidx, pidx, 0, flip);
	  // 	  FloatGrid4 part_posterior(boost::extents[part_app.m_exp_param.num_scale_steps()][_part_posterior.shape()[0]][_part_posterior.shape()[1]][_part_posterior.shape()[2]]);
	  // 	  part_posterior[0] = _part_posterior;

	  FloatGrid4 part_posterior;

	  for (uint scaleidx = 0; scaleidx < part_app.m_exp_param.num_scale_steps(); ++scaleidx) {
	    FloatGrid3 tmp_part_posterior = part_app.loadPartMarginal(imgidx, pidx, scaleidx, flip);

	    if (scaleidx == 0)
	      part_posterior.resize(boost::extents[part_app.m_exp_param.num_scale_steps()][tmp_part_posterior.shape()[0]][tmp_part_posterior.shape()[1]][tmp_part_posterior.shape()[2]]);

	    part_posterior[scaleidx] = tmp_part_posterior;
	  }

	  multi_array_op::computeExpGrid(part_posterior);

	  int rnd_seed = (int)1e5;

	  boost_math::int_matrix idxmat;
	  disc_ps::discrete_sample(part_posterior, part_app.m_exp_param.dai_num_samples(), idxmat, rnd_seed);

	  assert(idxmat.size1() == (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_rot_idx.size() == 0);

	  for (uint idx = 0; idx < idxmat.size1(); ++idx) {
	    vect_scale_idx.push_back(idxmat(idx, 0));
	    vect_rot_idx.push_back(idxmat(idx, 1));
	    vect_iy.push_back(idxmat(idx, 2));
	    vect_ix.push_back(idxmat(idx, 3));
	  }
	  
	  // 	  disc_ps::discrete_sample3(part_posterior, part_app.m_exp_param.dai_num_samples(), 
	  // 				    vect_rot_idx, vect_iy, vect_ix, rnd_seed);

	  uint num_samples = vect_rot_idx.size();
	  assert((int)num_samples == part_app.m_exp_param.dai_num_samples());
	  assert(num_samples == vect_iy.size() && num_samples == vect_ix.size() && num_samples == vect_scale_idx.size());

	  /** BEGIN filter repeating samples */
	  vector<double_vector> V;

	  for (uint samp_idx = 0; samp_idx < num_samples; ++samp_idx) {
	    double_vector v(4);
	    v(0) = vect_scale_idx[samp_idx];
	    v(1) = vect_rot_idx[samp_idx];
	    v(2) = vect_iy[samp_idx];
	    v(3) = vect_ix[samp_idx];
	    
	    V.push_back(v);
	  }

	  vector<double_vector> V2;
	  get_unique_elements(V, V2);

	  num_samples = V2.size();

	  vect_scale_idx.clear();
	  vect_rot_idx.clear();
	  vect_iy.clear();
	  vect_ix.clear();

	  vect_scale_idx.resize(num_samples);
	  vect_rot_idx.resize(num_samples);
	  vect_iy.resize(num_samples);
	  vect_ix.resize(num_samples);

	  for (uint idx = 0; idx < num_samples; ++idx) {
	    vect_scale_idx[idx] = (int)V2[idx](0);
	    vect_rot_idx[idx] = (int)V2[idx](1);
	    vect_iy[idx] = (int)V2[idx](2);
	    vect_ix[idx] = (int)V2[idx](3);
	  }
	  /** END filter repeating samples */

	  for (uint idx = 0; idx < num_samples; ++idx) {
	    int scaleidx = vect_scale_idx[idx];
	    int ridx = vect_rot_idx[idx];
	    int iy = vect_iy[idx];
	    int ix = vect_ix[idx];
	    float score = (score_grid[0][ridx][iy][ix] > det_score_threshold ? score_grid[0][ridx][iy][ix] : det_score_threshold);
	    vect_scores.push_back(score);
	  }
	}
	else {
	  assert(false && "unknown part samples type");
	}
	
	/* DEBUG */
	/*
	vector<object_detect::PartHyp> part_hyp_list;
	QString qsHypDir = "/scratch/BS/pool0/leonid/log_dir/exp_ramanan_075-1000-dense-0125-48-map-fix-inf-bktr/part_marginals/";
	loadPartHyp(part_app, qsHypDir, imgidx, part_hyp_list);
	
	int ridx = part_hyp_list[pidx].m_rotidx;
	int iy = part_hyp_list[pidx].m_y;
	int ix = part_hyp_list[pidx].m_x;
	int num_samples = vect_scale_idx.size();
	vect_scale_idx[num_samples - 1] = scaleidx;
	vect_rot_idx[num_samples - 1] = ridx;
	vect_iy[num_samples - 1] = iy;
	vect_ix[num_samples - 1] = ix;
	vect_scores[num_samples - 1] = score_grid[scaleidx][ridx][iy][ix];
	*/
	/* DEBUG */
		
	/** save samples */
	//QString qsOutputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";
	QString qsOutputFilename = get_samples_filename(part_app.m_exp_param, imgidx, pidx);

	MATFile *f = matlab_io::mat_open(qsOutputFilename, "wz");
	assert(f != 0);

        matlab_io::mat_save_stdcpp_vector(f, "vect_scale_idx", vect_scale_idx);
        matlab_io::mat_save_stdcpp_vector(f, "vect_rot_idx", vect_rot_idx);
        matlab_io::mat_save_stdcpp_vector(f, "vect_iy", vect_iy); 
        matlab_io::mat_save_stdcpp_vector(f, "vect_ix", vect_ix);

	if (vect_scores.size() > 0)
	  matlab_io::mat_save_stdcpp_vector(f, "vect_scores", vect_scores);

	matlab_io::mat_close(f);

      }// parts
    } // images
  }

//   void samples_to_bbox(const PartApp &part_app, int pidx, int scaleidx, 
//                        const vector<int> &vect_rot_idx, const vector<int> &vect_iy, const vector<int> &vect_ix, 
//                        vector<PartBBox> &samples)
//   {
//     int num_samples = vect_rot_idx.size();

//     assert((int)vect_iy.size() == num_samples && 
//            (int)vect_ix.size() == num_samples);

//     for (int samp_idx = 0; samp_idx < num_samples; ++samp_idx) {
//       PartBBox bbox;
//       bbox_from_pos(part_app.m_exp_param, part_app.m_window_param.part(pidx), 
//                     scaleidx, vect_rot_idx[samp_idx], vect_ix[samp_idx], vect_iy[samp_idx], bbox);

//       samples.push_back(bbox);
//     }// samples
//   }

  void addDPMscore(const PartApp &part_app, int imgidx, 
		   int img_width, int img_height, int pidx,
		   int num_samples, Factor1d &boosting_factor){
    
    cout << "addDPMscore" << endl;
    QString qsDPMdir = part_app.m_exp_param.test_dpm_torso_dir().c_str();
    std::vector<FloatGrid2> dpm_prior_grid;

    object_detect::loadDPMScoreGrid(qsDPMdir, imgidx, dpm_prior_grid, false);
    
    assert(dpm_prior_grid[0].shape()[0] == img_height);
    assert(dpm_prior_grid[0].shape()[1] == img_width);
		  
    bool bRot = false;
    if (dpm_prior_grid.size() > 1)
      bRot = true;
		  
    QString qsInputFilename = get_samples_filename(part_app.m_exp_param, imgidx, pidx);
    assert(filesys::check_file(qsInputFilename));

    vector<int> vect_scale_idx; 
    vector<int> vect_rot_idx; 
    vector<int> vect_iy; 
    vector<int> vect_ix;
    cout << "loading samples from " << qsInputFilename.toStdString() << endl;

    bool res5 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scale_idx", vect_scale_idx);
    bool res1 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_rot_idx", vect_rot_idx);
    bool res2 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_iy", vect_iy);
    bool res3 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_ix", vect_ix);
    
    assert(res5 && res1 && res2 && res3);
    
    assert(vect_scale_idx.size() == num_samples);
    assert(vect_rot_idx.size() == num_samples);
    assert(vect_iy.size() == num_samples);
    assert(vect_ix.size() == num_samples);
		  
    for (int sampidx = 0; sampidx < num_samples; ++sampidx) {
      int ridx = vect_rot_idx[sampidx];
      int iy = vect_iy[sampidx];
      int ix = vect_ix[sampidx];
		    
      if (!bRot)
	ridx = 0;
      boosting_factor.fval[sampidx] = sqrt(boosting_factor.fval[sampidx] * dpm_prior_grid[ridx][iy][ix]);
    }
    
  }

  void samples_to_bbox(const PartApp &part_app, int pidx, 
                       const vector<int> &vect_scale_idx, const vector<int> &vect_rot_idx, const vector<int> &vect_iy, const vector<int> &vect_ix, 
                       vector<PartBBox> &samples)
  {
    int num_samples = vect_rot_idx.size();

    assert((int)vect_iy.size() == num_samples && 
           (int)vect_ix.size() == num_samples && 
	   (int)vect_scale_idx.size() == num_samples);

    for (int samp_idx = 0; samp_idx < num_samples; ++samp_idx) {
      PartBBox bbox;
      bbox_from_pos(part_app.m_exp_param, part_app.m_window_param.part(pidx), 
                    vect_scale_idx[samp_idx], vect_rot_idx[samp_idx], vect_ix[samp_idx], vect_iy[samp_idx], bbox);

      samples.push_back(bbox);
    }// samples
  }

  void findObjDai(const PartApp &part_app, int firstidx, int lastidx, bool bForceRecompute)
  {
    cout << "disc_ps::findObjDai" << endl;

    cout << "processing images from " << firstidx << " to " << lastidx << endl;

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      
//       QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
// 	padZeros(QString::number(imgidx), 4);


      QString qsPrecomputedSamplesDir = get_samples_dir(part_app.m_exp_param, imgidx);

      QString qsResultsDir = get_samples_post_dir(part_app.m_exp_param);

      assert(!qsResultsDir.isEmpty());

      if (!filesys::check_dir(qsResultsDir)) {
	cout << "creating "  << qsResultsDir.toStdString() << endl;
	assert(filesys::create_dir(qsResultsDir));
      }

      /** 
	  load samples
      */
      
      int nParts = part_app.m_part_conf.part_size();
      vector<uint> vNumSamples(nParts, 0);

      /** MA: we need this because of the awkward implementation that stores part hypothesis as bounding boxes 
	  and part scale is needed when computing detection scores and spatial factors
       */
//       vector<vector<double> > part_vect_scales;
//       vector<vector<PartBBox> > part_hyp(nParts, vector<PartBBox>());

      
      uint nSubjects;

      if (part_app.m_exp_param.dai_multiperson())
	nSubjects = 2;
      else
	nSubjects = 1;

      /** 
	  dimensions: subject, part, samples 
      */
      
      vector<vector<vector<double> > > mp_part_vect_scales(nSubjects, vector<vector<double> >(nParts, vector<double>()));
      
      vector<vector<vector<PartBBox> > >  mp_part_hyp(nSubjects, vector<vector<PartBBox> >(nParts, vector<PartBBox>()));

      /** these are not always available */
      vector<vector<vector<double> > > mp_part_scores(nSubjects, vector<vector<double> >(nParts, vector<double>()));

      /** index of sample in the original array (useful when saving posteriors computed independently for each subject) */
      vector<vector<vector<int> > > mp_part_sample_idx(nSubjects, vector<vector<int> >(nParts, vector<int>()));

      /** 
	  we can do inference with multiperson model even when sampling without bounding box prior
       */

      bool bHaveMultipersonDetections = false;

      for (int pidx = 0; pidx < nParts; ++pidx) {

	QString qsInputFilename = get_samples_filename(part_app.m_exp_param, imgidx, pidx);
	assert(filesys::check_file(qsInputFilename));

	vector<int> vect_scale_idx;
	vector<int> vect_rot_idx; 
	vector<int> vect_iy; 
        vector<int> vect_ix;
	vector<int> vect_didx;

	vector<double> vect_scores;
	
	cout << "loading samples from " << qsInputFilename.toStdString() << endl;

	bool res5 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scale_idx", vect_scale_idx);
	bool res1 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_rot_idx", vect_rot_idx);
        bool res2 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_iy", vect_iy);
        bool res3 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_ix", vect_ix);

        assert(res5 && res1 && res2 && res3);

	if (part_app.m_exp_param.dai_bbox_prior()) {
	  bool res6 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_didx", vect_didx);
	  bHaveMultipersonDetections = true;
	}
	
	//bHaveMultipersonDetections = res6;

	//if (part_app.m_exp_param.dai_bbox_prior())
	//  assert(bHaveMultipersonDetections);
	//else
	//  assert(!bHaveMultipersonDetections);

	vNumSamples[pidx] = vect_scale_idx.size();
	assert(vNumSamples[pidx] > 0);

	assert(vNumSamples[pidx] == vect_rot_idx.size() && 
	       vNumSamples[pidx] == vect_iy.size() && 
	       vNumSamples[pidx] == vect_ix.size());

	/**  */
	/* leonid 14.5.2012: always load the scores from the scoregrid to avoid differences in 
	 values due to recomputation */
	bool res4 = false;
	//if (part_app.m_exp_param.dai_samples_type() == "part_det") {
	  res4 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scores", vect_scores);
	  assert(res4);
	  //}

	if (vect_scores.size() > 0) 
	  assert(vNumSamples[pidx] == vect_scores.size());

	if (vect_didx.size() > 0) 
	  assert(vNumSamples[pidx] == vect_didx.size());

	/** assume single scale */
	//assert(part_app.m_exp_param.num_scale_steps() == 1);
	//int scaleidx = 0;
	
	
	/* leonid DEBUG */
	if (pidx == part_app.m_rootpart_idx && part_app.m_exp_param.has_torso_hyp_idx()){
	  QString qsFilename = QString(part_app.m_exp_param.part_marginals_dir().c_str()) + "/torso_hyp_imgidx" + padZeros(QString::number(imgidx), 4) + ".mat";
	  FloatGrid2 best_conf = matlab_io::mat_load_multi_array<FloatGrid2>(qsFilename, "torso_hyp");
	  object_detect::PartHyp hyp; 
	  hyp.fromVect(best_conf[boost::indices[0][index_range()]]);
	  vect_rot_idx[0] = hyp.m_rotidx;
	  vect_iy[0] = hyp.m_y;
	  vect_ix[0] = hyp.m_x;
	}
	/* leonid DEBUG */
	
	
	vector<PartBBox> part_hyp;
        samples_to_bbox(part_app, pidx,  
                        vect_scale_idx, vect_rot_idx, vect_iy, vect_ix, part_hyp);

	vector<double> vect_scales;
	for (uint idx = 0; idx < vNumSamples[pidx]; ++idx)
	  vect_scales.push_back(scale_from_index(part_app.m_exp_param, vect_scale_idx[idx]));

	if (!part_app.m_exp_param.dai_multiperson())
	  assert(nSubjects == 1);

	if (vect_didx.size() == 0 || !part_app.m_exp_param.dai_multiperson()) {	  
	  /** 
	      no subject index or single person version -> assign copies of all samples to every subject 
	  */
	  
	  assert(nSubjects == 1); // MA: otherwise how do we save posteriors in the current format ?
	  
	  mp_part_hyp[0][pidx] = part_hyp;
	  mp_part_vect_scales[0][pidx] = vect_scales;
	  mp_part_scores[0][pidx] = vect_scores;
	  for (uint idx = 0; idx < vNumSamples[pidx]; ++idx) {
	    mp_part_sample_idx[0][pidx].push_back(idx);
	  }


// 	  for (uint didx = 0; didx < nSubjects; ++didx) {
// 	    mp_part_hyp[didx][pidx] = part_hyp;
// 	    mp_part_vect_scales[didx][pidx] = vect_scales;
// 	    mp_part_scores[didx][pidx] = vect_scores;
// 	  }


	}
	else {
	  /** split samples according to subject */

	  assert(part_hyp.size() == vNumSamples[pidx]);
	  
	  for (uint idx = 0; idx < vNumSamples[pidx]; ++idx) {
	    uint didx = vect_didx[idx];
	    assert(didx < nSubjects);

	    mp_part_hyp[didx][pidx].push_back(part_hyp[idx]);
	    mp_part_vect_scales[didx][pidx].push_back(vect_scales[idx]);

	    /** save index in the original array */
	    mp_part_sample_idx[didx][pidx].push_back(idx);

	    if (vect_scores.size() > 0)
	      mp_part_scores[didx][pidx].push_back(vect_scores[idx]);
	  }
	}

	//part_vect_scales.push_back(vect_scales);
      }// parts 
      

      /**
	 compute factors
      */

      QString qsFactorsDir = get_factors_dir(part_app.m_exp_param, imgidx);
      if (!filesys::check_dir(qsFactorsDir)) 
	assert(filesys::create_dir(qsFactorsDir));

      vector<Factor1d> factors1d;
      vector<Factor2d> factors2d;

      vector<Joint> joints;
      bool flip = false;
      object_detect::loadJoints(part_app, joints, flip);

      /** factors in addition to standard pictorial structures */
      if (part_app.m_exp_param.has_dai_factors()) {
	FactorDefs factor_defs;
	cout << "loading factor definitions from: " << part_app.m_exp_param.dai_factors() << endl;
	parse_message_from_text_file(QString::fromStdString(part_app.m_exp_param.dai_factors()), factor_defs);

	cout << "\tnumber of repulsive factors: " << factor_defs.repulsive_factor_size() << endl;
	cout << "\tnumber of attractive factors: " << factor_defs.attractive_factor_size() << endl;

	/**
	   repulsive factors
	 */
	for (uint didx = 0; didx < nSubjects; ++didx) {

	  for (int idx = 0; idx < factor_defs.repulsive_factor_size(); ++idx) {
	    int rep_pidx1 = factor_defs.repulsive_factor(idx).pidx1();
	    int rep_pidx2 = factor_defs.repulsive_factor(idx).pidx2();

	    assert(rep_pidx1 < nParts && rep_pidx2 < nParts);

	    Factor2d repulsive_factor;
  
	    /** 
		assume that bounding boxes are already at the right scale 
	    */
	    compute_repulsive_factor(mp_part_hyp[didx][rep_pidx1], mp_part_hyp[didx][rep_pidx2], 
				     repulsive_factor, 
				     factor_defs.repulsive_factor(idx).min_relative_area(), 
				     factor_defs.repulsive_factor(idx).alpha());

	    repulsive_factor.varidx1 = nidx_from_pidx(nParts, didx, rep_pidx1);
	    repulsive_factor.varidx2 = nidx_from_pidx(nParts, didx, rep_pidx2);

	    factors2d.push_back(repulsive_factor);

	  }// repulsive factors	

	} // subjects

	/** 
	    attractive factors
	 */
	for (int idx = 0; idx < factor_defs.attractive_factor_size(); ++idx) {

	  int didx1 = factor_defs.attractive_factor(idx).didx1();
	  int didx2 = factor_defs.attractive_factor(idx).didx2();

	  int pidx1 = factor_defs.attractive_factor(idx).pidx1();
	  int pidx2 = factor_defs.attractive_factor(idx).pidx2();

	  cout << "attractive factor between parts: " << pidx1 << " and " << pidx2 << endl;

	  assert(didx1 < (int)nSubjects && didx2 < (int)nSubjects && pidx1 < nParts && pidx2 < nParts);
	    
	  double_vector mu(2);
	  mu(0) = factor_defs.attractive_factor(idx).mu_x();
	  mu(1) = factor_defs.attractive_factor(idx).mu_y();

	  double_vector sigma(2);
	  sigma(0) = factor_defs.attractive_factor(idx).sigma_x();
	  sigma(1) = factor_defs.attractive_factor(idx).sigma_y();

	  Factor2d attractive_factor;

	  compute_attractive_factor(mp_part_hyp[didx1][pidx1], mp_part_hyp[didx2][pidx2], 
				    mp_part_vect_scales[didx1][pidx1], mp_part_vect_scales[didx2][pidx2],
				    attractive_factor, mu, sigma);

	  attractive_factor.varidx1 = nidx_from_pidx(nParts, didx1, pidx1);
	  attractive_factor.varidx2 = nidx_from_pidx(nParts, didx2, pidx2);

	  factors2d.push_back(attractive_factor);	    
	}

      }


      /** Abc factor */
      kma::ImageContent *kma_input_image = 0;

      /** test if we can load the abc factor instead of computing it */
      bool bRecompute = true;

      if (bForceRecompute == false) {
	uint pidx = 0;
	uint factidx = 0;
	if ( filesys::check_file(get_samples_filename(part_app.m_exp_param, imgidx, pidx)) && 
	     filesys::check_file(get_factor_filename(part_app.m_exp_param, imgidx, "abc", factidx))) {

	  QFileInfo fi1(get_samples_filename(part_app.m_exp_param, imgidx, pidx));
	  QFileInfo fi2(get_factor_filename(part_app.m_exp_param, imgidx, "abc", factidx));

	  QDateTime dt1 = fi1.lastModified();
	  QDateTime dt2 = fi2.lastModified();

	  if (dt1 < dt2) 
	    bRecompute = false;
	}
      }

      /** assume that scores are available for all parts or for none of them */
      assert(mp_part_scores.size() > 0);
      assert(mp_part_scores[0].size() > 0);

      //if (mp_part_scores[0][0].size() == 0 && bRecompute) 
      if (bRecompute) 	
	kma_input_image = kma::load_convert_gray_image(part_app.m_test_annolist[imgidx].imageName().c_str());

      for (uint didx = 0; didx < nSubjects; ++didx) {

	/** begin debug */
// 	for (int pidx = 0; pidx < nParts; ++pidx) {


// 	  if (didx == 1) {
// 	    for (uint idx = 0; idx < mp_part_vect_scales[didx][pidx].size(); ++idx) {
// 	      mp_part_vect_scales[didx][pidx][idx] = 0.7;
// 	    }
// 	  }

// 	  cout << endl << endl;
// 	  cout << "didx: " << didx << ", scale: " << mp_part_vect_scales[didx][pidx][0] << endl;
// 	  cout << endl << endl;
// 	}

	/** end debug */

	for (int pidx = 0; pidx < nParts; ++pidx) {	  

	  uint nidx = nidx_from_pidx(nParts, didx, pidx);

	  Factor1d boosting_factor;
	  //QString qsFactorFilename = get_factor_filename(part_app.m_exp_param, imgidx, "abc", factors1d.size());
	  QString qsFactorFilename = get_factor_filename(part_app.m_exp_param, imgidx, "abc", nidx);

	  if (bRecompute) {
	    /**
	       some parts don't have the part detector and are there to conveniently formulate the spatial model
	    */
	    if (part_app.m_part_conf.part(pidx).is_detect()) {
	      cout << "computing unary factor, number of samples: " << mp_part_hyp[didx][pidx].size() << endl;

	      /**
		 if sample from posteriors we recompute the score, if sample from test_scoregrid the scores are reused 
	      */
	      if (mp_part_scores[didx][pidx].size() == 0) {
		assert(kma_input_image != 0);

		/** assume single scale, not anymore :)  */
		//assert(part_app.m_exp_param.num_scale_steps() == 1);
		//int scaleidx = 0;
		//compute_boosting_score_factor(part_app, pidx, scaleidx, kma_input_image, part_hyp[pidx], boosting_factor);

		//compute_boosting_score_factor(part_app, pidx, kma_input_image, part_hyp[pidx], part_vect_scales[pidx], boosting_factor);

		compute_boosting_score_factor(part_app, pidx, kma_input_image, mp_part_hyp[didx][pidx], mp_part_vect_scales[didx][pidx], boosting_factor);
		
		/* leonid DEBUG */
		if (part_app.m_exp_param.has_torso_hyp_idx())
		  for (uint idx = 1; idx < mp_part_scores[didx][pidx].size(); ++idx) 
		    boosting_factor.fval[idx] = 0.0;
		/* leonid DEBUG */
				
		// add DPM score
		if (part_app.m_exp_param.has_test_dpm_torso_dir() && pidx == part_app.m_rootpart_idx){
		  addDPMscore(part_app, imgidx, kma_input_image->x(), kma_input_image->y(), 
			      pidx, mp_part_hyp[didx][pidx].size(), boosting_factor);
		}

		/** */
		boosting_factor.varidx = nidx;
	      }
	      else {
		//boosting_factor.varidx = pidx;
		boosting_factor.varidx = nidx;
		boosting_factor.fval.resize(boost::extents[mp_part_scores[didx][pidx].size()]);

		for (uint idx = 0; idx < mp_part_scores[didx][pidx].size(); ++idx) 
		  boosting_factor.fval[idx] = mp_part_scores[didx][pidx][idx];

		/* leonid DEBUG */
		if (part_app.m_exp_param.has_torso_hyp_idx())
		  for (uint idx = 1; idx < mp_part_scores[didx][pidx].size(); ++idx) 
		    boosting_factor.fval[idx] = 0.0;
		/* leonid DEBUG */
		
		//add DPM score
		if (part_app.m_exp_param.has_test_dpm_torso_dir() && pidx == part_app.m_rootpart_idx){
		  addDPMscore(part_app, imgidx, kma_input_image->x(), kma_input_image->y(), 
			      pidx, mp_part_hyp[didx][pidx].size(), boosting_factor);
		}

	      }
	    }
	    else {
	      /* use uniform score over all positions */
	      //boosting_factor.varidx = pidx;

	      boosting_factor.varidx = nidx;
	      boosting_factor.fval.resize(boost::extents[mp_part_hyp[didx][pidx].size()]);

	      for (uint idx = 0; idx < mp_part_hyp[didx][pidx].size(); ++idx) 
		boosting_factor.fval[idx] = 1.0 / mp_part_hyp[didx][pidx].size();
	  
	    }

	    /** save factor to disc */
	    assert(part_app.m_bExternalSamplesDir == false && "external samples dir");
	    cout << "saving " << qsFactorFilename.toStdString() << endl;
	    boosting_factor.save_factor(qsFactorFilename);

	  }// if bRecompute
	  else {
	    cout << "loading " << qsFactorFilename.toStdString() << endl;
	    boosting_factor.load_factor(qsFactorFilename);
	  }

	  /** order is not important, the node id is set in varidx */
	  factors1d.push_back(boosting_factor);

	  //boosting_factor.dai_print(cout);
	}// parts
      }// subjects

      if (kma_input_image != 0) {
	delete kma_input_image;
	kma_input_image = 0;
      }

      /** DEBUG */
      for (uint didx = 0; didx < nSubjects; ++didx) 
	for (int pidx = 0; pidx < nParts; ++pidx)
	  cout << "didx: " << didx << ", scale: " << mp_part_vect_scales[didx][pidx][0] << endl;
      /** DEBUG - end */
    
      /** spatial factor */
      cout << "computing spatial factors" << endl;

      uint spatial_factidx = 0;

      for (uint didx = 0; didx < nSubjects; ++didx) {

	for (uint jidx = 0; jidx < joints.size(); ++jidx) {

	  Factor2d spatial_factor;

	  QString qsFactorFilename = get_factor_filename(part_app.m_exp_param, imgidx, "spatial", spatial_factidx);

	  /** DEBUG -> always recompute spatial factors */
	  //if (true) {
	  if (bRecompute) {

	    uint child_pidx = joints[jidx].child_idx;
	    uint parent_pidx = joints[jidx].parent_idx;

	    compute_spatial_factor(part_app, joints[jidx], 
				   mp_part_hyp[didx][child_pidx], mp_part_hyp[didx][parent_pidx], 
				   mp_part_vect_scales[didx][child_pidx], mp_part_vect_scales[didx][parent_pidx], 
				   spatial_factor);

	    /** in case of multiple subjects "compute_spatial_factor" sets node indices incorrectly (it takes them from the joints structure)*/
	    spatial_factor.varidx1 = nidx_from_pidx(nParts, didx, child_pidx);
	    spatial_factor.varidx2 = nidx_from_pidx(nParts, didx, parent_pidx);
	    
	    assert(part_app.m_bExternalSamplesDir == false && "external samples dir");
	    cout << "saving " << qsFactorFilename.toStdString() << endl;
	    spatial_factor.save_factor(qsFactorFilename);
	  }
	  else {
	    cout << "loading " << qsFactorFilename.toStdString() << endl;
	    spatial_factor.load_factor(qsFactorFilename);
	  }
      
	  factors2d.push_back(spatial_factor);

	  ++spatial_factidx;
	}
      }

      /** some checking */
      cout << "number of unary factors: " << factors1d.size() << endl;

      for (uint idx = 0; idx < factors1d.size(); ++idx) {
	//int pidx = factors1d[idx].varidx;

	uint pidx = pidx_from_nidx(nParts, factors1d[idx].varidx);
	uint didx = didx_from_nidx(nParts, factors1d[idx].varidx);

	assert(pidx < (uint)nParts);
	assert(didx < nSubjects);

	//assert(factors1d[idx].fval.shape()[0] == part_hyp[pidx].size());
	assert(factors1d[idx].fval.shape()[0] == mp_part_hyp[didx][pidx].size());
      }

      cout << "number of pairwise factors: " << factors2d.size() << endl;

      for (uint idx = 0; idx < factors2d.size(); ++idx) {
	uint pidx1 = pidx_from_nidx(nParts, factors2d[idx].varidx1);
	uint pidx2 = pidx_from_nidx(nParts, factors2d[idx].varidx2);

	uint didx1 = didx_from_nidx(nParts, factors2d[idx].varidx1);
	uint didx2 = didx_from_nidx(nParts, factors2d[idx].varidx2);

	assert(factors2d[idx].fval.shape()[0] == mp_part_hyp[didx1][pidx1].size());
	assert(factors2d[idx].fval.shape()[1] == mp_part_hyp[didx2][pidx2].size());

	cout << "factor: " << didx1 << "/" << pidx1 << " - " << didx2 << "/" << pidx2 << endl;
      }

      assert(factors1d.size() > 0 || factors2d.size() > 0);

      /** 
	  DAI inference 
      */

//       stringstream str_stream;
//       dai_print_stream(str_stream, factors1d, factors2d);

//       dai::FactorGraph fg;
//       str_stream >> fg;

      
      vector<dai::Var> dai_vars;
      vector<dai::Factor> dai_factors;

      for (uint didx = 0; didx < nSubjects; ++didx)
	for (uint pidx = 0; pidx < (uint)nParts; ++pidx) {
	  uint nidx = nidx_from_pidx(nParts, didx, pidx);
	  dai_vars.push_back(dai::Var(nidx, mp_part_hyp[didx][pidx].size()));
	}

      for (uint idx = 0; idx < factors1d.size(); ++idx) {
	uint nidx = factors1d[idx].varidx;
	assert(dai_vars.size() > nidx);

	dai::Factor f( dai_vars[nidx] );
	
	for (uint idx2 = 0; idx2 < factors1d[idx].fval.shape()[0]; ++idx2) {
	  //f.set(idx2, factors1d[idx].fval[idx2]);
	  f[idx2] = factors1d[idx].fval[idx2];
	}

	dai_factors.push_back(f);	
      }

      for (uint idx = 0; idx < factors2d.size(); ++idx) {
	uint nidx1 = factors2d[idx].varidx1;
	uint nidx2 = factors2d[idx].varidx2;

	assert(dai_vars.size() > nidx1 && dai_vars.size() > nidx2);
	
	dai::Factor f2( dai::VarSet(dai_vars[nidx1], dai_vars[nidx2]) );

	int idx0 = 0;

	/** MA: for pairwise factors one should first add rows than columns, 

	    note: when factor is defined between variables with indices I1 and I2, it is assumed 
	    that I1 < I2, otherwise the order in which we add values is reversed
	 */
	

	if (nidx1 > nidx2) {
	  for (uint idx1 = 0; idx1 < factors2d[idx].fval.shape()[0]; ++idx1) {
	    for (uint idx2 = 0; idx2 < factors2d[idx].fval.shape()[1]; ++idx2) {
	      //f2.set(idx0, factors2d[idx].fval[idx1][idx2]);
	      f2[idx0] = factors2d[idx].fval[idx1][idx2];
	      ++idx0;
	    }
	  }
	}
	else {
	  for (uint idx2 = 0; idx2 < factors2d[idx].fval.shape()[1]; ++idx2) {
	    for (uint idx1 = 0; idx1 < factors2d[idx].fval.shape()[0]; ++idx1) {
	      //f2.set(idx0, factors2d[idx].fval[idx1][idx2]);
	      f2[idx0] = factors2d[idx].fval[idx1][idx2];
	      ++idx0;
	    }
	  }
	}

	dai_factors.push_back(f2);
      }
      
      dai::FactorGraph fg(dai_factors);

      size_t  maxiter = 500;
      //double  tol = 1e-9;
      double  tol = 1e-20;
      size_t  verb = 10;

      dai::PropertySet opts;
      opts.Set("maxiter",maxiter);
      opts.Set("tol",tol);
      opts.Set("verbose",verb);

      string szInference;
      if (part_app.m_exp_param.dai_bp_type() == "sumprod")
	szInference = "SUMPROD";
      else if (part_app.m_exp_param.dai_bp_type() == "maxprod")
	szInference = "MAXPROD";
      else
	assert(false && "unknown BP type");

      dai::BP bp(fg, opts("updates",string("SEQFIX"))("logdomain",false)("inference",szInference));

      bp.init();
      bp.run();


      /** 
	  save results 
      */

      vector<vector<double_vector> > mp_part_posterior(nSubjects, vector<double_vector>(nParts, double_vector()));

      for (size_t nidx = 0; nidx < fg.nrVars(); ++nidx) {
	dai::Factor cur_belief = bp.belief(fg.var(nidx));

	int nStates = cur_belief.states();

	boost_math::double_vector cur_posterior(nStates);
	for (int sidx = 0; sidx < nStates; ++sidx) {
	  cur_posterior[sidx] = cur_belief[sidx];
	}

	//part_posterior.push_back(cur_posterior);

	uint pidx = pidx_from_nidx(nParts, nidx);
	uint didx = didx_from_nidx(nParts, nidx);

	assert(nStates == (int)mp_part_hyp[didx][pidx].size());

	assert(pidx < (uint)nParts);
	assert(didx < nSubjects);

	mp_part_posterior[didx][pidx] = cur_posterior;

	cout << "nidx: " << nidx << ", subject: " << didx << ", part: " << pidx << endl;
	
	//if (pidx == 5) {
	  double maxval;
	  int maxidx;
	  boost_math::get_max(cur_posterior, maxval, maxidx);
	  cout << "\t " << mp_part_hyp[didx][pidx][maxidx].part_pos(0) << " " << mp_part_hyp[didx][pidx][maxidx].part_pos(1) << " " << maxval << endl;
	  //}
	

	  double Z = 0;
	  for (uint idx = 0; idx < cur_posterior.size(); ++idx)
	    Z += cur_posterior(idx);

	  cout << "\t number of states: " << nStates << endl;
	  cout << "\t Z: " << Z << endl;

      }
            /**
	 compute score of the best configuration
       */
      vector<int> best_conf(nParts);
      int didx = 0;
      assert(nSubjects == 1);

      for (int pidx = 0; pidx < nParts; ++pidx) {
	double maxval;
	int maxidx;
	boost_math::get_max(mp_part_posterior[didx][pidx], maxval, maxidx);

	assert(factors1d[pidx].varidx == pidx);

	cout << "\t part " << pidx << ":" << 
	  mp_part_hyp[didx][pidx][maxidx].part_pos(0) << " " << mp_part_hyp[didx][pidx][maxidx].part_pos(1) << 
	  ", boosting score: " << factors1d[pidx].fval[maxidx] << std::endl;
	  
	best_conf[pidx] = maxidx;
	
	/* DEBUG */
	/*
	uint ridx = part_hyp_list[pidx].m_rotidx;
	uint iy = part_hyp_list[pidx].m_y;
	uint ix = part_hyp_list[pidx].m_x;
	vector<object_detect::PartHyp> part_hyp_list;
	QString qsHypDir = "/scratch/BS/pool0/leonid/log_dir/exp_ramanan_075-1000-map-fix-inf-substr-mean-no-wraparound-factors/part_marginals/";
	loadPartHyp(part_app, qsHypDir, imgidx, part_hyp_list);
	*/
	//assert(mp_part_hyp[didx][pidx][maxidx].part_pos(0) == ix && 
	//       mp_part_hyp[didx][pidx][maxidx].part_pos(1) == iy);// &&
	//rot_idx == ridx);
	/* DEBUG */
	
      }

      double L = 0;
      double L_p = 0;
      for (uint idx = 0; idx < factors2d.size(); ++idx) {
	uint pidx1 = pidx_from_nidx(nParts, factors2d[idx].varidx1);
	uint pidx2 = pidx_from_nidx(nParts, factors2d[idx].varidx2);

	double fval = factors2d[idx].fval[best_conf[pidx1]][best_conf[pidx2]];
	L += log(fval); 
	L_p += log(fval); 
	cout << "factor: " << pidx1 << " - " << pidx2 << ", " << fval << std::endl;
	//cout << "factor: " << pidx1 << " - " << pidx2 << ", " << log(fval) << std::endl;
      }
      double L_u = 0;
      for(uint idx = 0; idx < factors1d.size(); ++idx) {
	uint pidx = pidx_from_nidx(nParts, factors1d[idx].varidx);

	double fval = factors1d[idx].fval[best_conf[pidx]];
	L += log(fval); 
	L_u += log(fval); 
	cout << "factor: " << pidx <<  ", " << fval << std::endl;
	//cout << "factor: " << pidx <<  ", " << log(fval) << std::endl;
      }
      
      cout << "L: " << L << endl;
      cout << "L_u: " << L_u << endl;
      cout << "L_p: " << L_p << endl;
      
      //QString qsResultsFilename = qsResultsDir + "/samples_imgidx" + padZeros(QString::number(imgidx), 4) + "_post.mat";
      QString qsResultsFilename = get_samples_post_filename(part_app.m_exp_param, imgidx);

      cout << "saving results to " << qsResultsFilename.toStdString() << endl;

      MATFile *f = matlab_io::mat_open(qsResultsFilename, "wz");
      assert(f != 0);

      for (int pidx = 0; pidx < nParts; ++pidx) {
	QString qsVarName = "samples_post_part" + QString::number(pidx);

	/** concatenate posteriors over all subjects (they should be in the same order as before) */
	double_vector _part_posterior(vNumSamples[pidx]);

	for (uint idx = 0; idx < _part_posterior.size(); ++idx)
	  _part_posterior(idx) = -123;
	
	for (int didx = nSubjects - 1; didx >= 0; --didx) {

	  for (uint idx = 0; idx < mp_part_posterior[didx][pidx].size(); ++idx) {
	    assert(idx < mp_part_sample_idx[didx][pidx].size());

	    uint _idx = mp_part_sample_idx[didx][pidx][idx];
	    assert(_idx < _part_posterior.size());
	    _part_posterior(_idx) = mp_part_posterior[didx][pidx][idx];

	  }
	}

	for (uint idx = 0; idx < _part_posterior.size(); ++idx)
	  assert(_part_posterior(idx) >= 0);

	matlab_io::mat_save_double_vector(f, qsVarName, _part_posterior);
      }

      /** save score of the best configuration */
      matlab_io::mat_save_double(f, "L", L);
      
      if (f != 0)
	matlab_io::mat_close(f);
    } // images

  }


  void visSamples(const PartApp &part_app, int firstidx, int lastidx)
  {
    cout << "processing images from " << firstidx << " to " << lastidx << endl;

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      //QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
      //padZeros(QString::number(imgidx), 4);

      QString qsResultsDir;

      if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) {
	qsResultsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
					      "/part_marginals_samples_vis");
      }
      else if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
	qsResultsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
					      "/test_scoregrid_samples_vis");
      }
      else {
	assert(false && "unknown part samples type");
      }

      qsResultsDir += "/" + padZeros(QString::number(imgidx), 4);

      if (!filesys::check_dir(qsResultsDir))
	assert(filesys::create_dir(qsResultsDir));

      /** 
	  load samples
      */
      
      int nParts = part_app.m_part_conf.part_size();
      vector<vector<PartBBox> > part_hyp(nParts, vector<PartBBox>());

      /** these are not always available */
      vector<vector<double> > part_scores(nParts, vector<double>());

      QImage img;
      assert(img.load(part_app.m_test_annolist[imgidx].imageName().c_str()));

      for (int pidx = 0; pidx < nParts; ++pidx) {
	//QString qsInputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";
	
	QString qsInputFilename = get_samples_filename(part_app.m_exp_param, imgidx, pidx);
	
	assert(filesys::check_file(qsInputFilename));

	vector<int> vect_scale_idx; 
	vector<int> vect_rot_idx; 
	vector<int> vect_iy; 
        vector<int> vect_ix;
	vector<double> vect_scores;
	vector<int> vect_didx;

	cout << "loading samples from " << qsInputFilename.toStdString() << endl;

	bool res5 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scale_idx", vect_scale_idx);
	bool res1 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_rot_idx", vect_rot_idx);
        bool res2 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_iy", vect_iy);
        bool res3 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_ix", vect_ix);

	bool res6 = false;

	if (part_app.m_exp_param.dai_bbox_prior())
	  res6 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_didx", vect_didx);
	
	/** when using detections directly we also have access to their scores, 
	    when using marinals we recompute the scores 
	 */
	bool res4 = false;
	if (part_app.m_exp_param.dai_samples_type() == "part_det") {
	  res4 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scores", vect_scores);
	  assert(res4);
	  assert(vect_scores.size() > 0);
	}

        assert(res5 && res1 && res2 && res3);

	if (res4) {
	  part_scores[pidx] = vect_scores;
	}

	/** assume single scale */
	//assert(part_app.m_exp_param.num_scale_steps() == 1);
	//int scaleidx = 0;

//         samples_to_bbox(part_app, pidx, scaleidx, 
//                         vect_rot_idx, vect_iy, vect_ix, part_hyp[pidx]);

	samples_to_bbox(part_app, pidx, 
			vect_scale_idx, vect_rot_idx, vect_iy, vect_ix, part_hyp[pidx]);

	QImage _img = img.convertToFormat(QImage::Format_RGB32);
	QPainter painter(&_img);
	painter.setRenderHints(QPainter::Antialiasing);
	    
	for (uint sidx = 0; sidx < part_hyp[pidx].size(); ++sidx) {
	  int pen_width = 2;
	  int coloridx = -1;

	  if (res6)
	    coloridx = -vect_didx[sidx] - 1;

	  draw_bbox(painter, part_hyp[pidx][sidx], coloridx, pen_width);
	}
	
	QString qsResultsFile = qsResultsDir + "/samples_pidx" + padZeros(QString::number(pidx), 2) + ".jpg";

	cout << "saving " << qsResultsFile.toStdString() << endl;
	assert(_img.save(qsResultsFile));
	  
      }// parts 
      


    }// images
	  

  }


}// namespace 
