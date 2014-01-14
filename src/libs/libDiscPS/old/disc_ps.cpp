#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <dai/alldai.h>

#include <libMultiArray/multi_array_op.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libMatlabIO/matlab_io.hpp>

#include <libPartApp/partapp_aux.hpp>

#include "disc_ps.h"
#include "unique_vect.hpp"
#include "factors.h"

#include "disc_sample.hpp"


using namespace std;
using namespace boost::lambda;
using boost::multi_array_types::index_range;

namespace disc_ps {

  using boost_math::double_vector;
  using object_detect::Joint;

  void partSample(const PartApp &part_app, int firstidx, int lastidx)
  {
    cout << "disc_ps::postSample" << endl << endl;

    /** make sure we don't overwrite samples from other experiment */
    assert(!part_app.m_bExternalSamplesDir && 
	   "using external samples dir, new samples should be generated using original experiment file");

    int nParts = part_app.m_part_conf.part_size();

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {

      QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
	padZeros(QString::number(imgidx), 4);
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


	/** extract samples either from posteriors or from part detections */

	if (part_app.m_exp_param.dai_samples_type() == "part_det") {
	  assert(part_app.m_exp_param.num_scale_steps() == 1);

	  int scaleidx = 0;
	  bool bInterpolate = false;

	  std::vector<std::vector<FloatGrid2> > score_grid;
	  part_app.loadScoreGrid(score_grid, imgidx, pidx, flip, bInterpolate);

	  assert(score_grid.size() == 1);
	  assert(score_grid[0].size() == part_app.m_exp_param.num_rotation_steps());

	  double det_score_threshold = 0.0;

	  for (uint ridx = 0; ridx < score_grid[0].size(); ++ridx) {
	    uint img_height = score_grid[0][ridx].shape()[0];
	    uint img_width = score_grid[0][ridx].shape()[1];

	    for (uint ix = 0; ix < img_width; ++ix) 
	      for (uint iy = 0; iy < img_height; ++iy) {
		if (score_grid[0][ridx][iy][ix] > det_score_threshold) {
		  vect_rot_idx.push_back(ridx);
		  vect_iy.push_back(iy);
		  vect_ix.push_back(ix);
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
	    vector<double> _vect_scores;

	    for (int idx = 0; idx < part_app.m_exp_param.dai_num_samples(); ++idx) {
	      int idx2 = all_scores[idx].second;
	      
	      _vect_rot_idx.push_back(vect_rot_idx[idx2]);
	      _vect_iy.push_back(vect_iy[idx2]);
	      _vect_ix.push_back(vect_ix[idx2]);
	      _vect_scores.push_back(vect_scores[idx2]);
	    }

	    
	    vect_rot_idx = _vect_rot_idx;
	    vect_ix = _vect_ix;
	    vect_iy = _vect_iy;
	    vect_scores = _vect_scores;
	  }
	  
	  assert(vect_rot_idx.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_ix.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_iy.size() <= (uint)part_app.m_exp_param.dai_num_samples());
	  assert(vect_scores.size() <= (uint)part_app.m_exp_param.dai_num_samples());

	  assert(vect_ix.size() == vect_rot_idx.size() && 
		 vect_ix.size() == vect_iy.size() && 
		 vect_ix.size() == vect_scores.size());


	}
	else if (part_app.m_exp_param.dai_samples_type() == "part_post") {
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

	  /** BEGIN debug */

	  /** 

	      ver2

	   */

//  	  double scale_sigma = 0.05;
//  	  double rot_sigma = 7.5;
//  	  double pos_sigma = 5;

//  	  double scale_step_size = (part_app.m_exp_param.max_object_scale() - part_app.m_exp_param.min_object_scale()) / part_app.m_exp_param.num_scale_steps();
// 	  double scale_sigma_idx = scale_sigma / scale_step_size;

// 	  double rot_step_size = (exp_param.max_part_rotation() - exp_param.min_part_rotation())/exp_param.num_rotation_steps();
// 	  double rot_sigma_idx = rot_sigma / rot_step_size;

// 	  int num_scales = part_posterior.shape()[0];
// 	  int num_rot = part_posterior.shape()[1];
// 	  int num_y = part_posterior.shape()[2];
// 	  int num_y = part_posterior.shape()[3];

//  	  for (uint sampidx = 0; sampidx < idxmat.size1(); ++sampidx) {
// 	    sample_sidx = idxmat(sampidx, 0);
// 	    sample_ridx = idxmat(sampidx, 1);

// 	    sample_y = idxmat(sampidx, 2);
// 	    sample_x = idxmat(sampidx, 3);
	    
// 	    int sidx1 = std::max(round(sample_sidx - 3*scale_sigma_idx), 0);
// 	    int sidx2 = std::min(round(sample_sidx + 3*scale_sigma_idx), num_scales - 1);

// 	    double invZ = 1.0 / (square(2*M_PI)*square(pos_sigma)*rot_sigma_idx*scale_sigma_idx);

// 	    for (int sidx = sidx1; sidx <= sidx2; ++sidx) {
	      
// 	      int ridx1 = round(sample_ridx - 3*rot_sigma_idx);
// 	      int ridx2 = round(sample_ridx + 3*rot_sigma_idx);

// 	      for (int _ridx = ridx1; _ridx <= ridx2; ++_ridx) {

// 		int ridx;

// 		if (_ridx >= 0 && _ridx < num_rot) 
// 		  ridx = _ridx;
// 		else if (_ridx < 0) 
// 		  ridx = num_rot + _ridx;		    
// 		else 
// 		  ridx = _ridx - num_rot;

// 		int yidx1 = max(round(sample_y - 3*pos_sigma), 0);
// 		int yidx2 = min(round(sample_y + 3*pos_sigma), num_y - 1);

// 		for (int yidx = yidx1; yidx <= yidx2; ++yidx) {

// 		  int xidx1 = max(round(sample_x - 3*pos_sigma), 0);
// 		  int xidx2 = max(round(sample_x + 3*pos_sigma), num_x - 1);

// 		  for (int xidx = xidx1; xidx < xidx2; ++xidx) {

// 		    dx = (xidx - sample_x) / pos_sigma;
// 		    dy = (yidx - sample_y) / pos_sigma;

// 		    // _ridx is before wrap-around
// 		    dr = (_ridx - sample_ridx) / rot_sigma_idx;

// 		    ds = (sidx - sample_sidx) / scale_sigma_idx;

// 		    double w = invZ * exp(-0.5*(square(dx) + square(dy) + square(dr) + square(ds)));
		    
// 		    part_posterior[sidx][ridx][yidx][xidx] = part_posterior
		    
// 		  }// x
// 		}// y
// 	      }// rotations
// 	    }// scales	    

// 	  }// samples


	  /** 

	      ver1 

	   */

	  
// 	  FloatGrid4 part_posterior_discount(part_posterior.shape());
	  
// 	  for (uint sampidx = 0; sampidx < idxmat.size1(); ++sampidx) 
// 	    part_posterior_discount[idxmat[0]][idxmat[1]][idxmat[2]][idxmat[3]] = part_posterior[idxmat[0]][idxmat[1]][idxmat[2]][idxmat[3]];
	  
// 	  /** also discount the neighbouring positions */

// 	  double disc_scale_sigma = 0.05;
// 	  double disc_rot_sigma = 7.5;
// 	  double disc_pos_sigma = 5;
	  
// 	  double scale_step_size = part_app.m_exp_param.max_object_scale() - 


// 	  FloatGrid4 tmpgrid(part_posterior.shape());

// 	  // scale 
// 	  for (int idx1 = 0; idx1 < part_posterior.shape[1]; ++idx1)
// 	    for (int idx2 = 0; idx2 < part_posterior.shape[2]; ++idx2)
// 	      for (int idx3 = 0; idx3 < part_posterior.shape[3]; ++idx3) {
// 		FloatGrid4View1 view_in = part_posterior_discount[boost::indices[index_range][idx1][idx2][idx3]];
// 		FloatGrid4View1 view_out = tmpgrid[boost::indices[index_range][idx1][idx2][idx3]];

// 		multi_array_op::grid_filter_1d(view_in, view_out, f_scale);
// 	      }
		

	  /** END debug */


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
	}
	else {
	  assert(false && "unknown part samples type");
	}

	/** save samples */
	QString qsOutputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";
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



  void findObjDai(const PartApp &part_app, int firstidx, int lastidx)
  {
    cout << "disc_ps::findObjDai" << endl;

    cout << "processing images from " << firstidx << " to " << lastidx << endl;

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      
      QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
	padZeros(QString::number(imgidx), 4);

      QString qsResultsDir;

      if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("part_marginals_samples")) {
	qsResultsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
					      "/part_marginals_samples_post");
      }
      else if (QString::fromStdString(part_app.m_exp_param.dai_samples_dir()).endsWith("test_scoregrid_samples")) {
	qsResultsDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + 
					      "/test_scoregrid_samples_post");
      }
      else {
	assert(false && "unknown part samples type");
      }

      if (!filesys::check_dir(qsResultsDir)) {
	cout << "creating "  << qsResultsDir.toStdString() << endl;
	assert(filesys::create_dir(qsResultsDir));
      }

      /** 
	  load samples
      */
      
      int nParts = part_app.m_part_conf.part_size();

      /** MA: we need this because of the awkward implementation that stores part hypothesis as bounding boxes 
	  and part scale is needed when computing detection scores and spatial factors
       */
      vector<vector<double> > part_vect_scales;

      vector<vector<PartBBox> > part_hyp(nParts, vector<PartBBox>());

      /** these are not always available */
      vector<vector<double> > part_scores(nParts, vector<double>());

      QImage img;
      assert(img.load(part_app.m_test_annolist[imgidx].imageName().c_str()));

      for (int pidx = 0; pidx < nParts; ++pidx) {
	QString qsInputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";
	assert(filesys::check_file(qsInputFilename));

	vector<int> vect_scale_idx;
	vector<int> vect_rot_idx; 
	vector<int> vect_iy; 
        vector<int> vect_ix;
	vector<double> vect_scores;

	cout << "loading samples from " << qsInputFilename.toStdString() << endl;

	bool res5 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scale_idx", vect_scale_idx);
	bool res1 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_rot_idx", vect_rot_idx);
        bool res2 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_iy", vect_iy);
        bool res3 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_ix", vect_ix);

	
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

        samples_to_bbox(part_app, pidx,  
                        vect_scale_idx, vect_rot_idx, vect_iy, vect_ix, part_hyp[pidx]);

	vector<double> vect_scales;
	for (uint idx = 0; idx < vect_scale_idx.size(); ++idx)
	  vect_scales.push_back(scale_from_index(part_app.m_exp_param, vect_scale_idx[idx]));

	part_vect_scales.push_back(vect_scales);
      }// parts 
      

      /**
	 compute factors
      */

      vector<Factor1d> factors1d;
      vector<Factor2d> factors2d;

      vector<Joint> joints;
      bool flip = false;
      object_detect::loadJoints(part_app, joints, flip);

      /** abc factor */
      kma::ImageContent *kma_input_image = 0;

      /** assume that scores are available for all parts or for none of them */
      if (part_scores[0].size() == 0) 
	kma_input_image = kma::load_convert_gray_image(part_app.m_test_annolist[imgidx].imageName().c_str());

      for (int pidx = 0; pidx < nParts; ++pidx) {

	Factor1d boosting_factor;

	/**
	   some parts don't have the part detector and are there to conviniently formulate the spatial model
	 */
	if (part_app.m_part_conf.part(pidx).is_detect()) {
	  if (part_scores[pidx].size() == 0) {
	    assert(kma_input_image != 0);

	    /** assume single scale, not anymore :)  */
	    //assert(part_app.m_exp_param.num_scale_steps() == 1);
	    //int scaleidx = 0;
	    //compute_boosting_score_factor(part_app, pidx, scaleidx, kma_input_image, part_hyp[pidx], boosting_factor);

	    compute_boosting_score_factor(part_app, pidx, kma_input_image, part_hyp[pidx], part_vect_scales[pidx], boosting_factor);
	  }
	  else {
	    boosting_factor.varidx = pidx;
	    boosting_factor.fval.resize(boost::extents[part_scores[pidx].size()]);

	    for (uint idx = 0; idx < part_scores[pidx].size(); ++idx) 
	      boosting_factor.fval[idx] = part_scores[pidx][idx];
	  }
	}
	else {
	  boosting_factor.varidx = pidx;
	  boosting_factor.fval.resize(boost::extents[part_hyp[pidx].size()]);

	  for (uint idx = 0; idx < part_hyp[pidx].size(); ++idx) 
	    boosting_factor.fval[idx] = 1.0 / part_hyp[pidx].size();
	  
	}

	factors1d.push_back(boosting_factor);
	boosting_factor.dai_print(cout);
      }// parts

      if (kma_input_image != 0) {
	delete kma_input_image;
	kma_input_image = 0;
      }
    
      /** spatial factor */
      cout << "computing spatial factors" << endl;

      for (uint jidx = 0; jidx < joints.size(); ++jidx) {
	Factor2d spatial_factor;

	compute_spatial_factor(joints[jidx], 
			       part_hyp[joints[jidx].child_idx], part_hyp[joints[jidx].parent_idx], 
			       part_vect_scales[joints[jidx].child_idx], part_vect_scales[joints[jidx].parent_idx], 
			       spatial_factor);
      
	factors2d.push_back(spatial_factor);
      }

      /** some checking */
      cout << "number of unary factors: " << factors1d.size() << endl;

      for (uint idx = 0; idx < factors1d.size(); ++idx) {
	int pidx = factors1d[idx].varidx;
	assert(factors1d[idx].fval.shape()[0] == part_hyp[pidx].size());
      }

      cout << "number of pairwise factors: " << factors2d.size() << endl;

      for (uint idx = 0; idx < factors2d.size(); ++idx) {
	int pidx1 = factors2d[idx].varidx1;
	int pidx2 = factors2d[idx].varidx2;

	assert(factors2d[idx].fval.shape()[0] == part_hyp[pidx1].size());
	assert(factors2d[idx].fval.shape()[1] == part_hyp[pidx2].size());

	cout << "factor, pidx1: " << pidx1 << ", pidx2: " << pidx2 << ", ok" << endl;
      }

      assert(factors1d.size() > 0 || factors2d.size() > 0);

      /** 
	  DAI inference 
      */
      // ...

      stringstream str_stream;
      dai_print_stream(str_stream, factors1d, factors2d);

      dai::FactorGraph fg;
      str_stream >> fg;

      size_t  maxiter = 500;
      double  tol = 1e-9;
      size_t  verb = 1;

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

      vector<double_vector> part_posterior;

      for (size_t i = 0; i < fg.nrVars(); i++ ) {
	dai::Factor cur_belief = bp.belief(fg.var(i));

	int nStates = cur_belief.states();

	boost_math::double_vector cur_posterior(nStates);
	for (int sidx = 0; sidx < nStates; ++sidx) {
	  cur_posterior[sidx] = cur_belief[sidx];
	}

	part_posterior.push_back(cur_posterior);
      }

      QString qsResultsFilename = qsResultsDir + "/samples_imgidx" + padZeros(QString::number(imgidx), 4) + "_post.mat";

      cout << "saving results to " << qsResultsFilename.toStdString() << endl;

      MATFile *f = matlab_io::mat_open(qsResultsFilename, "wz");
      assert(f != 0);

      for (int pidx = 0; pidx < nParts; ++pidx) {
	QString qsVarName = "samples_post_part" + QString::number(pidx);
	matlab_io::mat_save_double_vector(f, qsVarName, part_posterior[pidx]);
      }
      
      if (f != 0)
	matlab_io::mat_close(f);


    } // images

  }


  void visSamples(const PartApp &part_app, int firstidx, int lastidx)
  {
    cout << "processing images from " << firstidx << " to " << lastidx << endl;

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      QString qsPrecomputedSamplesDir = QString::fromStdString(part_app.m_exp_param.dai_samples_dir() + "/samples_imgidx") + 
	padZeros(QString::number(imgidx), 4);

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

      if (!filesys::check_dir(qsResultsDir)) {
	for (uint pidx = 0; pidx < (uint)part_app.m_part_conf.part_size(); ++pidx) {
	  QString qsResultsDirPart = qsResultsDir + QString("_pidx") + padZeros(QString::number(pidx), 2);
	  cout << "creating "  << qsResultsDirPart.toStdString() << endl;
	  assert(filesys::create_dir(qsResultsDir));
	}
      }

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
	QString qsInputFilename = qsPrecomputedSamplesDir + "/samples_pidx" + QString::number(pidx) + ".mat";
	assert(filesys::check_file(qsInputFilename));

	vector<int> vect_scale_idx; 
	vector<int> vect_rot_idx; 
	vector<int> vect_iy; 
        vector<int> vect_ix;
	vector<double> vect_scores;

	cout << "loading samples from " << qsInputFilename.toStdString() << endl;

	bool res5 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_scale_idx", vect_scale_idx);
	bool res1 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_rot_idx", vect_rot_idx);
        bool res2 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_iy", vect_iy);
        bool res3 = matlab_io::mat_load_stdcpp_vector(qsInputFilename, "vect_ix", vect_ix);

	
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
	  draw_bbox(painter, part_hyp[pidx][sidx], coloridx, pen_width);
	}
	
// 	QString qsResultsDirPart = qsResultsDir + "/" + QString("_pidx") + padZeros(QString::number(pidx), 2);
// 	if (!filesys::check_dir(qsResultsDirPart)) {
// 	  cout << "creating "  << qsResultsDirPart.toStdString() << endl;
// 	  assert(filesys::create_dir(qsResultsDirPart));
// 	}

	QString qsResultsFile = qsResultsDir + "/samples_pidx" + padZeros(QString::number(pidx), 2) + ".jpg";
	
	cout << "saving " << qsResultsFile.toStdString() << endl;
	
	assert(_img.save(qsResultsFile));
	  
      }// parts 
      


    }// images
	  

  }


}// namespace 
