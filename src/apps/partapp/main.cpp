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

#include <QFile>
#include <QTextStream>
#include <QString>

#include <boost/program_options.hpp>

#include <libPartApp/partapp.h>
#include <libPartApp/partapp_aux.hpp>

#include <libPartDetect/partdetect.h>

#include <libMisc/misc.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libPictStruct/objectdetect.h>
#include <libPictStruct/objectdetect_sample.h>

// this is only needed for szOption_EvalGt
#include <libPartDetect/partdetect.h>

#include <libMatlabIO/matlab_io.h>
#include <libMatlabIO/matlab_io.hpp>

#include <libDiscPS/disc_ps.h>

// so that we can parse expparam before initalizing PartApp
#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartEval/parteval.h>

using namespace std;
namespace po = boost::program_options;

const char* szOption_Help = "help";
const char* szOption_ExpOpt = "expopt";
const char* szOption_TrainClass = "train_class";
const char* szOption_TrainBootstrap = "train_bootstrap";
const char* szOption_Pidx = "pidx";
const char* szOption_Tidx = "tidx";
const char* szOption_EvalThresh = "eval_thresh";

const char* szOption_BootstrapPrep = "bootstrap_prep";
const char* szOption_BootstrapDetect = "bootstrap_detect";
const char* szOption_BootstrapShowRects = "bootstrap_showrects";

const char* szOption_First = "first";
const char* szOption_NumImgs = "numimgs";

const char* szOption_PartDetect = "part_detect";
const char* szOption_PartDetectDPM = "part_detect_dpm";
const char* szOption_HeadDetectDPM = "head_detect_dpm";
const char* szOption_FindObj = "find_obj";
const char* szOption_FindObjMix = "find_obj_mix";
const char* szOption_FindObjRoi = "find_obj_roi";

const char* szOption_pc_learn = "pc_learn";
const char* szOption_pc_learn_types = "pc_learn_types";
const char* szOption_SaveRes = "save_res";
const char* szOption_SaveCombRespTrain = "save_resp_train";
const char* szOption_SaveCombRespTest = "save_resp_test";
const char* szOption_Clus = "clus";
const char* szOption_SavePredictedPartConfTrain = "save_pred_conf_train";
const char* szOption_SavePredictedPartConfTest = "save_pred_conf_test";
const char* szOption_TrainLDA_Pwise = "train_lda_pwise";
const char* szOption_TrainLDA_Urot = "train_lda_urot";
const char* szOption_TrainLDA_Upos = "train_lda_upos";

const char* szOption_EvalSegments = "eval_segments";
const char* szOption_EvalSegmentsMix = "eval_segments_mix";
const char* szOption_EvalSegmentsDAI = "eval_segments_dai";
const char* szOption_EvalSegmentsDidx = "eval_didx";
const char* szOption_EvalSegmentsRPC = "eval_segments_rpc";
const char* szOption_ComputePosPrior = "compute_pos_prior";

const char* szOption_EvalSegmentsRoi = "eval_segments_roi";

const char* szOption_VisSegments = "vis_segments";
const char* szOption_VisSegmentsMix = "vis_segments_mix";
const char* szOption_VisSegmentsDAI = "vis_segments_dai";
const char* szOption_VisSegmentsRoi = "vis_segments_roi";

const char* szOption_VisDAISamples = "vis_dai_samples";

const char* szOption_Distribute = "distribute";
const char* szOption_NCPU = "ncpu";
const char* szOption_BatchNumber = "batch_num";

const char* szOption_VisParts = "vis_parts";

const char *szOption_PartSample = "part_sample";
const char *szOption_FindObjDai = "find_obj_dai";
const char *szOption_Recompute = "recompute";

const char *szOption_CompIdx = "compidx";

const char* szOption_ComputePoseLL = "compute_pose_ll";

/**
   initialize first and last indices from command line parameters 
   
   check that indices are in the valid range

   (this could also be used to support automatic splitting of data in chunks, see old code)
 */

void init_firstidx_lastidx_hard(vector<int> validation_all_idx, po::variables_map cmd_vars_map, int &firstidx, int &lastidx)
{
  int param_firstidx;
  int param_lastidx;

  int maxidx = -1;
  for(int i=0;i<validation_all_idx.size();i++)
    if (maxidx < validation_all_idx[i])
      maxidx = validation_all_idx[i];
  
  if (cmd_vars_map.count(szOption_First))
    param_firstidx = cmd_vars_map[szOption_First].as<int>();
  else
    param_firstidx = 0;

  if (cmd_vars_map.count(szOption_NumImgs)) { 
    param_lastidx = param_firstidx + cmd_vars_map[szOption_NumImgs].as<int>() - 1;
  }
  else {
    param_lastidx = maxidx;
  }

  firstidx = param_firstidx;
  lastidx = param_lastidx;
  
  check_bounds_and_update(firstidx, 0, maxidx + 2);
  check_bounds_and_update(lastidx, 0, maxidx+1);  
  
}

void init_firstidx_lastidx(const AnnotationList &annolist, po::variables_map cmd_vars_map, int &firstidx, int &lastidx)
{
  int param_firstidx;
  int param_lastidx;

  if (cmd_vars_map.count(szOption_First))
    param_firstidx = cmd_vars_map[szOption_First].as<int>();
  else
    param_firstidx = 0;

  if (cmd_vars_map.count(szOption_NumImgs)) { 
    param_lastidx = param_firstidx + cmd_vars_map[szOption_NumImgs].as<int>() - 1;
  }
  else {
    param_lastidx = annolist.size() - 1;
  }

  if (cmd_vars_map.count(szOption_Distribute)) {
    assert(cmd_vars_map.count(szOption_NCPU));
    assert(cmd_vars_map.count(szOption_BatchNumber));

    int ncpu = cmd_vars_map[szOption_NCPU].as<int>();
    int batch_num = cmd_vars_map[szOption_BatchNumber].as<int>();

    int num_per_cpu = (int)ceil((param_lastidx - param_firstidx + 1)/(float)ncpu);
    firstidx = param_firstidx + batch_num*num_per_cpu;
    lastidx = firstidx + num_per_cpu - 1;
    cout << "ncpu: " << ncpu << ", num_per_cpu: " << num_per_cpu << endl;
  }
  else {
    firstidx = param_firstidx;
    lastidx = param_lastidx;
  }

  /* allow firstidx to be > then lastidx, no images will be processed in this case */
  check_bounds_and_update(firstidx, 0, (int)annolist.size() + 1);
  check_bounds_and_update(lastidx, 0, (int)annolist.size());  
}

int main(int argc, char *argv[])
{
  cout << "multi-component version" << endl;
  
  /* parse command line options */
  po::options_description cmd_options_desc("command line options:");
  po::variables_map cmd_vars_map;

  cmd_options_desc.add_options()
    (szOption_Help, "help message")
    (szOption_ExpOpt, po::value<string>(), "experiment parameters")   
    (szOption_TrainClass, "train part detector")
    (szOption_TrainBootstrap, "train part detector, adding hard negative samples")
    (szOption_Pidx, po::value<int>(), "0-based index of the part")
    (szOption_Tidx, po::value<int>(), "0-based index of the part type")
    (szOption_CompIdx, po::value<int>(), "0-based index of the component")
    (szOption_EvalThresh, po::value<int>(), "pcp/rpc threshold used during evaluation")

    (szOption_BootstrapPrep, "create bootstrapping dataset (crops of objects with some background)")
    (szOption_BootstrapDetect, "run previously trained classifer on bootstrapping images")
    (szOption_BootstrapShowRects, "show top negatives on bootstrapping images")

    (szOption_First, po::value<int>(), "index of first image")
    (szOption_NumImgs, po::value<int>(), "number of images to process")

    (szOption_PartDetect, "run AdaBoost part detector on the test set")
    (szOption_PartDetectDPM, "run DPM part detector on the test set")
    (szOption_HeadDetectDPM, "run DPM head detector on the test set")
    
    (szOption_FindObj, "find_obj")
    (szOption_FindObjMix, "choose a mixture component for evaluation")
    (szOption_FindObjRoi, "same as find_obj, but for the \"region of interest\" only ")

    (szOption_pc_learn, "estimate prior on part configurations with maximum likelihood")
    (szOption_pc_learn_types, "estimate prior on part configurations with maximum likelihood for multiple joint types")
    (szOption_SaveRes, "save object recognition results in al/idl formats")
    (szOption_SaveCombRespTrain, "save combined detector responces on train set")
    (szOption_SaveCombRespTest, "save combined detector responces on test set")
    (szOption_SavePredictedPartConfTrain, "save predicted prior for each train image")
    (szOption_SavePredictedPartConfTest, "save predicted prior for each test image")
    (szOption_Clus, "cluster the training data")
    (szOption_TrainLDA_Pwise, "train LDA classifier to predict pairwise factors")
    (szOption_TrainLDA_Urot, "train LDA classifier to predict unary rotation factors")
    (szOption_TrainLDA_Upos, "train LDA classifier to predict unary position factors")
    
    (szOption_EvalSegments, "evaluate part localization according to Ferrari's criteria")
    (szOption_EvalSegmentsMix, "evaluate part localization according to Ferrari's criteria; PS mix")
    (szOption_EvalSegmentsDidx, po::value<int>(), "index of the subject to evaluate")
    (szOption_EvalSegmentsDAI, "evaluate part localization according to Ferrari's criteria loading the DAI results")
    (szOption_EvalSegmentsRoi, "evaluate results for computed for regions of interest")
    (szOption_EvalSegmentsRPC, "evaluate part localization using RPC curve")
    (szOption_ComputePosPrior, "compute prior on part position")

    (szOption_VisSegments, "visualise part estimates from dense PS model")
    (szOption_VisSegmentsMix, "visualise part estimates from dense PS mixture model")
    (szOption_VisSegmentsDAI, "visualise part estimates from discrete PS model")
    (szOption_VisSegmentsRoi, "visualise part estimates for each region of interest")

    (szOption_VisDAISamples, "visualise samples used for DAI inference")

    (szOption_Distribute, "split processing into multiple chunks, keep track of the chunks which must be processed")
    (szOption_NCPU, po::value<int>(), "number of chunks")
    (szOption_BatchNumber, po::value<int>(), "current chunk")
    (szOption_VisParts, "visualize ground-truth part positions")

    (szOption_PartSample, "sample part locations and save them for later processing")
    (szOption_FindObjDai, "perform inference with discrete version of pictorial structures")
    (szOption_Recompute, "recompute")
    
    (szOption_ComputePoseLL, "compute Log-Likelihood of part configuration (pairwise factors only)")
    ;


  /* BEGIN: handle 'distribute' option */

  const QString qsArgsFile = ".curexp";
  const QString qsCurBatchFile = ".curexp_batchnum";
  const int MAX_ARGC = 255;
  const int MAX_ARG_LENGTH = 1024;
  cout << "argc: " << argc << endl;
  cout << "argv[0]: " << argv[0] << endl;
  assert(argc <= MAX_ARGC);

  int loaded_argc = 0;
  char loaded_argv[MAX_ARGC][MAX_ARG_LENGTH];
  char *loaded_argv_ptr[MAX_ARGC];
  for (int idx = 0; idx < MAX_ARGC; ++idx) {
    loaded_argv_ptr[idx] = loaded_argv[idx];
  }

  /* try to load parameters from experiment file if none are specified on the command line */
  if (argc <= 1 && filesys::check_file(qsArgsFile)) {
    // load .curexp, fill argc and argv
    cout << "reading arguments from " << qsArgsFile.toStdString() << endl;
    QFile qfile( qsArgsFile );

    assert(qfile.open( QIODevice::ReadOnly | QIODevice::Text));
    QTextStream stream( &qfile );
    QString line;

    while(!(line = stream.readLine()).isNull()) {
      if (!line.isEmpty()) {
        cout << line.toStdString() << endl;
        assert(line.length() < MAX_ARG_LENGTH);
        
        int cur_idx = loaded_argc;
        strcpy(loaded_argv[cur_idx], line.toStdString().c_str());
        ++loaded_argc;
      }
    }

  }

  /* process command option */

  try {

    if (loaded_argc > 0) {
      cout << "reading parameters from " << qsArgsFile.toStdString() << endl;
      po::store(po::parse_command_line(loaded_argc, loaded_argv_ptr, cmd_options_desc), cmd_vars_map);

      ifstream ifs;
      ifs.open(qsCurBatchFile.toStdString().c_str());
      po::store(po::parse_config_file(ifs, cmd_options_desc), cmd_vars_map);
      ifs.close();

      assert(cmd_vars_map.count("batch_num") > 0);
      int batch_num = cmd_vars_map["batch_num"].as<int>();
      cout << "batch_num: " << batch_num << endl;
      ++batch_num;

      cout << "updating " << qsCurBatchFile.toStdString() << endl;
      QFile qfile2(qsCurBatchFile);
      assert(qfile2.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
      QTextStream stream2( &qfile2 );
      stream2 << "batch_num = " << batch_num << endl;
    }
    else {
      cout << "reading command line parameters" << endl;

      po::command_line_style::style_t cmd_style = (po::command_line_style::style_t)(po::command_line_style::allow_short | 
										    po::command_line_style::short_allow_adjacent | 
										    po::command_line_style::short_allow_next |
										    po::command_line_style::allow_long | 
										    po::command_line_style::long_allow_adjacent | 
										    po::command_line_style::long_allow_next |
										    po::command_line_style::allow_sticky | 
										    //po::command_line_style::allow_guessing |
										    po::command_line_style::allow_dash_for_short);

      po::store(po::parse_command_line(argc, argv, cmd_options_desc, cmd_style), cmd_vars_map);
    }
      
    po::notify(cmd_vars_map);
  }
  catch (exception &e) {
    cerr << "error: " << e.what() << endl;
    return 1;
  }  

  /* "distribute" option means store parameters and initialize batch counter */
  if (argc > 1 && cmd_vars_map.count("distribute") > 0) {
    assert(cmd_vars_map.count("ncpu") > 0);
    
    cout << "saving command line parameters in " << qsArgsFile.toStdString() << endl;
    QFile qfile( qsArgsFile );
    assert(qfile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    QTextStream stream( &qfile );

    for (int idx = 0; idx < argc; ++idx) 
      stream << argv[idx] << endl;

    /* how to initialize program option explicitly ??? */
    cout << "intializing batch counter: " << qsCurBatchFile.toStdString() << endl;
    {
      QFile qfile2(qsCurBatchFile);
      assert(qfile2.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
      {
        QTextStream stream2( &qfile2 );
        stream2 << "batch_num = 0" << endl;
      }
      qfile2.close();
    }

    ifstream ifs;
    ifs.open(qsCurBatchFile.toStdString().c_str());
    po::store(po::parse_config_file(ifs, cmd_options_desc), cmd_vars_map);
    ifs.close();

    {
      QFile qfile2(qsCurBatchFile);
      assert(qfile2.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
      {
        QTextStream stream2( &qfile2 );
        stream2 << "batch_num = 1" << endl;
      }
      qfile2.close();
    }
  }

  /* END: handle 'distribute' option */


  /* process options which do not require initialized partapp */
  if (cmd_vars_map.count(szOption_Help)) {
    cout << cmd_options_desc << endl << endl;
    return 1;
  }

  /* initialize partapp from parameter file */
  if (cmd_vars_map.count(szOption_ExpOpt) == 0) {
    cout << cmd_options_desc << endl << endl;
    cout << "'expopt' parameter missing" << endl;
    return 1;
  }

//   PartApp part_app;
//   part_app.init(qsExpParam);  


  QString qsExpParam = cmd_vars_map[szOption_ExpOpt].as<string>().c_str();
  cout << "initializing from " << qsExpParam.toStdString() << endl;
  ExpParam exp_param;
  parse_message_from_text_file(qsExpParam, exp_param);
 
  vector<PartApp> partapp_components;
  
  if (exp_param.is_multicomponent()) {
    PartApp::init_multicomponent(qsExpParam, partapp_components);
  }
  else {
    PartApp part_app;
    part_app.init(qsExpParam);
    partapp_components.push_back(part_app);
  }

  int min_compidx = 0; 
  int max_compidx = partapp_components.size() - 1;
  if (cmd_vars_map.count(szOption_CompIdx) > 0) {
    int compidx = cmd_vars_map[szOption_CompIdx].as<int>();
    assert(compidx >= 0 && compidx < partapp_components.size());
    
    min_compidx = compidx;
    max_compidx = compidx;
  }
 
  for (uint compidx = min_compidx; compidx <= max_compidx; ++compidx) {

    PartApp &part_app = partapp_components[compidx];

    if (cmd_vars_map.count(szOption_TrainClass)) {
      assert(!part_app.m_exp_param.is_multicomponent());
      assert(!part_app.m_bExternalClassDir && "classifiers should be trained in the original project");
 
      bool bBootstrap = cmd_vars_map.count(szOption_TrainBootstrap); // default is false
      if (cmd_vars_map.count(szOption_Pidx)) {

	int pidx = cmd_vars_map[szOption_Pidx].as<int>();
	assert(pidx < part_app.m_part_conf.part_size());
	assert(part_app.m_part_conf.part(pidx).is_detect());
	
	if (part_app.m_part_conf.part(pidx).has_mult_types()){
	  
	  vector<AnnotationList> annolistClusters;
	  
	  part_detect::loadPartTypeData(part_app.m_exp_param.class_dir().c_str(), part_app.m_window_param, part_app.m_part_conf, pidx, annolistClusters);
	  	
	  if (cmd_vars_map.count(szOption_Tidx)){
	    int tidx = cmd_vars_map[szOption_Tidx].as<int>();
	    int nTypes = part_detect::getNumPartTypes(part_app.m_window_param, pidx);
	    if (not (tidx < nTypes)){
	      cout << "tidx " << tidx << " >= nPartTypes " << nTypes << endl;
	      continue;
	    }
	    part_app.m_train_annolist = annolistClusters[tidx];
	    cout << "training classifier for part " << pidx << ", part type " << tidx << endl;
	      
	    part_detect::abc_train_class(part_app, pidx, bBootstrap, tidx);
	  }
	  else
	    for(int tidx = 0; tidx < annolistClusters.size(); tidx++){
	      part_app.m_train_annolist = annolistClusters[tidx];
	      cout << "training classifier for part " << pidx << ", part type " << tidx << endl;
	      part_detect::abc_train_class(part_app, pidx, bBootstrap, tidx);
	    }
	}
	else{
	  cout << "training classifier for part " << pidx << endl;
	  part_detect::abc_train_class(part_app, pidx, bBootstrap);
	}
      }
      else {
	for (int pidx = 0; pidx < part_app.m_part_conf.part_size(); ++pidx) {
	  if (part_app.m_part_conf.part(pidx).is_detect()) {
	    cout << "training classifier for part " << pidx << endl;
	    part_detect::abc_train_class(part_app, pidx, bBootstrap);
	  }
	}
      }

    }// train class
    else if (cmd_vars_map.count(szOption_BootstrapPrep)) {
      assert(!part_app.m_exp_param.is_multicomponent());
      
      if (part_app.m_exp_param.bootstrap_dataset_size() > 0)
	part_detect::prepare_bootstrap_dataset(part_app, part_app.m_bootstrap_annolist, 0, part_app.m_bootstrap_annolist.size() - 1);
      else
	part_detect::prepare_bootstrap_dataset(part_app, part_app.m_train_annolist, 0, part_app.m_train_annolist.size() - 1);
    }
    else if (cmd_vars_map.count(szOption_BootstrapDetect)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      if (part_app.m_exp_param.bootstrap_dataset_size() > 0)
	init_firstidx_lastidx(part_app.m_bootstrap_annolist, cmd_vars_map, firstidx, lastidx);
      else
	init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);
      
      part_detect::bootstrap_partdetect(part_app, firstidx, lastidx);
    }
    else if (cmd_vars_map.count(szOption_BootstrapShowRects)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      assert(cmd_vars_map.count("pidx") && "part index missing");
      int pidx = cmd_vars_map["pidx"].as<int>();

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);

      int num_rects = 50;
      double min_score = 0.1;
      vector<PartBBox> v_rects;
      vector<double> v_rects_scale;
    
      bool bIgnorePartRects = true;
      bool bDrawRects = true;

      /* create debug directory to save bootstrap images */
      if (!filesys::check_dir("./debug")) {
	cout << "creating ./debug" << endl;
	filesys::create_dir("./debug");
      }

      part_detect::bootstrap_get_rects(part_app, firstidx, pidx, num_rects, min_score, 
				       v_rects, v_rects_scale,
				       bIgnorePartRects, bDrawRects, pidx);

    }
    else if (cmd_vars_map.count(szOption_pc_learn)) {
      assert(!part_app.m_exp_param.is_multicomponent());
      object_detect::learn_conf_param(part_app, part_app.m_train_annolist);
    }
    else if (cmd_vars_map.count(szOption_Clus)) {
      assert(!part_app.m_exp_param.is_multicomponent());
      // TODO: clustering is done during init
      // better ways?
      cout << "Clustering... Done!" << endl;
    }
    else if (cmd_vars_map.count(szOption_pc_learn_types)) {
      assert(!part_app.m_exp_param.is_multicomponent());
      
      object_detect::learn_conf_param_pred_data(part_app);
    }
    else if (cmd_vars_map.count(szOption_SaveRes)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int scoreProbMapType = object_detect::SPMT_NONE;

      object_detect::saveRecoResults(part_app, scoreProbMapType);
    }
    else if (cmd_vars_map.count(szOption_EvalSegmentsRPC)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;
      
      if (cmd_vars_map.count(szOption_Pidx) && cmd_vars_map.count(szOption_Tidx)) {
	int pidx = cmd_vars_map[szOption_Pidx].as<int>();
	int tidx = cmd_vars_map[szOption_Tidx].as<int>();
	int nTypes = part_detect::getNumPartTypes(part_app.m_window_param, pidx);
	if (nTypes > 1){
	  vector<AnnotationList> annolistClusters;
	  part_detect::loadPartTypeData(part_app.m_exp_param.class_dir().c_str(), 
					part_app.m_window_param, 
					part_app.m_part_conf, pidx, annolistClusters);
	  eval_segments_rpc(part_app, annolistClusters[tidx], firstidx, lastidx, pidx, tidx);
	}
	else
	  eval_segments_rpc(part_app, part_app.m_train_annolist, firstidx, lastidx, pidx, tidx);
      }
      else
	for (int pidx = 0; pidx < part_app.m_part_conf.part_size(); ++pidx) {
	  int nTypes = part_detect::getNumPartTypes(part_app.m_window_param, pidx);
	  if (nTypes > 1)
	    for (int tidx = 0; tidx < nTypes; ++tidx){
	      vector<AnnotationList> annolistClusters;
	      part_detect::loadPartTypeData(part_app.m_exp_param.class_dir().c_str(), 
					    part_app.m_window_param, 
					    part_app.m_part_conf, pidx, annolistClusters);
	      eval_segments_rpc(part_app, annolistClusters[tidx], firstidx, lastidx, pidx, tidx);
	    }
	  else{
	    eval_segments_rpc(part_app, part_app.m_train_annolist, firstidx, lastidx, pidx, 0);
	    break;
	  }
	}
    }
    else if (cmd_vars_map.count(szOption_EvalSegmentsMix)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;
      
      double ratio = 0.0;
      
      QString qsSegEndPointsMixDir = partapp_components[0].m_exp_param.mix_dir().c_str();
      eval_segments(part_app, part_app.m_test_annolist, firstidx, lastidx, ratio, qsSegEndPointsMixDir);
      break;
    }
    else if (cmd_vars_map.count(szOption_VisSegmentsMix)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;
      
      QString qsHypDirName = partapp_components[0].m_exp_param.mix_dir().c_str();
      
      vis_segments(part_app, part_app.m_test_annolist, firstidx, lastidx, qsHypDirName);
      break;
    }
    else if (cmd_vars_map.count(szOption_VisSegmentsDAI)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;    
      
      vis_segments(part_app, part_app.m_test_annolist, firstidx, lastidx, "", EVAL_TYPE_DISC_PS);
    }
    else if (cmd_vars_map.count(szOption_VisSegmentsRoi)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;    

      QString qsHypDirName = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/part_marginals_roi").c_str();
      
      vis_segments_roi(part_app, firstidx, lastidx, qsHypDirName);
    }
    else if (cmd_vars_map.count(szOption_VisDAISamples)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;

      disc_ps::visSamples(part_app, firstidx, lastidx);
    }
    else if (cmd_vars_map.count(szOption_EvalSegmentsDAI)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;

      int eval_didx = -1; 

      if (cmd_vars_map.count(szOption_EvalSegmentsDidx)) {
	eval_didx = cmd_vars_map[szOption_EvalSegmentsDidx].as<int>();
	cout << "evaluating detections for subject: " << eval_didx << endl;
      }
      double ratio = 0.0;
      eval_segments(part_app, part_app.m_test_annolist, firstidx, lastidx, ratio, "", EVAL_TYPE_DISC_PS, eval_didx);
    }
    else if (cmd_vars_map.count(szOption_EvalSegmentsRoi)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      cout << "processing images: " << firstidx << " to " << lastidx << endl;
      
      QString qsHypDirName = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/part_marginals_roi").c_str();
      
      eval_segments_roi(part_app, firstidx, lastidx, qsHypDirName);
    }
    else if (cmd_vars_map.count(szOption_VisParts)) {
      assert(!part_app.m_exp_param.is_multicomponent());

      int firstidx, lastidx;
      init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);

      for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
	QImage img = visualize_parts(part_app.m_part_conf, part_app.m_window_param, part_app.m_train_annolist[imgidx]);

	QString qsPartConfPath;
	QString qsPartConfName;
	QString qsPartConfExt;
	filesys::split_filename_ext(part_app.m_exp_param.part_conf().c_str(), qsPartConfPath, qsPartConfName, qsPartConfExt);

	QString qsDebugDir = (part_app.m_exp_param.log_dir() + "/" + 
			      part_app.m_exp_param.log_subdir() + "/debug").c_str();

	if (!filesys::check_dir(qsDebugDir))
	  filesys::create_dir(qsDebugDir);

	QString qsFilename = qsDebugDir + "/parts-" + qsPartConfName + "-imgidx" + 
	  padZeros(QString::number(imgidx), 4) + ".png";
	cout << "saving " << qsFilename.toStdString() << endl;
    
	assert(img.save(qsFilename));    
      }

    }else if (cmd_vars_map.count(szOption_ComputePosPrior)) {
      
      object_detect::computeTorsoPosPriorParams(part_app);
      
    }
    else{
      bool bShowHelpMessage = true;
    
      if (cmd_vars_map.count(szOption_PartDetect)) {
	assert(!part_app.m_exp_param.is_multicomponent());

	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);

	bool bSaveImageScoreGrid = part_app.m_exp_param.save_image_scoregrid();
	cout << "bSaveImageScoreGrid: " << bSaveImageScoreGrid << endl;
	
	vector<vector<FloatGrid3> > part_detections;
	part_detect::partdetect(part_app, firstidx, lastidx, false, bSaveImageScoreGrid, part_detections, part_app.m_test_annolist, QString(part_app.m_exp_param.scoregrid_dir().c_str()));
	if (part_app.m_exp_param.flip_orientation())
	  part_detect::partdetect(part_app, firstidx, lastidx, true, bSaveImageScoreGrid, part_detections, part_app.m_test_annolist, QString(part_app.m_exp_param.scoregrid_dir().c_str()));

	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_PartDetectDPM)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	
	if (cmd_vars_map.count(szOption_Pidx)) {
	  int pidx = cmd_vars_map[szOption_Pidx].as<int>();
	  QString qsUnaryDPMDir = (part_app.m_exp_param.test_dpm_unary_dir() + "/pidx_").c_str() + padZeros(QString::number(pidx), 4);
	  if (!filesys::check_dir(qsUnaryDPMDir))
	    filesys::create_dir(qsUnaryDPMDir);
	  QString qsModelDPMDir = (part_app.m_exp_param.dpm_model_dir() + "/pidx_").c_str() + padZeros(QString::number(pidx), 4);
	  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx)
	    part_detect::partdetect_dpm(part_app, imgidx, qsModelDPMDir, qsUnaryDPMDir);
	}
	else
	  for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx)
	    part_detect::partdetect_dpm_all(part_app, imgidx);
	  	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_HeadDetectDPM)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	
	QString qsUnaryDPMDir = (part_app.m_exp_param.test_dpm_unary_dir() + "/head").c_str();
	if (!filesys::check_dir(qsUnaryDPMDir))
	  filesys::create_dir(qsUnaryDPMDir);
	QString qsModelDPMDir = (part_app.m_exp_param.dpm_model_dir() + "/head").c_str();
	part_detect::partdetect_dpm(part_app, firstidx, qsModelDPMDir, qsUnaryDPMDir);
	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_FindObj)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	
	int scoreProbMapType = object_detect::SPMT_NONE;
	object_detect::findObjectDataset(part_app, firstidx, lastidx, scoreProbMapType);
	
	bShowHelpMessage = false;
      }

      if (cmd_vars_map.count(szOption_FindObjMix)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	
	object_detect::findObjectDatasetMix(partapp_components, firstidx, lastidx);
	// based on resuls of all components
	break;
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_PartSample)) {
	assert(!part_app.m_exp_param.is_multicomponent());

	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
      
	if (part_app.m_exp_param.dai_bbox_prior()) 
	  disc_ps::partSampleWithPrior(part_app, firstidx, lastidx);	
	else
	  disc_ps::partSample(part_app, firstidx, lastidx);	

	bShowHelpMessage = false;
      }

      if (cmd_vars_map.count(szOption_FindObjDai)) {

	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);

	bool bForceRecompute = cmd_vars_map.count(szOption_Recompute);
	disc_ps::findObjDai(part_app, firstidx, lastidx, bForceRecompute);
	/*
	bool bGenerateNewSamples = true;
		
	for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx)
	  disc_ps::exec(part_app, imgidx, bGenerateNewSamples, bForceRecompute);
	*/
	bShowHelpMessage = false;
      }

      if (cmd_vars_map.count(szOption_FindObjRoi)) {

	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);

	for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) 
	  object_detect::findObjectImageRoi(part_app, imgidx);

	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_EvalSegments)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	
	double ratio = 0.0;
	
	QString qsHypDirName = part_app.m_exp_param.part_marginals_dir().c_str();
	
	//int eval_type = EVAL_TYPE_UNARIES;
	
	eval_segments(part_app, part_app.m_test_annolist, firstidx, lastidx, ratio, qsHypDirName);

	bShowHelpMessage = false;
	
      }
      
      if (cmd_vars_map.count(szOption_VisSegments)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	
	QString qsHypDirName = (part_app.m_exp_param.log_dir() + "/" + 
				part_app.m_exp_param.log_subdir() + "/part_marginals").c_str();
	
	vis_segments(part_app, part_app.m_test_annolist, firstidx, lastidx, qsHypDirName);
	
	bShowHelpMessage = false;

      }
	
      if (cmd_vars_map.count(szOption_SaveCombRespTrain)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_validation_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	
	QString qsDetRespDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir()) + "/resp_train";
	QString qsScoreGridDir = part_app.m_exp_param.scoregrid_train_dir().c_str();
	
	for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
	  QString qsImgName = part_app.m_validation_annolist[imgidx].imageName().c_str();
	  
	  int nParts = part_app.m_part_conf.part_size();
	  	  
	  int rootidx_det = -1;
	  if (part_app.m_exp_param.has_rootidx_det())
	    rootidx_det = part_app.m_exp_param.rootidx_det();
	  
	  cout << "rootidx_det: " << rootidx_det << endl;
	  
	  assert(part_app.m_exp_param.has_torso_det_train_dir());
	  QString qsLogDir = part_app.m_exp_param.torso_det_train_dir().c_str();
	  QString qsFilename = qsLogDir + "/pose_est_imgidx" + padZeros(QString::number(imgidx),4) + ".mat";
	  cout << "loading " << qsFilename.toStdString().c_str() << endl;
	  boost_math::double_matrix best_conf;
	  matlab_io::mat_load_double_matrix(qsFilename, "best_conf", best_conf);

	  int root_pos_x = best_conf(rootidx_det,4);
	  int root_pos_y = best_conf(rootidx_det,5);
	  int root_rot   = best_conf(rootidx_det,3);
	  
	  cout << root_pos_x << endl;
	  cout << root_pos_y << endl;
	  
	  vector<vector<FloatGrid3> > part_detections;
	  part_detect::partdetect(part_app, imgidx, imgidx, false, false, part_detections, part_app.m_validation_annolist, QString(part_app.m_exp_param.scoregrid_train_dir().c_str()), false);
	  
	  part_detect::saveCombResponces(part_app, imgidx, qsDetRespDir, qsScoreGridDir, qsImgName, root_pos_x, root_pos_y, root_rot, part_detections, false);
	  
	}
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_SaveCombRespTest)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	
	QString qsDetRespDir = QString::fromStdString(part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir()) + "/resp_test";
	QString qsScoreGridDir = part_app.m_exp_param.scoregrid_dir().c_str();
		
	for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
	  QString qsImgName = part_app.m_test_annolist[imgidx].imageName().c_str();
	  
	  int nParts = part_app.m_part_conf.part_size();

	  int rootidx_det = -1;
	  if (part_app.m_exp_param.has_rootidx_det())
	    rootidx_det = part_app.m_exp_param.rootidx_det();
	  
	  cout << "rootidx_det: " << rootidx_det << endl;
	  
	  assert(part_app.m_exp_param.has_torso_det_test_dir());
	  QString qsLogDir = part_app.m_exp_param.torso_det_test_dir().c_str();
	  QString qsFilename = qsLogDir + "/pose_est_imgidx" + padZeros(QString::number(imgidx),4) + ".mat";	    
	  cout << "loading " << qsFilename.toStdString().c_str() << endl;
	  boost_math::double_matrix best_conf;
	  matlab_io::mat_load_double_matrix(qsFilename, "best_conf", best_conf);
	  
	  int root_pos_x = best_conf(rootidx_det,4);
	  int root_pos_y = best_conf(rootidx_det,5);
	  int root_rot   = best_conf(rootidx_det,3);
	  
	  cout << root_pos_x << endl;
	  cout << root_pos_y << endl;
	  
	  vector<vector<FloatGrid3> > part_detections;
	  part_detect::partdetect(part_app, imgidx, imgidx, false, false, part_detections, part_app.m_test_annolist, QString(part_app.m_exp_param.scoregrid_dir().c_str()), false);
	  
	  part_detect::saveCombResponces(part_app, imgidx, qsDetRespDir, qsScoreGridDir, qsImgName, root_pos_x, root_pos_y, root_rot, part_detections, false);
	}
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_SavePredictedPartConfTrain)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_validation_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	
	object_detect::savePredictedPartConf(part_app, firstidx, lastidx, false);
	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_TrainLDA_Pwise)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	if (cmd_vars_map.count(szOption_Pidx)) {
	  int pidx = cmd_vars_map[szOption_Pidx].as<int>();
	  object_detect::trainLDA(part_app, 0, pidx);
	}
	else
	  object_detect::trainLDA(part_app, 0);
	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_TrainLDA_Urot)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	if (cmd_vars_map.count(szOption_Pidx)) {
	  int pidx = cmd_vars_map[szOption_Pidx].as<int>();
	  object_detect::trainLDA(part_app, 1, pidx);
	}
	else
	  object_detect::trainLDA(part_app, 1);
	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_TrainLDA_Upos)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	if (cmd_vars_map.count(szOption_Pidx)) {
	  int pidx = cmd_vars_map[szOption_Pidx].as<int>();
	  object_detect::trainLDA(part_app, 2, pidx);
	}
	else
	  object_detect::trainLDA(part_app, 2);
	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_SavePredictedPartConfTest)) {
	assert(!part_app.m_exp_param.is_multicomponent());
	
	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	
	object_detect::savePredictedPartConf(part_app, firstidx, lastidx, true);
	
	bShowHelpMessage = false;
      }
      
      if (cmd_vars_map.count(szOption_ComputePoseLL)) {
	assert(!part_app.m_exp_param.is_multicomponent());

	int firstidx, lastidx;
	init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
	cout << "processing images: " << firstidx << " to " << lastidx << endl;
	object_detect::computePoseLL(part_app, firstidx, lastidx);
	
	bShowHelpMessage = false;
      }
      if (bShowHelpMessage) {
	cout << cmd_options_desc << endl;
	return 1;
      }
    }

  } // components
  
  return 0;
}


