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

#include <algorithm>
#include <QProcess>

#include <libFilesystemAux/filesystem_aux.h>
#include <libBoostMath/boost_math.h>
#include <libBoostMath/boost_math.hpp>
#include <libPrediction/matlab_runtime.h>
#include <libDPM/dpm_params.h>

#include "partdetect.h"
#include "partdef.h"
#include "kmeans.hpp"

using boost_math::double_matrix;
using boost_math::double_vector;
using namespace std;

namespace part_detect { 
  
  void getAlignedClusters(vector<AnnotationList> &annolistClusters, int pidx){
    
    QString qsClusDir = "/BS/leonid-pose/work/experiments-new-data/log_dir/poselets-dpm-lsp-kmedoids-new/";
    QString qsFilename = qsClusDir + "/pidx" + padZeros(QString::number(pidx),4) + "/nSeeds0200/clusAll.al";
    AnnotationList annolist;
    if (filesys::check_file(qsFilename.toStdString().c_str()))
      annolist.load(qsFilename.toStdString());

    int nTypes = 0;
    for (int imgidx=0; imgidx<annolist.size();imgidx++){
      int id = annolist[imgidx][0].silhouetteID();
      if (nTypes != id){
	nTypes = nTypes + 1;
      }
    }
    
    for(int cidx = 0; cidx < nTypes;cidx++){
      AnnotationList annolistClus;
      for (int imgidx=0; imgidx<annolist.size();imgidx++){
	int id = annolist[imgidx][0].silhouetteID();
	if (id == cidx + 1){
	  Annotation annotation = annolist[imgidx];
	  annolistClus.addAnnotation(annotation);
	}
	
      }
      annolistClusters.push_back(annolistClus);
    }
    cout << "annolistClusters.size(): " << annolistClusters.size() << endl;
  }
  
  void compute_part_type_window_param(const AnnotationList &annolist, const PartConfig &partconf, 
				      PartWindowParam &windowparam, const ExpParam &exp_param, QString qsClassDir)
  {
    int nParts = partconf.part_size();
    int nImagesTotal = annolist.size();

    /** 
        compute reference object height, use only rectangles that have all parts 
        annotated (rectangles with missing parts are sometimes smaller then reference object height)
    */
    double train_object_height = 0;
    int n = 0;
    for (int imgidx = 0; imgidx < nImagesTotal; ++imgidx) {
      int nRects = annolist[imgidx].size(); 
      for (int ridx = 0; ridx < nRects; ++ridx) {

        bool bHasAllParts = true;

        if (bHasAllParts) {
          train_object_height += abs(annolist[imgidx][ridx].bottom() - annolist[imgidx][ridx].top());
          ++n;
        }

      }// rectangles
    }// images

    assert(n > 0);
    
    train_object_height /= n;
    windowparam.set_train_object_height(train_object_height);
    
    cout << "train_object_height: " << train_object_height << endl;

    int rootpart_idx = -1;
    
    for (int pidx = 0; pidx < nParts; ++pidx) {
      
      if (partconf.part(pidx).is_root()) {
        rootpart_idx = pidx;
        break;
      }
    }
    
    assert(rootpart_idx >=0 && "missing root part");

    /** determine average dimensions of the parts */
    windowparam.clear_part();
    for (int pidx = 0; pidx < nParts; ++pidx) {
      vector<AnnotationList> annolistClusters;
      vector<AnnotationList> annolistClustersTest;
      vector<double> rot_clus;
      
      if (partconf.part(pidx).has_mult_types()){
	
	// split training data w.r.t. part types 
	part_detect::get_part_type_clusters(annolist, partconf, exp_param, annolistClusters, pidx);
	// save data subdivision
	for (int tidx = 0; tidx < annolistClusters.size(); tidx++){
	  QString qsFilename = qsClassDir + "/trainlist_pidx" + padZeros(QString::number(pidx), 4) + "_tidx_" + padZeros(QString::number(tidx),4) + ".al";
	  annolistClusters[tidx].saveXML(qsFilename.toStdString().c_str(), false);
	}
      }
      else 
	annolistClusters.push_back(annolist);
      
      for (uint tidx = 0; tidx < annolistClusters.size(); ++tidx){
	AnnotationList annolistClus = annolistClusters[tidx];
	uint nImages = annolistClus.size();

	double rot_x_axis = 0.0;
	if (rot_clus.size() > 0)
	  rot_x_axis = rot_clus[tidx];
	cout << "rot_x_axis: " << rot_x_axis << endl;
	
	PartWindowParam::PartParam *pPartParam = windowparam.add_part();
	pPartParam->set_part_id(partconf.part(pidx).part_id());
	pPartParam->set_window_size_x(0);
	pPartParam->set_window_size_y(0);
	pPartParam->set_pos_offset_x(0);
	pPartParam->set_pos_offset_y(0);
	pPartParam->set_type_id(tidx);
	pPartParam->set_root_offset_x(0);
	pPartParam->set_root_offset_y(0);

	if (partconf.part(pidx).is_detect()) {
	  double sum_window_size_x = 0;
	  double sum_window_size_y = 0;
	  double sum_pos_offset_x = 0;
	  double sum_pos_offset_y = 0;
	  // Leonid: learn offsets of poselets w.r.t. the root part
	  double sum_root_offset_x = 0;
	  double sum_root_offset_y = 0;
	  double sum_root_rot = 0;
	  vector<double> root_offset_x;
	  vector<double> root_offset_y;
	  vector<double> root_rot;

	  int nAnnoRects = 0;
	  int nRootPartOffsets = 0;
	  
	  for (int imgidx = 0; imgidx < nImages; ++imgidx) {
	    int nRects = annolistClus[imgidx].size(); 
	    
	    for (int ridx = 0; ridx < nRects; ++ridx) {
	      
	      if (annorect_has_part(annolistClus[imgidx][ridx], partconf.part(pidx))) {
		PartBBox bbox;
		get_part_bbox(annolistClus[imgidx][ridx], partconf.part(pidx), bbox);

		if (fabs(norm_2(bbox.part_x_axis) - 1.0) < 1e-6 && 
		    fabs(norm_2(bbox.part_y_axis) - 1.0) < 1e-6) {
		  
		  sum_window_size_x += (bbox.max_proj_x - bbox.min_proj_x);
		  sum_window_size_y += (bbox.max_proj_y - bbox.min_proj_y);
		  sum_pos_offset_x += bbox.min_proj_x;
		  sum_pos_offset_y += bbox.min_proj_y;
		  
		  ++nAnnoRects;

		  if (annorect_has_part(annolistClus[imgidx][ridx], partconf.part(rootpart_idx))) {
		    
		    PartBBox root_bbox;
		    get_part_bbox(annolistClus[imgidx][ridx], partconf.part(rootpart_idx), root_bbox);
		    
		    if (fabs(norm_2(root_bbox.part_x_axis) - 1.0) < 1e-6 && 
			fabs(norm_2(root_bbox.part_y_axis) - 1.0) < 1e-6) {
		      
		      double offset_x = bbox.part_pos(0) - root_bbox.part_pos(0);
		      double offset_y = bbox.part_pos(1) - root_bbox.part_pos(1);
		      double rot = atan2(root_bbox.part_x_axis(1), root_bbox.part_x_axis(0))/3.14*180;
		      
		      sum_root_offset_x += offset_x;
		      sum_root_offset_y += offset_y;
		      sum_root_rot += rot;
		      
		      root_offset_x.push_back(offset_x);
		      root_offset_y.push_back(offset_y);
		      root_rot.push_back(rot);
		      
		      ++nRootPartOffsets;
		    }
		    
		  }
		  else {
		    cout << "Warning: found degenerated part" << endl;
		  }
		}
	      }// has part
	    }// rects
	    
	  }// images
	  
	  assert(nAnnoRects > 0);
	  cout << "processed rects: " << nAnnoRects << endl;

	  pPartParam->set_window_size_x((int)(sum_window_size_x/nAnnoRects));
	  pPartParam->set_window_size_y((int)(sum_window_size_y/nAnnoRects));

	  /* average offset of part position with respect to top/left corner */
	  pPartParam->set_pos_offset_x((int)(-sum_pos_offset_x/nAnnoRects));
	  pPartParam->set_pos_offset_y((int)(-sum_pos_offset_y/nAnnoRects));
	  
	  /* average offset of part center w.r.t. the root center*/
	  pPartParam->set_root_offset_x((int)(sum_root_offset_x/nRootPartOffsets));
	  pPartParam->set_root_offset_y((int)(sum_root_offset_y/nRootPartOffsets));
	  pPartParam->set_root_rot((float)(sum_root_rot/nRootPartOffsets));
	  
	  double_matrix root_offset(root_offset_x.size(),3);
	  for (int i = 0; i < root_offset_x.size();i++){
	    root_offset(i,0) = root_offset_x[i];
	    root_offset(i,1) = root_offset_y[i];
	    root_offset(i,2) = root_rot[i];
	  }
	  QString qsFilename = exp_param.class_dir().c_str() + QString("/root_offset_pidx_") + 
	    QString::number(pidx) + QString("_tidx_") + QString::number(tidx) + ".mat";
	  cout << "saving " << qsFilename.toStdString() << endl;
	  MATFile *f = matlab_io::mat_open(qsFilename, "wz");
	  assert(f != 0);
	  matlab_io::mat_save_double_matrix(f, "root_offset", root_offset);
	}
      }// part types
    }// parts
    
    /** compute offset of the root part with respect to bounding box center */

    cout  << "computing offset of the root part " << endl;
    
    double_vector bbox_offset = boost_math::double_zero_vector(2);

    int total_rects = 0;
    for (int imgidx = 0; imgidx < nImagesTotal; ++imgidx) {

      for (uint ridx = 0; ridx < annolist[imgidx].size(); ++ridx) {
       
        if (annorect_has_part(annolist[imgidx][ridx], partconf.part(rootpart_idx))) {
        //if (bHasAllParts) {
          PartBBox bbox;
          get_part_bbox(annolist[imgidx][ridx], partconf.part(rootpart_idx), bbox);
      
	  if (fabs(norm_2(bbox.part_x_axis) - 1.0) < 1e-6 && 
	      fabs(norm_2(bbox.part_y_axis) - 1.0) < 1e-6) {

	    double_vector bbox_center(2);
	    bbox_center(0) = 0.5*(annolist[imgidx][ridx].left() + annolist[imgidx][ridx].right());
	    bbox_center(1) = 0.5*(annolist[imgidx][ridx].top() + annolist[imgidx][ridx].bottom());

	    bbox_offset += bbox_center - bbox.part_pos;
	    ++total_rects;
	  }
	  else {
	    cout << "Warning: found degenerated part" << endl;
	  }

        }
      }
    }

    bbox_offset /= total_rects;

    windowparam.set_bbox_offset_x(bbox_offset(0));
    windowparam.set_bbox_offset_y(bbox_offset(1));
  }

  bool compare_vec_desc(vector<float> a, vector<float> b){
    return (a[1] > b[1]);
  }
  
  bool compare_vec_asc(vector<float> a, vector<float> b){
    return (a[1] < b[1]);
  }
  
  void get_part_type_clusters(const AnnotationList &annolist, const PartConfig &partconf, const ExpParam &exp_param,
			      std::vector<AnnotationList> &clusterlist, int pidx){

    cout << "\nCluster examples for pidx: " << pidx << "\n" << endl;
    
    int max_num_part_types = 1;
    if (partconf.part(pidx).has_max_num_part_types())
      max_num_part_types = partconf.part(pidx).max_num_part_types();
    
    cout << "max_num_part_types: " << max_num_part_types << endl;    
    
    int annopoint_num = partconf.part(pidx).part_pos().size();
    int annopoint_idx[annopoint_num];
    
    cout << "Annopoints:";
    for(int i=0;i<annopoint_num;i++){
      annopoint_idx[i] = partconf.part(pidx).part_pos().Get(i);
      cout << " " << annopoint_idx[i];
    }
    cout << endl;
    
    // default parameters
    bool bNormalize = false;
    int ndim_joint = 2;
    int ndim_aspect_ratio = 0;
    int ndim_vp = 0;
    float vis_weight = 1;
    float vp_weight = 0;
    float rejection_rate = 0;
    float min_clus_size = 10;
    
    bNormalize = exp_param.normalize_kmeans();
    ndim_joint = exp_param.ndim_joint();
    min_clus_size = exp_param.min_clus_size();
    vis_weight = exp_param.vis_weight();
    vp_weight = exp_param.vp_weight();
    
    
    if (vp_weight > 0)
      ndim_vp = 1;
    
    cout << "bNormalize: " << bNormalize << endl;
    cout << "ndim_joint: " << ndim_joint << endl;
    cout << "ndim_aspect_ratio: " << ndim_aspect_ratio << endl;
    cout << "ndim_vp: " << ndim_vp << endl;
    cout << "vis_weight: " << vis_weight << endl;
    cout << "rejection_rate: " << rejection_rate << endl;
    
    assert(ndim_joint == 2 || ndim_joint == 3);
    assert(ndim_aspect_ratio == 0 || ndim_aspect_ratio == 2);
    
    int dim_num = ndim_joint*annopoint_num + ndim_aspect_ratio + ndim_vp;
    int cluster_num = max_num_part_types;
    int it_max = 100;
    int it_num;
    int point_num = 0;
    
    // count the total number of annotated parts
    for (uint imgidx = 0; imgidx < annolist.size(); ++imgidx)
      for (uint ridx = 0; ridx < 1; ++ridx) //annolist[imgidx].size()
      if (annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx)))
	point_num++;
    
    double point[point_num*dim_num];
            
    int p_num = 0;
    int idx_refannopoint = partconf.part(pidx).ref_pos();
    int mapping[point_num];
    
    int id = 0;
    // collect the points for k-means clustering
    for (uint imgidx = 0; imgidx < annolist.size(); ++imgidx) 
      for (uint ridx = 0; ridx < 1; ++ridx){ //annolist[imgidx].size()
	if (annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx))){
	  
	  PartBBox bbox;
	  get_part_bbox(annolist[imgidx][ridx], partconf.part(pidx), bbox);
	  /*
	  cout << pidx << "; bbox.part_pos(0): " << bbox.part_pos(0) << "; bbox.part_pos(1): " << bbox.part_pos(1) << "; bbox.min_proj_x: " << bbox.min_proj_x << ", bbox.max_proj_x: " << bbox.max_proj_x << 
	    ", bbox.min_proj_y: " << bbox.min_proj_y << ", bbox.max_proj_y: " << bbox.max_proj_y << endl;
	  getchar();
	  */
	  const AnnoPoint* ref_annopoint = annolist[imgidx][ridx].get_annopoint_by_id(idx_refannopoint);
	  assert(ref_annopoint != NULL);
	  
	  float diff_x =  ref_annopoint->x - bbox.part_pos(0);
	  float diff_y =  ref_annopoint->y - bbox.part_pos(1);
	  
	  float ref_x = bbox.part_pos(0) + diff_x*bbox.part_x_axis(0) - diff_y*bbox.part_y_axis(0);
	  float ref_y = bbox.part_pos(1) - diff_x*bbox.part_x_axis(1) + diff_y*bbox.part_y_axis(1);
	  
	  for(int p=0; p<annopoint_num;p++){
	    const AnnoPoint *annopoint = annolist[imgidx][ridx].get_annopoint_by_id(annopoint_idx[p]);
	    assert(annopoint != NULL);
	    
	    diff_x =  annopoint->x - bbox.part_pos(0);
	    diff_y =  annopoint->y - bbox.part_pos(1);
	    
	    float cur_x = bbox.part_pos(0) + diff_x*bbox.part_x_axis(0) - diff_y*bbox.part_y_axis(0);
	    float cur_y = bbox.part_pos(1) - diff_x*bbox.part_x_axis(1) + diff_y*bbox.part_y_axis(1);
	    
	    point[p_num*dim_num + p*ndim_joint    ] = annopoint->x - ref_annopoint->x;
	    point[p_num*dim_num + p*ndim_joint + 1] = annopoint->y - ref_annopoint->y;
	    
	    if (ndim_joint == 3)
	      point[p_num*dim_num + p*ndim_joint + 2] = vis_weight*annopoint->is_visible;
	    //delete annopoint;
	  }
	  if (ndim_aspect_ratio == 2 && ndim_vp == 0){
	    point[p_num*dim_num + annopoint_num*ndim_joint    ] = annolist[imgidx][ridx].x2() - annolist[imgidx][ridx].x1();
	    point[p_num*dim_num + annopoint_num*ndim_joint + 1] = annolist[imgidx][ridx].y2() - annolist[imgidx][ridx].y1();
	    assert(false);
	  }
	  else if (ndim_aspect_ratio == 0 && ndim_vp == 1){
	    point[p_num*dim_num + annopoint_num*ndim_joint] = annolist[imgidx][ridx].silhouetteID()*vp_weight;
	    assert(false);
	  }
	  else if (ndim_aspect_ratio == 2 && ndim_vp == 1){
	    point[p_num*dim_num + annopoint_num*ndim_joint    ] = annolist[imgidx][ridx].x2() - annolist[imgidx][ridx].x1();
	    point[p_num*dim_num + annopoint_num*ndim_joint + 1] = annolist[imgidx][ridx].y2() - annolist[imgidx][ridx].y1();
	    point[p_num*dim_num + annopoint_num*ndim_joint + 2] = annolist[imgidx][ridx].silhouetteID()*vp_weight;
	    assert(false);
	  }
	  mapping[p_num] = id;
	  p_num++;
	}
	else{
	  cout << "WARNING!" << endl;
	  cout << "Annorect contains no part" << endl;
	}
	id++;
      }//ridx

    assert(p_num == point_num);
    
    if (bNormalize){// mean/variance normalizaition
      assert(false);
      double* mean = (double*)malloc(sizeof(double)*dim_num);
      double* sigma = (double*)malloc(sizeof(double)*dim_num);
      memset(mean, 0, sizeof(double)*dim_num);
      memset(sigma, 0, sizeof(double)*dim_num);
      
      for (uint annoidx = 0; annoidx < point_num; annoidx++)
	for(int d=0; d<dim_num;d++)
	  mean[d] += 1.0/point_num*point[annoidx*dim_num + d];
      
      for (uint annoidx = 0; annoidx < point_num; annoidx++)
	for(int d=0; d<dim_num;d++)
	  sigma[d] += (point[annoidx*dim_num + d] - mean[d])*(point[annoidx*dim_num + d] - mean[d]);
      
      for(int d=0; d<dim_num;d++)
	sigma[d] = max(sqrt(sigma[d]/(point_num-1)), 1e-3);
      
      for (uint annoidx = 0; annoidx < point_num; annoidx++){
	for(int d=0; d<dim_num;d++)
	  point[annoidx*dim_num + d] = (point[annoidx*dim_num + d] - mean[d])/sigma[d];
      }
      delete mean; delete sigma;
    }
    
    cout << "k-means parameters:" << endl;
    cout << "annopoint_num: " << annopoint_num << endl;
    cout << "dim_num: " << dim_num << endl;
    cout << "point_num: " << point_num << endl;
    cout << "cluster_num: " << cluster_num << endl;
    cout << "it_max: " << it_max << endl;
    cout << "min_clus_size: " << min_clus_size << endl;
            
    int cluster[point_num];
    double cluster_center[dim_num*cluster_num];
    int cluster_population[cluster_num];
    double cluster_energy[cluster_num];
    
    cout << "\nRunning k-means... ";
    kmeans_03 (dim_num, point_num, cluster_num, it_max, 
	       it_num, point, cluster, cluster_center, 
	       cluster_population, cluster_energy);
    cout << "done after " << it_num << " iterations\n" << endl;
    
    for(int cidx = 0; cidx<cluster_num;cidx++){
      if (cluster_population[cidx] >= min_clus_size){
	// distribute annotations according to the cluster assignment
	AnnotationList annolistClus;
	for(int a = 0; a<point_num;a++){
	  if (cluster[a] == cidx){

	    int scale = 1.0;
	    int annoidx = mapping[a];
	    Annotation annotation = annolist[annoidx];
	    for (int ridx = 0; ridx < annotation.size(); ridx++){
	      PartBBox bbox;
	      get_part_bbox(annotation[ridx], partconf.part(pidx), bbox);
	      annotation[ridx].setCoords(bbox.x1, bbox.y1, bbox.x2, bbox.y2);
	    }
	    annolistClus.addAnnotation(annotation);
	  }
	}
	clusterlist.push_back(annolistClus);
      }
      else
	cout << "Examples in cluster " << cidx << ": " << cluster_population[cidx] << " < min_clusSize = " << min_clus_size << endl;
    }
    
    p_num = -1;
    cout << endl;
    
    int clusidx = 0;
    for(int cidx = 0; cidx<cluster_num;cidx++){
      if (cluster_population[cidx] >= min_clus_size){
	double min_dist = numeric_limits<double>::infinity();
	int medoid_idx = -1;
	int imgidx = -1;
	vector<vector<float> > distList(cluster_population[cidx],vector<float>(2));
	for(int a = 0; a<point_num;a++){
	  if (cluster[a] == cidx){
	    imgidx++;
	    double dist = 0;
	    for(int d = 0; d < dim_num;d++){
	      dist += (point[a*dim_num + d] - cluster_center[cidx*dim_num + d])*
		(point[a*dim_num + d] - cluster_center[cidx*dim_num + d]);
	    }
	    dist = sqrt(dist);
	    distList[imgidx][0] = imgidx;
	    distList[imgidx][1] = dist;
	    if (min_dist > dist){
	      min_dist = dist;
	      medoid_idx = imgidx;
	    }
	    
	  }// if
	}// points
	sort(distList.begin(), distList.end(), compare_vec_asc);
	
	cout << "cluster: " << cidx << ", medoidId: " << distList[0][0] << endl; 
	cout << "cluster size: " << clusterlist[clusidx].size() << endl;
	cout << "minDist: " << distList[0][1] << ", maxDist: " << distList[distList.size()-1][1] << endl;
	assert(abs(min_dist - distList[0][1]) < 1e-3);
	
	clusterlist[clusidx][medoid_idx][0].m_nObjectId = 100000;
	
	// reject portion of points far from the cluster center
	if (rejection_rate > 0){
	  cout << "Rejecting border points..." << endl;
	  AnnotationList annolistNew;
	  int lastpidx = (int)(distList.size()*(1.0-rejection_rate));
	  for(int p = 0; p < lastpidx; p++)
	    annolistNew.addAnnotation(clusterlist[clusidx][distList[p][0]]);
	  if (annolistNew.size() > 0){
	    clusterlist[clusidx] = annolistNew;
	    cout << "cluster size: " << clusterlist[clusidx].size() << endl;
	    cout << "minDist: " << distList[0][1] << ", maxDist: " << distList[lastpidx-1][1] << endl;
	  }
	  else{
	    cout << "WARNING! annolistNew.size() == 0"  << endl;
	    cout << "Preserving all examples" << endl;
	  }
	}
	clusidx++;
      } 
    }
  }

  int getPartById(const PartWindowParam &windowparam, int pidx, int tidx){
    
    if (tidx > -1){
      for(int i=0; i<windowparam.part_size(); ++i)
	if (windowparam.part(i).part_id() - 1 == pidx && 
	    windowparam.part(i).type_id() == tidx)
	  return i;
    }
    else
      for(int i=0; i<windowparam.part_size(); ++i)
	if (windowparam.part(i).part_id() - 1 == pidx)
	  return i;
	  
    return -1;
    //assert(false && "part not found");
  }
  
  int getNumPartTypes(const PartWindowParam &windowparam, int pidx){
    
    int numPartTypes = 0;
    for(int i=0; i<windowparam.part_size(); ++i)
      if (windowparam.part(i).part_id() - 1 == pidx)
	numPartTypes++;
    
    return numPartTypes;
  }
  
  void loadPartTypeData(QString qsClassDir, const PartWindowParam &windowparam, const PartConfig &partconf, int pidx,
			vector<AnnotationList> &annolistClusters, QString qsListName){
    
    int tidx = -1;
    int nTypes = 0;
    while(getPartById(windowparam, pidx, ++tidx) > -1)
      nTypes++;
    annolistClusters.resize(nTypes);
    
    for (int tidx = 0; tidx < annolistClusters.size(); tidx++){
      QString qsFilename = qsClassDir + "/" + qsListName + "list_pidx" + padZeros(QString::number(pidx), 4) + "_tidx_" + padZeros(QString::number(tidx),4) + ".al";
      AnnotationList annolist;
      if (filesys::check_file(qsFilename.toStdString().c_str()))
	annolist.load(qsFilename.toStdString());
      else if (!(qsListName.compare(QString("test")) == 0)){
	assert(false);
      }
      annolistClusters[tidx] = annolist;
    }
    
  }
  
  void saveCombResponces(const PartApp &part_app, int imgidx, QString qsDetRespDir, QString qsScoreGridDir, 
			 QString qsImgName, int root_pos_x, int root_pos_y, float root_rot, 
			 std::vector<std::vector<FloatGrid3> > &part_detections, bool bLoadScoreGrid)
  {
    
    cout << "saveCombResponces()" << endl;
    
    int nParts = part_app.m_part_conf.part_size();
    int nPartsTotal = part_app.m_window_param.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    
    int img_width, img_height;
    {
      kma::ImageContent *kmaimg = kma::load_convert_gray_image(qsImgName.toStdString().c_str());
      
      assert(kmaimg != 0);
      img_width = kmaimg->x();
      img_height = kmaimg->y();
      delete kmaimg;
    }
    
    cout << "nParts: " << nParts << endl;
    cout << "nScales: " << nScales << endl;
    cout << "nRotations: " << nRotations << endl;
    cout << "img_height: " << img_height << endl;
    cout << "img_width: " << img_width << endl;
    cout << "imgidx: " << imgidx << endl;
    cout << "bLoadScoreGrid: " << bLoadScoreGrid << endl;
    
    if (!filesys::check_dir(qsDetRespDir))
      filesys::create_dir(qsDetRespDir);

    // assume parts with multiple types come after atomic parts
    int nAtomicParts = 0;
    for (int pidx = 0; pidx < nParts; ++pidx)
      if (getNumPartTypes(part_app.m_window_param, pidx) < 2)
	nAtomicParts++;
    
    int nTopBestAcrossPartTypeVis = 5;
    int nTopBestWithinPartTypeVis = 1;
    int nTopSuperPartTypes = 3;
    int nTopBestWithinPartType = 1;
    int minNumPos = 50;
    
    if (part_app.m_exp_param.has_num_best_part_det_vis())
      nTopBestWithinPartTypeVis = part_app.m_exp_param.num_best_part_det_vis();
    
    if (nTopBestWithinPartTypeVis > nTopBestWithinPartType)
      nTopBestWithinPartTypeVis = nTopBestWithinPartType;
    
    boost_math::double_matrix mxDetResp(nPartsTotal*nTopBestWithinPartType,10);
    map<const int, PartBBox> superPartBounds;
    
    vector<vector<PartBBox> > vtBestBBoxList(nParts, vector<PartBBox>(nTopBestAcrossPartTypeVis*nTopBestWithinPartTypeVis));
    vector<vector<int> > vtBestPartTypeList(nParts, vector<int>(nTopBestAcrossPartTypeVis*nTopBestWithinPartTypeVis));
    int pidxAll = -1;
    
    for (int pidx = 0; pidx < nParts; ++pidx) {
      
      int num_part_types = getNumPartTypes(part_app.m_window_param, pidx);
      vector<vector<float> > vtBestDetResp(num_part_types*nTopBestWithinPartTypeVis, vector<float>(7));

      if (part_app.m_part_conf.part(pidx).is_detect()) {
	
	for (int tidx = 0; tidx < num_part_types; ++tidx){
	  
	  double_matrix root_offset;
	  
	  pidxAll++;

	  vector<vector<FloatGrid2> > cur_part_detections;
	  	  
	  if (bLoadScoreGrid){
	    
	    bool bInterpolate = false;
	    if (part_app.m_exp_param.has_interpolate())
	      bInterpolate = part_app.m_exp_param.interpolate();
	    
	    cout << "bInterpolate: " << bInterpolate << endl;
	    
	    bool flip = false;
	    
	    part_app.loadScoreGrid(cur_part_detections, imgidx, pidxAll, flip, bInterpolate, qsScoreGridDir, qsImgName);
	    
	    assert((int)cur_part_detections.size() == nScales);
	    assert((int)cur_part_detections[0].size() == nRotations);
	  }
	  else{
	    cur_part_detections.resize(nScales, vector<FloatGrid2>(nRotations, FloatGrid2(boost::extents[img_height][img_width])));
	    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx)
	      for (int rotidx = 0; rotidx < nRotations; ++rotidx) 
		cur_part_detections[scaleidx][rotidx] = part_detections[pidxAll][scaleidx][rotidx]; 
	  }
	  
	  vector<vector<float> > all_detections(nScales*nRotations*img_width*img_height,vector<float>(5));
	  int idx = -1;
	  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
	    for (int rotidx = 0; rotidx < nRotations; ++rotidx)
	      for (int iy0 = 0; iy0 < img_height; ++iy0) 
		for (int ix0 = 0; ix0 < img_width; ++ix0){
		  idx++;
		  all_detections[idx][0] = 0;//-numeric_limits<float>::infinity();
		  all_detections[idx][1] = 0;
		  all_detections[idx][2] = 0;
		  all_detections[idx][3] = 0;
		  all_detections[idx][4] = 0;
		}
	  
	  int firstix = 0, firstiy = 0, lastix = img_width, lastiy = img_height;
	  int pidx_window_param = getPartById(part_app.m_window_param, pidx, tidx);
	  if (part_app.m_part_conf.part(pidx).has_mult_types()&&part_app.m_exp_param.has_poselet_strip()){
	    int offset_x = part_app.m_window_param.part(pidx_window_param).root_offset_x();
	    int offset_y = part_app.m_window_param.part(pidx_window_param).root_offset_y();
	    int det_pos_x = offset_x + root_pos_x;
	    int det_pos_y = offset_y + root_pos_y;
	    int delta = part_app.m_exp_param.poselet_strip()/2;
	    //cout << "pidx: " << pidx << ", tidx: " << tidx << "; root: " << root_pos_x << " " << root_pos_y << "; offset: " << offset_x << " " << offset_y << endl;
	    firstix = max(firstix, det_pos_x - delta);
	    firstix = min(firstix, img_width);
	    lastix = min(lastix, det_pos_x + delta);
	    lastix = max(lastix, 0);
	    firstiy = max(firstiy, det_pos_y - delta);
	    firstiy = min(firstiy, img_height);
	    lastiy = min(lastiy, det_pos_y + delta);
	    lastiy = max(lastiy, 0);
	  }
	  int firstrotidx = 0, lastrotidx = nRotations;
	  if (part_app.m_part_conf.part(pidx).has_mult_types()&&part_app.m_exp_param.has_poselet_rot_strip()){
	    
	    float root_rot_tidx = part_app.m_window_param.part(pidx_window_param).root_rot();
	    
	    int delta = part_app.m_exp_param.poselet_rot_strip()/2;
	    float dist1 = abs(root_rot_tidx - root_rot);
	    float dist2 = abs(360 - dist1);
	    float angleDist = dist1 < dist2 ? dist1 : dist2;
	    if (angleDist > delta){
	      firstrotidx = 0;
	      lastrotidx = 0;
	    }
	    cout << "pidx: " << pidx << ", tidx: " << tidx << "; root_rot: " << root_rot << "; root_rot_tidx: " << root_rot_tidx << "; angleDist: " << angleDist << endl;
	  }
	  
	  idx = -1;
	  float max_score = -numeric_limits<float>::infinity();
	  int best_rotidx = - 1, best_iy = -1, best_ix = -1, best_scaleidx = -1;
	  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx)
	    for (int rotidx = firstrotidx; rotidx < lastrotidx; ++rotidx){
	      
	      PartBBox bbox;
	      // strip part detections
	      double scale = scale_from_index(part_app.m_exp_param, scaleidx);
	      int pos_offset_x = part_app.m_window_param.part(pidx_window_param).pos_offset_x();
	      int pos_offset_y = part_app.m_window_param.part(pidx_window_param).pos_offset_y();
	      int ext_x_pos = 0;//part_app.m_part_conf.part(pidx).ext_x_pos();
	      int ext_y_pos = 0;//part_app.m_part_conf.part(pidx).ext_y_pos();
	      int fix = scale*(pos_offset_x - ext_x_pos);
	      int lix = img_width - scale*(pos_offset_x - ext_x_pos);
	      int fiy = scale*(pos_offset_y - ext_y_pos);
	      int liy = img_height - scale*(pos_offset_y - ext_y_pos);
	      
	      firstix = max(firstix, fix);
	      lastix =  min(lastix,  lix);
	      firstiy = max(firstiy, fiy);
	      lastiy =  min(lastiy,  liy);
	      
	      for (int iy0 = firstiy; iy0 < lastiy; ++iy0) 
		for (int ix0 = firstix; ix0 < lastix; ++ix0){
		  idx++;

		  float score = cur_part_detections[scaleidx][rotidx][iy0][ix0];
		  if (score < 0)
		    score = 0;
		    
		  all_detections[idx][0] = rotidx;
		  all_detections[idx][1] = score;
		  all_detections[idx][2] = iy0;
		  all_detections[idx][3] = ix0;
		  all_detections[idx][4] = scaleidx;

		}
	    }
	  sort(all_detections.begin(), all_detections.end(), compare_vec_desc);
	  //}// min_num_pos

	  for (uint topidx = 0; topidx < nTopBestWithinPartType; ++topidx){
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,0) = pidx;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,1) = tidx;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,2) = all_detections[topidx][1];//max_score;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,3) = all_detections[topidx][0];//best_rotidx;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,4) = all_detections[topidx][2];//best_iy;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,5) = all_detections[topidx][3];//best_ix;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,6) = all_detections[topidx][4];//best_scaleidx;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,7) = img_width;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,8) = img_height;
	    mxDetResp(pidxAll*nTopBestWithinPartType + topidx,9) = nRotations;
	  }
	  
	  for (uint topidx = 0; topidx < nTopBestWithinPartTypeVis; ++topidx){
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][0] = mxDetResp(pidxAll*nTopBestWithinPartType + topidx,1);
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][1] = mxDetResp(pidxAll*nTopBestWithinPartType + topidx,2);
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][2] = mxDetResp(pidxAll*nTopBestWithinPartType + topidx,3);
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][3] = mxDetResp(pidxAll*nTopBestWithinPartType + topidx,4);
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][4] = mxDetResp(pidxAll*nTopBestWithinPartType + topidx,5);
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][5] = mxDetResp(pidxAll*nTopBestWithinPartType + topidx,6);
	    vtBestDetResp[tidx*nTopBestWithinPartTypeVis + topidx][6] = tidx;
	  }
	}// tidx
      }// if is_detect
      sort(vtBestDetResp.begin(), vtBestDetResp.end(), compare_vec_desc);
      
      int nTopCurr = min(num_part_types*nTopBestWithinPartTypeVis, nTopBestAcrossPartTypeVis*nTopBestWithinPartTypeVis);
      for(int i = 0; i < nTopCurr; i++){
	PartBBox bbox;
	int scaleidx = vtBestDetResp[i][5];
	double scale = scale_from_index(part_app.m_exp_param, scaleidx);
	int pidx_window_param = getPartById(part_app.m_window_param, pidx, vtBestDetResp[i][0]);
	bbox_from_pos(part_app.m_exp_param, part_app.m_window_param.part(pidx_window_param), scaleidx, (int)vtBestDetResp[i][2], (int)vtBestDetResp[i][4], (int)vtBestDetResp[i][3], bbox);
	bbox.min_proj_y += scale * part_app.m_part_conf.part(pidx).ext_y_neg();
	bbox.max_proj_y -= scale * part_app.m_part_conf.part(pidx).ext_y_pos();
	bbox.min_proj_x += scale * part_app.m_part_conf.part(pidx).ext_x_neg();
	bbox.max_proj_x -= scale * part_app.m_part_conf.part(pidx).ext_x_pos();
	bbox.min_proj_x -= scale * part_app.m_part_conf.part(pidx).ext_x_neg_vis();
	bbox.max_proj_x += scale * part_app.m_part_conf.part(pidx).ext_x_pos_vis();
	vtBestBBoxList[pidx][i] = bbox;
	vtBestPartTypeList[pidx][i] = vtBestDetResp[i][6];//tidx
      }
      
    }// pidx
    
    QString qsVisDir = qsDetRespDir + "/vis_resp/";
    if (!filesys::check_dir(qsVisDir))
      filesys::create_dir(qsVisDir);
    
    for (int pidx = 0; pidx < nParts; ++pidx){ 
      
      QImage img_original;
      assert(img_original.load(qsImgName));
      QPainter painter(&img_original);
      painter.setRenderHints(QPainter::Antialiasing);
      painter.setPen(Qt::yellow);
      
      int num_part_types = getNumPartTypes(part_app.m_window_param, pidx);
      int nTopCurr = min(num_part_types*nTopBestWithinPartTypeVis, nTopBestAcrossPartTypeVis*nTopBestWithinPartTypeVis);
      map<const int, int> colorMap;
      int nColors = -1, coloridx = -1;
      for(int i = 0; i < nTopCurr; i++){
	int tidx = vtBestPartTypeList[pidx][i];
	map<int,int>::iterator it = colorMap.find(tidx);
	
	if (it == colorMap.end()){
	  colorMap.insert(pair<const int,int>(tidx,++nColors)); 
	  coloridx = nColors;
	}
	else
	  coloridx = it->second;
	
	// currently can't use more colors
	if (coloridx < 5 && !(vtBestBBoxList[pidx][i].part_pos(0) == 0 && vtBestBBoxList[pidx][i].part_pos(1) == 0))
	  draw_bbox(painter, vtBestBBoxList[pidx][i], coloridx, 2);
      }
      
      QString qsOutImg = qsVisDir + "/imgidx_" + padZeros(QString::number(imgidx), 4) + "_pidx_" + padZeros(QString::number(pidx), 4) + ".png";
      //cout << "saving " << qsOutImg.toStdString() << endl;
      assert(img_original.save(qsOutImg));
    }
    
    QString qsFilename = qsDetRespDir + "/resp_" + padZeros(QString::number(imgidx), 4) + ".mat";
    matlab_io::mat_save_double_matrix(qsFilename, "resp", mxDetResp);
    //cout << "save " << qsFilename.toStdString().c_str() << endl;
    
    cout << "done." << endl;
  }
  
  void covertToPoints(const PartConfig &partconf, AnnotationList &annolist, int pidx, 
		      const vector<int> &annopoint_idx, vector<vector<float> > &points, vector<int> &mapping){
    
    int idx_refannopoint = partconf.part(pidx).ref_pos();
    
    int idx = 0;
    // assume single bbox per image
    for (uint imgidx = 0; imgidx < annolist.size(); ++imgidx) 
      for (uint ridx = 0; ridx < annolist[imgidx].size(); ++ridx){ 
	if (annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx))){
	  
	  PartBBox bbox;
	  get_part_bbox(annolist[imgidx][ridx], partconf.part(pidx), bbox);
	  
	  const AnnoPoint* ref_annopoint = annolist[imgidx][ridx].get_annopoint_by_id(idx_refannopoint);
	  assert(ref_annopoint != NULL);
	  vector<float> cur_anno(annopoint_idx.size()*2);
	  for(int p=0; p<annopoint_idx.size();p++){
	    const AnnoPoint *annopoint = annolist[imgidx][ridx].get_annopoint_by_id(annopoint_idx[p]);
	    assert(annopoint != NULL);
	    cur_anno[p*2    ] = annopoint->x - ref_annopoint->x;
	    cur_anno[p*2 + 1] = annopoint->y - ref_annopoint->y;
	  }
	  points.push_back(cur_anno);
	  mapping.push_back(idx);
	}
	else{
	  cout << "WARNING!" << endl;
	  cout << "Annorect contains no part" << endl;
	  cout << annolist[imgidx].imageName() << endl;
	  //assert(false);
	}
	idx++;
      }
  }
  
  int partdetect_dpm(const PartApp &part_app, int imgidx, QString qsModelDPMDir, QString qsUnaryDPMDir){
    
    cout << "partdetect_dpm()" << endl;
    cout << "imgidx: " << imgidx << endl;
    
    QString qsCMD = "run_partdetect_dpm.sh "  + QString(MATLAB_RUNTIME);
   
    vector<QString> fileNames;
    if (filesys::getFileNames(qsModelDPMDir,"*_final.mat",fileNames))
      assert(fileNames.size() == 1);
    else{
      cout << "File not found" << endl;
      cout << qsModelDPMDir.toStdString().c_str() << "/*_final.mat" << endl;
      return 1;
    }
    
    QString qsModelDPMFilename = qsModelDPMDir + "/" + fileNames[0];
    
    QString qsCommandLine = qsCMD + " " + QString::number(imgidx+1) + " " +
      part_app.m_test_annolist[imgidx].imageName().c_str() + " " + 
      qsModelDPMFilename + " " + qsUnaryDPMDir + " " + QString::number(NUM_THREADS)  + " 1 " + QString::number(RESCALE_FACTOR) + " 1 1";

    runMatlabCode(qsCommandLine);
    
    return 1;

  }

  int partdetect_dpm_all(const PartApp &part_app, int imgidx){
    
    cout << "partdetect_dpm_all()" << endl;
    cout << "imgidx: " << imgidx << endl;
    
    
    QString qsCMD = "run_partdetect_dpm_all.sh "  + QString(MATLAB_RUNTIME);
    
    int nParts = part_app.m_part_conf.part_size();
    QString qsUnaryDPMDir = part_app.m_exp_param.test_dpm_unary_dir().c_str();
    QString qsModelDPMDir = part_app.m_exp_param.dpm_model_dir().c_str();
        
    QString qsCommandLine = qsCMD + " " + QString::number(imgidx+1) + " " + 
      part_app.m_test_annolist[imgidx].imageName().c_str() + " " + 
      QString::number(nParts) + " " + 
      qsModelDPMDir + " " + qsUnaryDPMDir + " " + QString::number(NUM_THREADS) + " 1 " + QString::number(RESCALE_FACTOR) + " 1 1";
    
    runMatlabCode(qsCommandLine);
        
    return 1;

  }
      
  void runMatlabCode(QString qsCommandLine){

    cout << "runMatlabCode()" << endl;
    cout << "qsCommandLine: " << qsCommandLine.toStdString() << endl;

    std::vector<QString> output;

    QObject *parent = NULL;
    QProcess *myProcess = new QProcess(parent);

    const int maxsize = 10000;
    char buff[maxsize];
    
    myProcess->setProcessChannelMode(QProcess::MergedChannels);
    
    myProcess->start(qsCommandLine);

    //myProcess->waitForFinished();
    int nWaitSec = 10000;
    myProcess->waitForFinished(1000*nWaitSec);
    //myProcess->waitForFinished(-1);

    while (myProcess->canReadLine()) {
      myProcess->readLine(buff, maxsize);
      cout << "\t" << buff;

      output.push_back(QString(buff));
    }

    cout << "state: " << myProcess->state() << endl;

    delete myProcess;
    cout << "done" << endl;
    
  }

}// namespace
