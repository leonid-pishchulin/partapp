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

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

#include <libMisc/misc.hpp>
#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <libPartDetect/partdef.h>
#include <libMultiArray/multi_array_op.hpp>
#include <libFilesystemAux/filesystem_aux.h>
#include <libPartApp/partapp_aux.hpp>

#include <libMatlabIO/matlab_io.h>

#include "objectdetect.h"

using std::cout;
using std::endl;

using boost_math::double_matrix;
using boost_math::double_vector;
using boost_math::double_zero_vector;
using boost_math::double_zero_matrix;

namespace object_detect {
  Joint::Joint(int _type, int _child_idx, int _parent_idx, 
               boost_math::double_vector _offset_c, 
               boost_math::double_vector _offset_p, 
               boost_math::double_matrix _C) : type(_type), child_idx(_child_idx), parent_idx(_parent_idx), 
                                               offset_c(_offset_c), offset_p(_offset_p), C(_C), rot_mean(0), rot_sigma(0)
  {}
  
  Joint::Joint(int _type, int _child_idx, int _parent_idx, 
               boost_math::double_vector _offset_c, 
               boost_math::double_vector _offset_p, 
               boost_math::double_matrix _C, 
               double _rot_mean, 
               double _rot_sigma) : type(_type), child_idx(_child_idx), parent_idx(_parent_idx), 
                                    offset_c(_offset_c), offset_p(_offset_p), C(_C), rot_mean(_rot_mean), rot_sigma(_rot_sigma)
  {}

  void save_joint(const PartApp &part_app, Joint &joint, int tidx)
  {
    
    assert(part_app.m_exp_param.spatial_dir().length() > 0);
    assert(filesys::check_dir(part_app.m_exp_param.spatial_dir().c_str()));

    // Leonid: support mixtures of pairwise terms
    QString qsTypeSuff = (tidx > -1? "_tidx_" + QString::number(tidx) : "");
    
    QString qsFilename = part_app.m_exp_param.spatial_dir().c_str() + QString("/joint_") +
      QString::number(joint.child_idx) + "_" + 
      QString::number(joint.parent_idx) + qsTypeSuff + ".mat";

    cout << "saving " << qsFilename.toStdString() << endl;
    MATFile *f = matlab_io::mat_open(qsFilename, "wz");
    assert(f != 0);

    matlab_io::mat_save_double(f, "type", joint.type);
    matlab_io::mat_save_double(f, "child_idx", joint.child_idx);
    matlab_io::mat_save_double(f, "parent_idx", joint.parent_idx);
    
    matlab_io::mat_save_double_vector(f, "offset_c", joint.offset_c);
    matlab_io::mat_save_double_vector(f, "offset_p", joint.offset_p);
    
    matlab_io::mat_save_double_matrix(f, "C", joint.C);

    matlab_io::mat_save_double(f, "rot_mean", joint.rot_mean);
    matlab_io::mat_save_double(f, "rot_sigma", joint.rot_sigma);
    matlab_io::mat_close(f);
  }

  void load_joint(const PartApp &part_app, int jidx, Joint &joint, int tidx) 
  {
    assert(jidx < part_app.m_part_conf.joint_size());
    assert(part_app.m_exp_param.spatial_dir().length() > 0);
    assert(filesys::check_dir(part_app.m_exp_param.spatial_dir().c_str()));

    int child_idx = part_app.m_part_conf.joint(jidx).child_idx();
    int parent_idx = part_app.m_part_conf.joint(jidx).parent_idx();

    // Leonid: support mixtures of pairwise terms
    QString qsTypeSuff = (tidx > -1? "_tidx_" + QString::number(tidx) : "");
    QString qsFilenameMix = part_app.m_exp_param.spatial_dir().c_str() + QString("/joint_") +
      QString::number(child_idx) + "_" + 
      QString::number(parent_idx) + qsTypeSuff + ".mat";
    
    QString qsFilenameGen = part_app.m_exp_param.spatial_dir().c_str() + QString("/joint_") +
      QString::number(child_idx) + "_" + 
      QString::number(parent_idx) + ".mat";
    
    // Leonid: learn position covariance from the whole data, not from every rotation cluster
    QString qsFilename = qsFilenameMix;
    //QString qsFilename = qsFilenameGen;
    
    if (!filesys::check_file(qsFilenameMix)){
      cout << "\nWarning! Empty joint mixture component!\n" << endl;
      qsFilename = qsFilenameGen;
      //assert(false);
    }
    
    cout << "loading " << qsFilename.toStdString() << endl;

    double d_type;
    double d_child_idx;
    double d_parent_idx;

    double_vector offset_c;
    double_vector offset_p;
    double_matrix C;
    
    matlab_io::mat_load_double(qsFilename, "type", d_type);
    matlab_io::mat_load_double(qsFilename, "child_idx", d_child_idx);
    matlab_io::mat_load_double(qsFilename, "parent_idx", d_parent_idx);
    assert((int)d_child_idx == child_idx && (int)d_parent_idx == parent_idx);
    
    matlab_io::mat_load_double_vector(qsFilename, "offset_c", offset_c);
    matlab_io::mat_load_double_vector(qsFilename, "offset_p", offset_p);
    matlab_io::mat_load_double_matrix(qsFilename, "C", C);

    if ((int)d_type == Joint::POS_GAUSSIAN)
      assert(C.size1() == 2 && C.size2() == 2);

    if ((int)d_type == Joint::ROT_GAUSSIAN) {
      double rot_mean;
      double rot_sigma;
      
      QString qsTypeSuff = (tidx > -1? "_tidx_" + QString::number(tidx) : "");
      QString qsFilenameMix = part_app.m_exp_param.spatial_dir().c_str() + QString("/joint_") +
	QString::number(child_idx) + "_" + 
	QString::number(parent_idx) + qsTypeSuff + ".mat";
    
      QString qsFilenameGen = part_app.m_exp_param.spatial_dir().c_str() + QString("/joint_") +
	QString::number(child_idx) + "_" + 
	QString::number(parent_idx) + ".mat";
    
      QString qsFilename = qsFilenameMix;
      if (!filesys::check_file(qsFilenameMix)){
	cout << "\nWarning! Empty joint mixture component!\n" << endl;
	qsFilename = qsFilenameGen;
	assert(false);
      }
      cout << "loading rot_mean from " << qsFilename.toStdString() << endl;
      
      matlab_io::mat_load_double(qsFilename, "rot_sigma", rot_sigma);
      matlab_io::mat_load_double(qsFilename, "rot_mean", rot_mean);
      
      joint.rot_mean = rot_mean;
      joint.rot_sigma = rot_sigma;
      //joint.rot_sigma = 10000;//rot_sigma;
    }
    joint.mix_comp_id = (tidx > -1? tidx : 0);
    joint.type = (int)d_type;
    joint.child_idx = child_idx;
    joint.parent_idx = parent_idx;
    joint.offset_c = offset_c;
    joint.offset_p = offset_p;
    joint.C = C;
    
  }
  
  void learn_conf_param_jidx(const PartApp &part_app,  const AnnotationList &train_annolist, double window_step, int jidx, 
			     boost::variate_generator<boost::mt19937, boost::uniform_real<> > &gen_01, int tidx = -1)
  {
   
    const PartConfig &part_conf = part_app.m_part_conf;
    if (JointTypeFromString(part_conf.joint(jidx).type().c_str()) == Joint::POS_GAUSSIAN) {
      cout << "joint idx: " << jidx << ", type: " << "Joint::POS_GAUSSIAN" << endl;
      
      double_vector m = boost_math::double_zero_vector(2);
      double_matrix C = boost_math::double_zero_matrix(2, 2);
      
      int child_idx = part_conf.joint(jidx).child_idx() - 1;
      int parent_idx = part_conf.joint(jidx).parent_idx() - 1;
      
      assert(child_idx < part_conf.part_size());
      assert(parent_idx < part_conf.part_size());
      assert(part_conf.part(child_idx).part_id() == child_idx + 1);
      assert(part_conf.part(parent_idx).part_id() == parent_idx + 1);
      
      int N = 0;
      /** estimate mean offset */
      for (int imgidx = 0; imgidx < (int)train_annolist.size(); ++imgidx) {
	for (int ridx = 0; ridx < (int)train_annolist[imgidx].size(); ++ridx) {
	  const AnnoRect &annorect = train_annolist[imgidx][ridx];
	  
	  if (annorect_has_part(annorect, part_conf.part(child_idx)) &&
	      annorect_has_part(annorect, part_conf.part(parent_idx))) {
	    
	    double_vector pos_child = get_part_position(annorect, part_conf.part(child_idx));
	    double_vector pos_parent = get_part_position(annorect, part_conf.part(parent_idx));
	    double_vector d = pos_child - pos_parent;
	    m += d;
	    ++N;
	  }
	  
	}
      }
      m /= N;
      
      /** estimate covariance */
      for (int imgidx = 0; imgidx < (int)train_annolist.size(); ++imgidx) {
	for (int ridx = 0; ridx < (int)train_annolist[imgidx].size(); ++ridx) {
	  const AnnoRect &annorect = train_annolist[imgidx][ridx];
	  
	  if (annorect_has_part(annorect, part_conf.part(child_idx)) &&
	      annorect_has_part(annorect, part_conf.part(parent_idx))) {
	    
	    double_vector pos_child = get_part_position(annorect, part_conf.part(child_idx));
	    double_vector pos_parent = get_part_position(annorect, part_conf.part(parent_idx));
	    double_vector d = (pos_child - pos_parent) - m;
	    C += outer_prod(d, d);
	  }
	  
	}
      }
      
      C /= N;
      
      cout << "number of points: " << N << endl;
      
      cout << "mean: " << endl;
      boost_math::print_vector(m);
      cout << "covariance: " << endl;
      boost_math::print_matrix(C);
      
      Joint joint(Joint::POS_GAUSSIAN, part_conf.joint(jidx).child_idx(), part_conf.joint(jidx).parent_idx(), -m, m, C);
      save_joint(part_app, joint);
    }
    else if (JointTypeFromString(part_conf.joint(jidx).type().c_str()) == Joint::ROT_GAUSSIAN) {
      cout << "joint idx: " << jidx << ", type: " << "Joint::ROT_GAUSSIAN" << endl;
      
      assert(part_conf.joint(jidx).joint_pos_size() > 0);
      int child_idx = part_conf.joint(jidx).child_idx() - 1;
      int parent_idx = part_conf.joint(jidx).parent_idx() - 1;
      
      assert(child_idx < part_conf.part_size());
      assert(parent_idx < part_conf.part_size());
      assert(part_conf.part(child_idx).part_id() == child_idx + 1);
      assert(part_conf.part(parent_idx).part_id() == parent_idx + 1);
      
      double_vector mean_offset_c = double_zero_vector(2);
      double_vector mean_offset_p = double_zero_vector(2); 
      double rot_mean = 0;
      double rot_sigma = 0;
      int N = 0;
      int N2 = 0;
      
      /** compute joint offsets in the child/parent coordinate systems */
      for (int imgidx = 0; imgidx < (int)train_annolist.size(); ++imgidx) {
	
	for (int ridx = 0; ridx < (int)train_annolist[imgidx].size(); ++ridx) {
	  
	  const AnnoRect &annorect = train_annolist[imgidx][ridx];
	  
	  if (annorect_has_part(annorect, part_conf.part(child_idx)) &&
	      annorect_has_part(annorect, part_conf.part(parent_idx))) {
	    
	    /** position of the joint */
	    double_vector joint_pos = double_zero_vector(2);
            
	    for (int idx = 0; idx < part_conf.joint(jidx).joint_pos_size(); ++idx) {
	      //int joint_pos_idx = part_conf.joint(jidx).joint_pos(idx);
	      //assert((int)annorect.m_vAnnoPoints.size() > joint_pos_idx);
	      
	      int joint_pos_id = part_conf.joint(jidx).joint_pos(idx);
              
	      const AnnoPoint *p = annorect.get_annopoint_by_id(joint_pos_id);
	      assert(p->id == joint_pos_id);
	      
	      //joint_pos(0) += annorect.m_vAnnoPoints[joint_pos_idx].x;
	      //joint_pos(1) += annorect.m_vAnnoPoints[joint_pos_idx].y;
	      
	      joint_pos(0) += p->x;
	      joint_pos(1) += p->y;
	      
	    }
	    joint_pos /= part_conf.joint(jidx).joint_pos_size();
	    
	    /** joint position in child cs */
	    PartBBox bbox_child;
	    get_part_bbox(annorect, part_conf.part(child_idx), bbox_child);
	    
	    //assert(fabs(norm_2(bbox_child.part_x_axis) - 1.0) < 1e-6);
	    //assert(fabs(norm_2(bbox_child.part_y_axis) - 1.0) < 1e-6);
	    
	    /** joint position in parent cs */
	    PartBBox bbox_parent;
	    get_part_bbox(annorect, part_conf.part(parent_idx), bbox_parent);
	    
	    //assert(fabs(norm_2(bbox_parent.part_x_axis) - 1.0) < 1e-6);
	    //assert(fabs(norm_2(bbox_parent.part_y_axis) - 1.0) < 1e-6);
	    
	    if (fabs(norm_2(bbox_child.part_x_axis) - 1.0) < 1e-6 && fabs(norm_2(bbox_child.part_y_axis) - 1.0) < 1e-6 && 
		fabs(norm_2(bbox_parent.part_x_axis) - 1.0) < 1e-6 && fabs(norm_2(bbox_parent.part_y_axis) - 1.0) < 1e-6) {
	      
	      double_vector offset_c(2);
	      offset_c(0) = inner_prod(joint_pos - bbox_child.part_pos, bbox_child.part_x_axis);
	      offset_c(1) = inner_prod(joint_pos - bbox_child.part_pos, bbox_child.part_y_axis);
	      mean_offset_c = mean_offset_c + offset_c;            
	      
	      double_vector offset_p(2);
	      offset_p(0) = inner_prod(joint_pos - bbox_parent.part_pos, bbox_parent.part_x_axis);
	      offset_p(1) = inner_prod(joint_pos - bbox_parent.part_pos, bbox_parent.part_y_axis);
	      mean_offset_p = mean_offset_p + offset_p;
	      
	      //             double parent_rot = atan2(bbox_parent.part_y_axis(1), bbox_parent.part_y_axis(0));
	      //             double child_rot = atan2(bbox_child.part_y_axis(1), bbox_child.part_y_axis(0));
	      
	      double parent_rot = atan2(bbox_parent.part_x_axis(1), bbox_parent.part_x_axis(0));
	      double child_rot = atan2(bbox_child.part_x_axis(1), bbox_child.part_x_axis(0));
	      
	      double joint_rot_offset = (child_rot - parent_rot);
	      
	      //cout << imgidx << " " << offset_c(0) << " " << offset_c(1) << " " << offset_p(0) << " " << offset_p(1) << " " << joint_rot_offset << endl;
	      //getchar();
	      
	      /** when averaging the rotations it actually *does* make sence to have rotations > M_PI */
	      
	      // 		while (joint_rot_offset < -M_PI)
	      // 		  joint_rot_offset += M_PI;
	      
	      // 		while (joint_rot_offset > M_PI)
	      // 		  joint_rot_offset -= M_PI;
	      while (joint_rot_offset < -M_PI)
		joint_rot_offset += 2*M_PI;
	      
	      while (joint_rot_offset > M_PI)
		joint_rot_offset -= 2*M_PI;
	      
	      rot_mean += joint_rot_offset;
	      //rot_mean += (child_rot - parent_rot);
	            
	      ++N;
	    }
	    else {
	      cout << "Warning: found degenerated part" << endl;
	    }
	    
	  }// if has parts
	  
	}//rectangles
	
      }//images
      
      mean_offset_c /= N;
      mean_offset_p /= N;
      rot_mean /= N;
      
      /** estimate variance in position of the joint and relative rotation of parts */
      
      double_matrix jointC = double_zero_matrix(2, 2);
      
      for (int imgidx = 0; imgidx < (int)train_annolist.size(); ++imgidx) {
	for (int ridx = 0; ridx < (int)train_annolist[imgidx].size(); ++ridx) {
	  //cout << "imgidx: " << imgidx << ", ridx: " << ridx << endl;
	  
	  const AnnoRect &annorect = train_annolist[imgidx][ridx];
	  //assert((int)annorect.m_vAnnoPoints.size() > joint_pos_idx);
	  
	  if (annorect_has_part(annorect, part_conf.part(child_idx)) &&
	      annorect_has_part(annorect, part_conf.part(parent_idx))) {
	    
	    PartBBox bbox_child;
	    get_part_bbox(annorect, part_conf.part(child_idx), bbox_child);
	    
	    PartBBox bbox_parent;
	    get_part_bbox(annorect, part_conf.part(parent_idx), bbox_parent);
	    
	    if (fabs(norm_2(bbox_child.part_x_axis) - 1.0) < 1e-6 && fabs(norm_2(bbox_child.part_y_axis) - 1.0) < 1e-6 && 
		fabs(norm_2(bbox_parent.part_x_axis) - 1.0) < 1e-6 && fabs(norm_2(bbox_parent.part_y_axis) - 1.0) < 1e-6) {
	      

	      /* add random offset to part positions to simulate detection on sparse grid */
	      double rnd_offset_x;
	      double rnd_offset_y;
	      rnd_offset_x = window_step * (gen_01() - 0.5);
	      rnd_offset_y = window_step * (gen_01() - 0.5);
	      
	      bbox_child.part_pos(0) += rnd_offset_x;
	      bbox_child.part_pos(1) += rnd_offset_y;
	      
	      
	      double	child_rot = atan2(bbox_child.part_x_axis(1), bbox_child.part_x_axis(0));
	      
	      
	      //             cout << "jidx: " << jidx << ", imgidx: " << imgidx << endl;
	      //             cout << "\tchild_rot: " << child_rot << " (" << deg_from_rad(child_rot) << ") " << endl;
	      //             printf("\t%.10f\n", child_rot); 
	      //             printf("\t%.10f\n", deg_from_rad(child_rot)); 
	      
	      //int child_rot_idx = index_from_rot(part_app.m_exp_param, child_rot/M_PI*180.0);
	      
	      //cout << "child_rot: " << child_rot << ", (" << deg_from_rad(child_rot) << ")" << endl;
	      
	      int child_rot_idx = index_from_rot_clip(part_app.m_exp_param, deg_from_rad(child_rot));
	      double child_rot2 = rad_from_deg(rot_from_index(part_app.m_exp_param, child_rot_idx));
	      
	      //             cout << "\tchild_rot2: " << child_rot2 << " (" << deg_from_rad(child_rot2) << ") " << endl;
	      
	      double_vector child_x_axis2(2);
	      child_x_axis2(0) = cos(child_rot2);
	      child_x_axis2(1) = sin(child_rot2);
	      
	      double_vector child_y_axis2(2);
	      child_y_axis2(0) = -child_x_axis2(1);
	      child_y_axis2(1) = child_x_axis2(0);
	      
	      double_vector joint_pos_child = bbox_child.part_pos + 
		child_x_axis2 * mean_offset_c(0) + 
		child_y_axis2 * mean_offset_c(1);
	      
	      //             double_vector joint_pos_child = bbox_child.part_pos + 
	      //               bbox_child.part_x_axis * mean_offset_c(0) + 
	      //               bbox_child.part_y_axis * mean_offset_c(1);
	      
	      
	      rnd_offset_x = window_step * (gen_01() - 0.5);
	      rnd_offset_y = window_step * (gen_01() - 0.5);
	      bbox_parent.part_pos(0) += rnd_offset_x;
	      bbox_parent.part_pos(1) += rnd_offset_y;
	      
	      double parent_rot = atan2(bbox_parent.part_x_axis(1), bbox_parent.part_x_axis(0));
	      
	      //int parent_rot_idx = index_from_rot(part_app.m_exp_param, parent_rot/M_PI*180.0);
	      int parent_rot_idx = index_from_rot_clip(part_app.m_exp_param, deg_from_rad(parent_rot));
	      double parent_rot2 = rad_from_deg(rot_from_index(part_app.m_exp_param, parent_rot_idx));
	      
	      double_vector parent_x_axis2(2);
	      parent_x_axis2(0) = cos(parent_rot2);
	      parent_x_axis2(1) = sin(parent_rot2);
	      
	      double_vector parent_y_axis2(2);
	      parent_y_axis2(0) = -parent_x_axis2(1);
	      parent_y_axis2(1) = parent_x_axis2(0);
	      
	      double_vector joint_pos_parent = bbox_parent.part_pos + 
		parent_x_axis2 * mean_offset_p(0) + 
		parent_y_axis2 * mean_offset_p(1);
	      
	      //             double_vector joint_pos_parent = bbox_parent.part_pos + 
	      //               bbox_parent.part_x_axis * mean_offset_p(0) + 
	      //               bbox_parent.part_y_axis * mean_offset_p(1);
	      
	      double_vector d = joint_pos_child - joint_pos_parent;
	      jointC += outer_prod(d, d);
	      
	      //             double parent_rot = atan2(bbox_parent.part_y_axis(1), bbox_parent.part_y_axis(0));
	      //             double child_rot = atan2(bbox_child.part_y_axis(1), bbox_child.part_y_axis(0));

	      double joint_rot_offset = (child_rot - parent_rot);
	      
	      while (joint_rot_offset < -M_PI)
		joint_rot_offset += 2*M_PI;
	      
	      while (joint_rot_offset > M_PI)
		joint_rot_offset -= 2*M_PI;
	      

	      //double joint_rot_offset = (child_rot - parent_rot) - rot_mean;
	      joint_rot_offset -= rot_mean;
	      /** same thing as when computing the mean, right ? */
	      
	      // 		while (joint_rot_offset < -M_PI)
	      // 		  joint_rot_offset += M_PI;
	      
	      // 		while (joint_rot_offset > M_PI)
	      // 		  joint_rot_offset -= M_PI;
	      
	      //rot_sigma += square((child_rot - parent_rot) - rot_mean);
	      
	      rot_sigma += square(joint_rot_offset);
	      
	      ++N2;
	    } // if parts are not degenerated
	    else {
	      cout << "Warning: found degenerated part" << endl;
	    }
	    
	  } // if has parts 
	  
	}// rectangles
      }// images
      
      assert(N == N2);
      
      jointC /= N2;
      rot_sigma = sqrt(rot_sigma/N2);
      
      cout << "number of points: " << N << endl;
      
      cout << "joint: " << jidx << endl;
      cout << "mean_offset_c: " << " ";
      boost_math::print_vector(mean_offset_c);
      
      cout << "mean_offset_p: " << " ";
      boost_math::print_vector(mean_offset_p);
      
      cout << "jointC: " << endl;
      boost_math::print_matrix(jointC);
      
      cout << "rot_mean: " << rot_mean << " (" << rot_mean / M_PI * 180.0 << ")" << endl;
      cout << "rot_sigma: " << rot_sigma << " (" << rot_sigma / M_PI * 180.0 << ")" << endl;
      cout << endl;
      
      Joint joint(Joint::ROT_GAUSSIAN, part_conf.joint(jidx).child_idx(), part_conf.joint(jidx).parent_idx(), 
		  mean_offset_c, mean_offset_p, jointC, rot_mean, rot_sigma);
      save_joint(part_app, joint, tidx);
    }
    else {
      assert(false && "unknown joint type");
    }
    
  }

  void learn_conf_param(const PartApp &part_app, const AnnotationList &train_annolist)
  {
    cout << "learn_conf_param" << endl;
    //const PartConfig &part_conf = part_app.m_part_conf;
    
    int nJoints = part_app.m_part_conf.joint_size();

    /* add random offset to part positions to simulate detection on sparse grid */
    boost::mt19937 rng(42u);
    boost::uniform_real<> dist_01(0.0, 1.0);
    boost::variate_generator<boost::mt19937, boost::uniform_real<> > gen_01(rng, dist_01);
    
    double window_step = part_app.m_abc_param.desc_step() * part_app.m_abc_param.window_desc_step_ratio();
    cout << "window_step: " << window_step << endl;

    /* process joints */ 
    for (int jidx = 0; jidx < nJoints; ++jidx) {
      learn_conf_param_jidx(part_app, train_annolist, window_step, jidx, gen_01);
    }// joints
  }
  
  void learn_conf_param_pred_data(const PartApp &part_app)
  {
    cout << "learn_conf_param_pred_data" << endl;
    //const PartConfig &part_conf = part_app.m_part_conf;
    
    int nJoints = part_app.m_part_conf.joint_size();

    /* add random offset to part positions to simulate detection on sparse grid */
    boost::mt19937 rng(42u);
    boost::uniform_real<> dist_01(0.0, 1.0);
    boost::variate_generator<boost::mt19937, boost::uniform_real<> > gen_01(rng, dist_01);
    
    double window_step = part_app.m_abc_param.desc_step() * part_app.m_abc_param.window_desc_step_ratio();
    cout << "window_step: " << window_step << endl;

    QString qsPredDataDir = part_app.m_exp_param.pred_data_dir().c_str();
    
    /* process joints */ 
    for (int jidx = 0; jidx < nJoints; ++jidx) {
      int nJointTypes = 1;
      if (part_app.m_part_conf.joint(jidx).has_num_joint_types())
	nJointTypes = part_app.m_part_conf.joint(jidx).num_joint_types();
      cout << "jidx: " << jidx << ", nJointTypes: " << nJointTypes << endl;
      if (nJointTypes > 1){
	
	std::vector<AnnotationList> annolistClusters(nJointTypes);
	
	QString qsFilename = qsPredDataDir + "/trainlist_pred_jidx_" + QString::number(jidx) + ".mat";
	cout << qsFilename.toStdString().c_str() << endl;
	boost_math::double_matrix clus_all;
	matlab_io::mat_load_double_matrix(qsFilename, "clusidx_train", clus_all);
	assert(clus_all.size1() == part_app.m_validation_annolist.size());
	for(uint imgidx = 0; imgidx < part_app.m_validation_annolist.size(); imgidx++){
	  assert(part_app.m_validation_annolist[imgidx].size() == 1);
	  int idClus = clus_all(imgidx,0);
	  if (idClus > -1)
	    annolistClusters[idClus].addAnnotation(part_app.m_validation_annolist[imgidx]);
	}
	
	for(int tidx = 0; tidx < nJointTypes; tidx++){
	  if (annolistClusters[tidx].size() == 0){
	    cout << "\nWarning! Empty cluster!\n" << endl;
	    continue;
	  }
	  cout << "tidx: " << tidx << endl;
	  learn_conf_param_jidx(part_app, annolistClusters[tidx], window_step, jidx, gen_01, tidx);
	}
      }
      else{
	learn_conf_param_jidx(part_app, part_app.m_train_annolist, window_step, jidx, gen_01);
      }
    }// joints
  }

}// namespace 
