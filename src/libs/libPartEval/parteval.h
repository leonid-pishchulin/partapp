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

#include <libPartApp/partapp.h>

#include <libPartApp/ExpParam.pb.h>

#include <libPartDetect/PartWindowParam.pb.h>
#include <libPartDetect/partdef.h>

#include <libBoostMath/boost_math.h>
#include <libPictStruct/objectdetect.h>

#include <libMatlabIO/matlab_io.h>

const int EVAL_TYPE_UNARIES = 0;
const int EVAL_TYPE_PS = 1;
const int EVAL_TYPE_DISC_PS = 2;
const int EVAL_TYPE_ROI = 3;
const int EVAL_TYPE_UNARIES_EXTENDED = 4;

void get_bbox_endpoints(PartBBox &bbox, boost_math::double_vector &endpoint_top, 
			boost_math::double_vector &endpoint_bottom, double &seg_len);

void vis_segments(const PartApp &part_app, const AnnotationList &annolist, 
		  int firstidx, int lastidx, QString qsHypDirName, int eval_type = EVAL_TYPE_PS);
void vis_segments_roi(const PartApp &part_app, int firstidx, int lastidx, QString qsHypDirName);

void eval_segments(const PartApp &part_app, 
		   const AnnotationList &annolist, 
		   int firstidx, int lastidx, double &ratio, 
		   QString qsHypDirName, const int eval_type = EVAL_TYPE_PS, 
		   int eval_didx = -1);

void eval_segments_rpc(const PartApp &part_app, const AnnotationList &annolist, int firstidx, int lastidx, int pidx, int tidx, float thresh = 0.5);

void loadPartHyp(const PartApp &part_app, QString qsHypDir, int imgidx,
		 std::vector<object_detect::PartHyp> &part_hyp);

void eval_segments_roi(const PartApp &part_app, int firstidx, int lastidx, QString qsHypDirName);

bool is_gt_match(PartBBox &gt_bbox, PartBBox &detect_bbox, bool match_x_axis = false, float factor = 0.5);
bool is_gt_bbox_match(PartBBox &gt_bbox, PartBBox &detect_bbox, float thresh = 0.5);

void draw_gt_circles(QPainter &painter, PartBBox &bbox);

void vis_backproject(const PartApp &part_app, int firstidx, int lastidx);
