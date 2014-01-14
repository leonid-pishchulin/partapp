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

#ifndef _PART_DEF_H_
#define _PART_DEF_H_

#include <QString>
#include <QPainter>

#include <libAnnotation/annotationlist.h>
#include <libAnnotation/annorect.h>

#include <libBoostMath/boost_math.h>

#include <libPartDetect/PartConfig.pb.h>
#include <libPartDetect/PartWindowParam.pb.h>

class QPolygonF;

/**
   structure that keeps data about part postition, orientation and bounding box
 */
struct PartBBox {

  PartBBox():part_pos(2), part_x_axis(2), part_y_axis(2), use_endpoints(false) {}

  PartBBox(int ox, int oy, double xaxis_x, double xaxis_y, 
	   double _min_x, double _max_x, 
	   double _min_y, double _max_y);

  boost_math::double_vector part_pos;
  boost_math::double_vector part_x_axis;
  boost_math::double_vector part_y_axis;

  double max_proj_x;
  double min_proj_x;

  double max_proj_y;
  double min_proj_y;

  float x1, x2;
  float y1, y2;
  bool use_endpoints;

};

/**
   compute position of part in the image (average over positions of annopoints)
 */
boost_math::double_vector get_part_position(const AnnoRect &annorect, const PartDef &partdef);

/**
   compute x axis of part coordinate system (in image coordinates)
 */
bool get_part_x_axis(const AnnoRect &annorect, const PartDef &partdef, 
                     boost_math::double_vector &part_x_axis);

/**
   compute part bounding box (smallest rectangle that has the same orientation as part and contains all points
   which define part position)
 */
bool get_part_bbox(const AnnoRect &annorect, const PartDef &partdef, 
                   PartBBox &part_bbox, double scale = 1.0);

void draw_bbox(QPainter &painter, const PartBBox &part_bbox, int coloridx = 0, int pen_width = 1);

/**
   visualize part positions and corresponding part bounding boxes, 
   visualization is saved in ./debug directory
 */
QImage visualize_parts(const PartConfig &conf, const PartWindowParam &window_param, const Annotation &annotation);

void get_part_polygon(PartBBox &part_bbox, QPolygonF &polygon); 

bool annorect_has_part(const AnnoRect &annorect, const PartDef &partdef);

void update_bbox_min_max_proj(PartBBox &part_bbox, std::vector<boost_math::double_vector> &corners);

void get_bbox_corners(const AnnoRect &annorect, const PartDef &partdef, std::vector<boost_math::double_vector> &corners);

#endif
