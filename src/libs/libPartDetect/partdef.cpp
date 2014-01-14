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
#include <fstream>
#include <climits>

#include <QString>
#include <QPointF>
#include <QImage>

#include <libBoostMath/boost_math.hpp>

#include "partdef.h"

using namespace std;

PartBBox::PartBBox(int ox, int oy, double xaxis_x, double xaxis_y, 
         double _min_x, double _max_x, 
         double _min_y, double _max_y) : part_pos(2), part_x_axis(2), part_y_axis(2), 
                                         max_proj_x(_max_x),
                                         min_proj_x(_min_x), 
                                         max_proj_y(_max_y), 
                                         min_proj_y(_min_y), use_endpoints(false)
{ 
  part_pos(0) = ox;
  part_pos(1) = oy;
  part_x_axis(0) = xaxis_x;
  part_x_axis(1) = xaxis_y;
  part_y_axis(0) = -xaxis_y;
  part_y_axis(1) = xaxis_x;  
}

void update_bbox_min_max_proj(PartBBox &part_bbox, vector<boost_math::double_vector> &corners)
{
  part_bbox.max_proj_x = -numeric_limits<double>::infinity();
  part_bbox.min_proj_x = numeric_limits<double>::infinity();

  part_bbox.max_proj_y = -numeric_limits<double>::infinity();
  part_bbox.min_proj_y = numeric_limits<double>::infinity();
    
  // part bounding box should include all points used to compute part position
  for (int i = 0; i < corners.size(); ++i) {
    
    boost_math::double_vector annopoint(2);
    annopoint(0) = corners[i](0);
    annopoint(1) = corners[i](1);
    
    annopoint = annopoint - part_bbox.part_pos;
    
    double proj_x = inner_prod(part_bbox.part_x_axis, annopoint);
    double proj_y = inner_prod(part_bbox.part_y_axis, annopoint);

    if (proj_x < part_bbox.min_proj_x)
      part_bbox.min_proj_x = proj_x;
    if (proj_x > part_bbox.max_proj_x)
      part_bbox.max_proj_x = proj_x;

    if (proj_y < part_bbox.min_proj_y)
      part_bbox.min_proj_y = proj_y;
    if (proj_y > part_bbox.max_proj_y)
      part_bbox.max_proj_y = proj_y;
    
  }// annopoints
  part_bbox.x1 = corners[0](0);
  part_bbox.y1 = corners[0](1);
  part_bbox.x2 = corners[1](0);
  part_bbox.y2 = corners[1](1);
}

void get_bbox_corners(const AnnoRect &annorect, const PartDef &partdef, vector<boost_math::double_vector> &corners)
{  
  boost_math::double_vector ul = boost_math::zero_double_vector(2);
  boost_math::double_vector lr = boost_math::zero_double_vector(2);
  boost_math::double_vector ll = boost_math::zero_double_vector(2);
  boost_math::double_vector ur = boost_math::zero_double_vector(2);
  double x_min =  numeric_limits<double>::infinity();
  double x_max = -numeric_limits<double>::infinity();
  double y_min =  numeric_limits<double>::infinity();
  double y_max = -numeric_limits<double>::infinity();

    
  for (int i = 0; i < partdef.part_pos_size(); ++i) {
    uint id = partdef.part_pos(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);
    assert(p != NULL);
    
    assert(p->id == (int)id);
    
    if (x_min > p->x)
      x_min = p->x;
    if (x_max < p->x)
      x_max = p->x;
    if (y_min > p->y)
      y_min = p->y;
    if (y_max < p->y)
      y_max = p->y;
  }

  ul(0) = x_min;
  ul(1) = y_min;
  lr(0) = x_max;
  lr(1) = y_max;
  ll(0) = x_min;
  ll(1) = y_max;
  ur(0) = x_max;
  ur(1) = y_min;
  
  corners.push_back(ul);
  corners.push_back(lr);
  corners.push_back(ll);
  corners.push_back(ur);
}

boost_math::double_vector get_part_position_complex(const AnnoRect &annorect, const PartDef &partdef) 
{
  assert(partdef.part_pos_size() > 2);
  vector<boost_math::double_vector> corners;
  get_bbox_corners(annorect, partdef, corners);
  boost_math::double_vector v(2);
  v(0) = 0.5*(corners[0](0) + corners[1](0));
  v(1) = 0.5*(corners[0](1) + corners[1](1));
  
  return v;
}

boost_math::double_vector get_part_position(const AnnoRect &annorect, const PartDef &partdef) 
{
  boost_math::double_vector v = boost_math::zero_double_vector(2);
  
  for (int i = 0; i < partdef.part_pos_size(); ++i) {
    uint id = partdef.part_pos(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);
    assert(p != NULL);
    
    /* make sure we get the right annopoint */
    assert(p->id == (int)id);
    
    v(0) += p->x;
    v(1) += p->y;
  }
  v *= 1.0/partdef.part_pos_size();

  return v;
}

bool get_part_x_axis_complex(const AnnoRect &annorect, const PartDef &partdef, 
			     boost_math::double_vector &part_x_axis)
{
  //cout << "get_part_x_axis_complex()" << endl;
  assert(partdef.part_x_axis_from_size() > 1 && partdef.part_x_axis_to_size() > 1);
    
  float from_x = 0, from_y = 0, to_x = 0, to_y = 0;
  int n_p = 0;
  for (int i = 0; i < partdef.part_x_axis_from_size(); i++){
    uint id = partdef.part_x_axis_from(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);
    
    assert(p != NULL);
    assert(p->id == (int)id);
    
    from_x += p->x;
    from_y += p->y;
    n_p ++;
  }
  from_x /= n_p;
  from_y /= n_p;
  
  n_p = 0;
  for (int i = 0; i < partdef.part_x_axis_to_size(); i++){
    uint id = partdef.part_x_axis_to(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);
    
    assert(p != NULL);
    assert(p->id == (int)id);
    
    to_x += p->x;
    to_y += p->y;
    n_p ++;
  }
  to_x /= n_p;
  to_y /= n_p;
  
  part_x_axis(0) = to_x - from_x;
  part_x_axis(1) = to_y - from_y;
  
  if (not(abs(part_x_axis(0)) > 1e-6 || abs(part_x_axis(1)) > 1e-6))
    return false;
  
  boost_math::double_matrix R = boost_math::get_rotation_matrix(partdef.part_x_axis_offset()*M_PI/180.0);
  part_x_axis = prod(R, part_x_axis);
  return true;
}

bool get_part_x_axis(const AnnoRect &annorect, const PartDef &partdef, 
                     boost_math::double_vector &part_x_axis) 
{
  //cout << "get_part_x_axis()" << endl;
  assert(part_x_axis.size() == 2);
  
  assert(partdef.part_x_axis_from_size() <= 1 && partdef.part_x_axis_to_size() <= 1);
  
  if (partdef.part_x_axis_from_size() == 0 && partdef.part_x_axis_to_size() == 0) {
    part_x_axis(0) = 1;
    part_x_axis(1) = 0;
  }
  else{
    
    assert(part_x_axis.size() == 2);
    uint fromid = partdef.part_x_axis_from(0);
    uint toid = partdef.part_x_axis_to(0);
    
    const AnnoPoint *from = annorect.get_annopoint_by_id(fromid);
    const AnnoPoint *to = annorect.get_annopoint_by_id(toid);

    /* make sure we get the right annopoint */
    if (from == NULL || to == NULL)
      return false;
    
    part_x_axis(0) = to->x - from->x;
    part_x_axis(1) = to->y - from->y;
     
    if (not(abs(part_x_axis(0)) > 1e-6 || abs(part_x_axis(1)) > 1e-6))
      return false;

    boost_math::double_matrix R = boost_math::get_rotation_matrix(partdef.part_x_axis_offset()*M_PI/180.0);
    part_x_axis = prod(R, part_x_axis);
  }
  return true;
}

bool get_part_bbox_atomic(const AnnoRect &annorect, const PartDef &partdef, 
			  PartBBox &part_bbox, double scale)
{
  assert(annorect_has_part(annorect, partdef));

  part_bbox.part_pos = get_part_position(annorect, partdef);
  
  bool bAxisValid = get_part_x_axis(annorect, partdef, part_bbox.part_x_axis);
  if (!bAxisValid)
    return false;
  
  part_bbox.part_x_axis /= norm_2(part_bbox.part_x_axis);
  
  part_bbox.part_y_axis = prod(boost_math::get_rotation_matrix(M_PI/2), part_bbox.part_x_axis);
  
  part_bbox.max_proj_x = -numeric_limits<double>::infinity();
  part_bbox.min_proj_x = numeric_limits<double>::infinity();

  part_bbox.max_proj_y = -numeric_limits<double>::infinity();
  part_bbox.min_proj_y = numeric_limits<double>::infinity();

  // part bounding box should include all points used to compute part position
  for (int i = 0; i < partdef.part_pos_size(); ++i) {
    
    uint id = partdef.part_pos(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);

    assert(p != NULL);
    assert(p->id == (int)id);

    boost_math::double_vector annopoint(2);

    annopoint(0) = p->x;
    annopoint(1) = p->y;

    annopoint = annopoint - part_bbox.part_pos;
    
    double proj_x = inner_prod(part_bbox.part_x_axis, annopoint);
    double proj_y = inner_prod(part_bbox.part_y_axis, annopoint);

    if (proj_x < part_bbox.min_proj_x)
      part_bbox.min_proj_x = proj_x;
    if (proj_x > part_bbox.max_proj_x)
      part_bbox.max_proj_x = proj_x;

    if (proj_y < part_bbox.min_proj_y)
      part_bbox.min_proj_y = proj_y;
    if (proj_y > part_bbox.max_proj_y)
      part_bbox.max_proj_y = proj_y;
    
  }// annopoints

  part_bbox.min_proj_x -= scale*partdef.ext_x_neg();
  part_bbox.max_proj_x += scale*partdef.ext_x_pos();

  part_bbox.min_proj_y -= scale*partdef.ext_y_neg();
  part_bbox.max_proj_y += scale*partdef.ext_y_pos();
  return true;
}

bool get_part_bbox_complex(const AnnoRect &annorect, const PartDef &partdef, 
                   PartBBox &part_bbox, double scale)
{
  assert(annorect_has_part(annorect, partdef));

  part_bbox.part_pos = get_part_position_complex(annorect, partdef);
  bool bAxisValid = true;
  
  if (partdef.part_x_axis_from_size() <= 1 && partdef.part_x_axis_to_size() <= 1)
    bAxisValid = get_part_x_axis(annorect, partdef, part_bbox.part_x_axis);
  else
    bAxisValid = get_part_x_axis_complex(annorect, partdef, part_bbox.part_x_axis);

  if (!bAxisValid)
    return false;
  
  part_bbox.part_x_axis /= norm_2(part_bbox.part_x_axis);
  
  part_bbox.part_y_axis = prod(boost_math::get_rotation_matrix(M_PI/2), part_bbox.part_x_axis);
    
  vector<boost_math::double_vector> corners;
  get_bbox_corners(annorect, partdef, corners);
  update_bbox_min_max_proj(part_bbox, corners);
    
  part_bbox.min_proj_x -= scale*partdef.ext_x_neg();
  part_bbox.max_proj_x += scale*partdef.ext_x_pos();

  part_bbox.min_proj_y -= scale*partdef.ext_y_neg();
  part_bbox.max_proj_y += scale*partdef.ext_y_pos();
  return true;
}

void get_part_polygon(const PartBBox &part_bbox, QPolygonF &polygon)
{
  polygon.clear();

  boost_math::double_vector t(2);
  t = part_bbox.part_x_axis*part_bbox.min_proj_x + part_bbox.part_y_axis*part_bbox.min_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));
    
  t = part_bbox.part_x_axis*part_bbox.max_proj_x + part_bbox.part_y_axis*part_bbox.min_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));

  t = part_bbox.part_x_axis*part_bbox.max_proj_x + part_bbox.part_y_axis*part_bbox.max_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));

  t = part_bbox.part_x_axis*part_bbox.min_proj_x + part_bbox.part_y_axis*part_bbox.max_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));
}

bool get_part_bbox(const AnnoRect &annorect, const PartDef &partdef, 
                   PartBBox &part_bbox, double scale){
  
  if (partdef.part_pos_size() < 3)
    return get_part_bbox_atomic(annorect, partdef, part_bbox, scale);
  else
    return get_part_bbox_complex(annorect, partdef, part_bbox, scale);
  
}

/** 
    coloridx: 0 - yellow, 1 - red, 2 - green, 3 - blue, if < 0 - only position without bounding box is drawn
 */
void draw_bbox(QPainter &painter, const PartBBox &part_bbox, int coloridx, int pen_width)
{    

  if (coloridx >= 0) {
    painter.setPen(Qt::yellow);
  
    int marker_radius = 3;
    int part_axis_length = 10;

    painter.drawEllipse(QRect((int)(part_bbox.part_pos(0) - marker_radius), (int)(part_bbox.part_pos(1) - marker_radius), 
			      2*marker_radius, 2*marker_radius));

    boost_math::double_vector v(2);
    v = part_bbox.part_pos + part_axis_length * part_bbox.part_x_axis;
    painter.drawLine((int)part_bbox.part_pos(0), (int)part_bbox.part_pos(1), (int)v(0), (int)v(1));

    painter.setPen(Qt::red);
    v = part_bbox.part_pos + part_axis_length * part_bbox.part_y_axis;
    painter.drawLine((int)part_bbox.part_pos(0), (int)part_bbox.part_pos(1), (int)v(0), (int)v(1));
    painter.setPen(Qt::yellow);

    QPen pen;

    if (coloridx == 0) 
      pen.setColor(Qt::yellow);
    else if (coloridx == 1)
      pen.setColor(Qt::red);
    else if (coloridx == 2)
      pen.setColor(Qt::green);
    else if (coloridx == 3)
      pen.setColor(Qt::blue);
    else
      pen.setColor(Qt::black);

    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(pen_width);

    painter.setPen(pen);

    QPolygonF polygon;
    get_part_polygon(part_bbox, polygon);
    painter.drawPolygon(polygon);
  }
  else {

    painter.setPen(Qt::yellow);

    if (coloridx == -1) 
      painter.setPen(Qt::yellow);
    else if (coloridx == -2)
      painter.setPen(Qt::red);
    else if (coloridx == -3)
      painter.setPen(Qt::green);
    else
      painter.setPen(Qt::blue);

    int x = part_bbox.part_pos(0);
    int y = part_bbox.part_pos(1);
    
    painter.drawLine(x-1, y, x+1, y);
    painter.drawLine(x, y-1, x, y+1);
  }

}

QImage visualize_parts(const PartConfig &conf, const PartWindowParam &window_param, const Annotation &annotation)
{
  assert(annotation.size() > 0);

  double scale = (annotation[0].bottom() - annotation[0].top())/window_param.train_object_height();
  cout << "visualize_parts, scale: " << scale << endl;

  QImage _img;
  cout << "loading image" << endl;
  assert(_img.load(annotation.imageName().c_str()));
  QImage img = _img.convertToFormat(QImage::Format_RGB32);

  QPainter painter(&img);  
  for (int pidx = 0; pidx < conf.part_size(); ++pidx) {
    PartBBox part_bbox;
    get_part_bbox(annotation[0], conf.part(pidx), part_bbox, scale);

    int coloridx = 1;
    int pen_width = 2;

    if (conf.part(pidx).is_detect()) {
      if (conf.part(pidx).is_root()) 
	draw_bbox(painter, part_bbox, 2, pen_width);
      else
	draw_bbox(painter, part_bbox, coloridx, pen_width);
    }
    else
      draw_bbox(painter, part_bbox, -1); // only draw center point and axis (skip bounding box)

  }

  cout << "done" << endl;
  return img;
}

bool annorect_has_part(const AnnoRect &annorect, const PartDef &partdef)
{
  bool bres = true;

  for (int idx = 0; idx < partdef.part_pos_size(); ++idx) {
    uint apidx = partdef.part_pos(idx);

    if (annorect.get_annopoint_by_id(apidx) == NULL) {
      bres = false;
      break;
    }
  }

  return bres;
}

