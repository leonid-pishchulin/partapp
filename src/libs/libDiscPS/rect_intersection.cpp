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

#include <QVector>
#include <QPointF>
#include <QMatrix>

#include <iostream>
#include <limits>
#include <vector>
#include <utility>

#include <cmath>
#include <cassert>
#include <algorithm>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

//#include "math_helpers.h"
#include <libMisc/misc.hpp>

#include "rect_intersection.h"

using namespace std;
using namespace boost::lambda;

void get_max_circle(const QVector<QPointF> &rect, QPointF& c, double &R)
{
  assert(rect.size() > 0);

  c = rect[0];

  for (int idx = 1; idx < rect.size(); ++idx)
    c += rect[idx];

  c /= rect.size();
  R = norm(rect[0] - c);

  for (int idx = 1; idx < rect.size(); ++idx) {
    double cur_R = norm(rect[idx] - c);

    if (cur_R > R)
      R = cur_R;
  }
}

double get_iou(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2)
{
  assert(rect1.size() == 4 && rect2.size() == 4);

  QPointF c1,c2;
  double R1, R2;

  get_max_circle(rect1, c1, R1);
  get_max_circle(rect2, c2, R2);

  double dc12 = norm(c1 - c2);

  if (dc12 > R1 + R2) {
    return 0;
  }
  else {
    double a1 = rect_area(rect1);
    double a2 = rect_area(rect2);

    double ia = intersection_area(rect1, rect2);

    return ia / (a1 + a2 - ia);
  }
}


/** 
    return true if circle and rectangle intersect and false otherwise

    circle is defined by center and radius
    rectangle is defined by center and two orthogonal vectors such that 
    rect_center +/- rect_px +/- rect_py give the corners of rectangle
*/
bool rect_circle_intersect(const QPointF &rect_center, const QPointF &rect_px, const QPointF &rect_py,
                           const QPointF &circle_center, float circle_radius)
{
  float norm_rect_px = norm(rect_px);
  float norm_rect_py = norm(rect_py);

  QPointF npx = rect_px / norm_rect_px;
  QPointF npy = rect_py / norm_rect_py;

  QPointF p = circle_center - rect_center;
  float proj_px = npx.x()*p.x() + npx.y()*p.y();
  float proj_py = npy.x()*p.x() + npy.y()*p.y();

  if (abs(proj_px) < norm_rect_px + circle_radius && abs(proj_py) < norm_rect_py + circle_radius) {
    return true;
  }
  else {
    return false;
  }

}


bool match_rect(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2,
                double c_thres, double o_thresh, double d_thresh, double a_thresh) 
{
  assert(rect1.size() == 4 && rect2.size() == 4);

  /* quick test if rects might intersect */
  QPointF c1 = (rect1[2] + rect1[0])/2;
  QPointF c2 = (rect2[2] + rect2[0])/2;
  double r1 = norm(rect1[0] - c1);
  double r2 = norm(rect2[0] - c2);

  double cd = norm(c1 - c2);

  if (cd > r1 + r2)
    return false;
  
  /* compute matching criteria */
  double A = intersection_area(rect1, rect2);

  double A1 = rect_area(rect1);
  double A2 = rect_area(rect2);

  double cover = A/A1;
  double overlap = A/A2;

  double d = get_point_distance(rect1, c2);

  QPointF dir1 = rect1[1] - rect1[0];
  dir1 /= norm(dir1);
  QPointF dir2 = rect2[1] - rect2[0];
  dir2 /= norm(dir2);

  //double alpha = acos(dir1.x()*dir2.x() + dir1.y()*dir2.y())*180.0/M_PI;
  //bool bMatch = (cover >= c_thres && overlap >= o_thresh && d <= d_thresh && alpha <= a_thresh);

  bool bMatch = (cover >= c_thres && overlap >= o_thresh && d <= d_thresh);

//   if (bMatch) {
//     cout << "cover " << cover << endl;
//     cout << "overlap " << overlap << endl;
//     cout << "distance " << d << endl;
//     cout << "angle " << alpha << endl;
//   }
  
  return bMatch;
}

double get_point_distance(const QVector<QPointF> &rect, const QPointF &pt)
{
  assert(rect.size() == 4);
  QPointF v1 = (rect[1] - rect[0])/2;
  QPointF v2 = (rect[3] - rect[0])/2;

  QPointF c = (rect[2] + rect[0])/2;
  QPointF pt2 = pt - c;

  QMatrix K(v1.x(), v2.x(), v1.y(), v2.y(), 0, 0);
  QMatrix invK = K.inverted();

  QPointF pt3(invK.m11()*pt2.x() + invK.m12()*pt2.y(), 
              invK.m21()*pt2.x() + invK.m22()*pt2.y());

  return sqrt(square(pt3.x()) + square(pt3.y()));
}

double rect_area(const QVector<QPointF> &rect)
{
  assert(rect.size() == 4);

//   QPointF v1 = rect[1] - rect[0];
//   QPointF v2 = rect[3] - rect[0];

//   return norm(v1) * norm(v2);

  return tri_area(rect[0], rect[1], rect[2]) + tri_area(rect[2], rect[3], rect[0]);
}

/**
 * find intersection area of two triangles
 */
double intersection_area(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2)
{
  /** BEGIN, workaround for bug when rect1 is the same as rect2 */
  double same_rect = true;
  assert(rect1.size() == rect2.size() && rect2.size() == 4);

  for (int idx = 0; idx < rect1.size(); ++idx) {
    double dx = (rect1[idx].x() - rect2[idx].x()); 
    double dy = (rect1[idx].y() - rect2[idx].y()); 

    double d = sqrt(dx*dx + dy*dy);
    if (d > 1e-6) {
      same_rect = false;
      break;
    }
  }
  if (same_rect) 
    return rect_area(rect1);
  /** END */

  QVector<QPointF> points;
  get_intersection_points(rect1, rect2, points);

  double a = 0;

  if (!points.empty()) {
    uint li = 0;

    uint nPoints = points.size();
    float eps = numeric_limits<float>::epsilon();
    /* find lowest point, for equal y choose one with smallest x */
    for (uint i = 0; i < nPoints; ++i) {

      if (points[i].y() < points[li].y())
        li = i;
      else if (fabs(points[i].y() - points[li].y()) < eps && points[i].x() < points[li].x())
        li = i;

    }// points

    /* order points by angle with respect to lowest point */
    std::vector<pair<float, int> > postan;
    std::vector<pair<float, int> > negtan;

    for (uint i = 0; i < nPoints; ++i) {
      if (i != li) {
        QPointF d = points[i] - points[li];

        float ptan;
        if (d.x() == 0)
          ptan = numeric_limits<float>::infinity();
        else 
          ptan = d.y() / d.x();

        if (d.x() >= 0)
          postan.push_back(std::pair<float, int>(ptan, i));
        else 
          negtan.push_back(std::pair<float, int>(ptan, i));
      }
    }// points
    
    sort(postan.begin(), postan.end(), bind(&std::pair<float, int>::first, _1) < 
         bind(&std::pair<float, int>::first, _2));

    sort(negtan.begin(), negtan.end(), bind(&std::pair<float, int>::first, _1) < 
         bind(&std::pair<float, int>::first, _2));

    vector<int> point_order;

    for (uint i = 0; i < postan.size(); ++i)
      point_order.push_back(postan[i].second);
    
    for (uint i = 0; i < negtan.size(); ++i) 
      point_order.push_back(negtan[i].second);
    
    int nPoints2 = point_order.size() - 1;

    for (int i = 0; i < nPoints2; ++i) {
      int ptidx2 = point_order[i];
      int ptidx3 = point_order[i+1];

      double ta = tri_area(points[li], points[ptidx2], points[ptidx3]);
      a += ta;
    }// points

  }// if has intersection points

  return a;
}

/**
* compute intersection points of two rectangles
*/
void get_intersection_points(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2, QVector<QPointF> &points)
{
  points.clear();

  /* edge intersections */
  for (uint i1 = 0; i1 < 4; ++i1)
    for (uint i2 = 0; i2 < 4; ++i2) {
      uint idx1 = (i1 + 1)%4;
      uint idx2 = (i2 + 1)%4;

      QPointF p;
      bool intersect = segment_intersect(rect1[i1], rect1[idx1], rect2[i2], rect2[idx2], p);

      if (intersect)
        points.push_back(p);

    }

  points_in_rect(rect1, rect2, points);
  points_in_rect(rect2, rect1, points);
}


/** 
 * return points inside of the rectangle
 */
inline int sign_func(double v)
{
  if (v <= 0)
    return -1;
  else
    return 1;
}
  

void points_in_rect(const QVector<QPointF> &points, const QVector<QPointF> &rect, QVector<QPointF> &points_inside)
{
  assert(rect.size() == 4);

  uint nPoints = points.size();

  for (uint ptidx = 0; ptidx < nPoints; ++ptidx) {
    bool inside = true;

    float x = points[ptidx].x();
    float y = points[ptidx].y();

    int dir_sign = -1;

    for (uint i = 0; i < 4; ++i) {
      uint ni = (i+1)%4;

      float x1 = rect[i].x();
      float y1 = rect[i].y();
      float x2 = rect[ni].x();
      float y2 = rect[ni].y();

      double s = -(x - x1)*(y2 - y1) + (y - y1)*(x2 - x1);

      //if ((x - x1)*(x2 - x1) + (y - y1)*(y2 - y1) < 0) {

      if (i == 0) {
        dir_sign = sign_func(s);
      }
      else {
        int cur_sign = sign_func(s);

        if (cur_sign != dir_sign) {
          inside = false;
          break;
        }

      }

    }// sides

    if (inside)
      points_inside.push_back(points[ptidx]);

  }// points
}

/**
 * area of the triangle defined by 3 points
 */
double tri_area(const QPointF &p1, const QPointF &p2, const QPointF &p3)
{
  QPointF v1 = p2 - p1;
  QPointF v2 = p3 - p1;

  return 0.5*fabs(v1.x()*v2.y() - v1.y()*v2.x());
}


/**
 * test if two segments intersect and return intersection point
 * p1, p2: first segment
 * p3, p4: second segment
 */
bool segment_intersect(const QPointF &p1, const QPointF &p2, 
                        const QPointF &p3, const QPointF &p4, QPointF &result)
{
  float x1 = p1.x();
  float y1 = p1.y();

  float x2 = p2.x();
  float y2 = p2.y();

  float x3 = p3.x();
  float y3 = p3.y();

  float x4 = p4.x();
  float y4 = p4.y();

  float d = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1);

  // test if segments are parallel
  bool res = false;
  if (d != 0) {
    float n1 = (x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3);
    float n2 = (x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3);
    float ua = n1/d;
    float ub = n2/d;

    // test if intersection is whithin both segments (including segment boundaries)
    //if (ua > 0 && ua < 1 && ub > 0 && ub < 1) {
    if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
      result.setX(x1 + ua*(x2 - x1));
      result.setY(y1 + ua*(y2 - y1));
      res = true;
    }
  }

  return res;
}
