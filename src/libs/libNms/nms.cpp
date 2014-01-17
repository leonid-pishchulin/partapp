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

#include <libMisc/misc.hpp>

#include <cmath>

#include "nms.h"
#include "rect_intersection.h"

using namespace std;

void rect_from_hyp(const HypothesisList::ObjectHypothesis &h, QVector<QPointF> &rect, 
                   double reference_height, double reference_width)
{
  int nRectHeight = (int)(h.scale() * reference_height);
  //int nRectWidth = (int)(h.scale() * reference_height / height_width_ratio);
  int nRectWidth = (int)(h.scale() * reference_width);

  double c, s;

  /** 
      MA: 2011-08-13, support only upright rectangles for now
   */
  c = 1;
  s = 0;
//   if (h.has_rad_rot()) {
//     c = cos(h.rad_rot());
//     s = sin(h.rad_rot());
//   }
//   else {
//     c = 1;
//     s = 0;
//   }

  QPointF v1(c*nRectWidth/2, s*nRectWidth/2);
  QPointF v2(-s*nRectHeight/2, c*nRectHeight/2);

  QPointF p(h.x(), h.y());
  rect.push_back(p - v1 - v2);
  rect.push_back(p + v1 - v2);
  rect.push_back(p + v1 + v2);
  rect.push_back(p - v1 + v2);
}

void nms_recursive(const ExpParam &exp_param, 
                   const HypothesisList hypothesis_list, 
                   vector<bool> &nms, 
                   double train_object_width, 
                   double train_object_height) 
{
  double nms_ellipse_size = 0.5;

  nms_recursive(hypothesis_list, nms, train_object_width, train_object_height, 
		nms_ellipse_size);


//   nms_recursive(hypothesis_list, nms, train_object_width, train_object_height, 
//                 exp_param.nms_ellipse_size());
}

void nms_recursive(const HypothesisList hypothesis_list, 
                   vector<bool> &nms, 
                   double train_object_width, 
                   double train_object_height, 
		   double ellipse_size)
{
  // convert hypothesis to rectangles 
  vector<QVector<QPointF> > rects;
  vector<double> rect_scores;

  for (int idx = 0; idx < hypothesis_list.hyp_size(); ++idx) {
    QVector<QPointF> cur_rect; 
    rect_from_hyp(hypothesis_list.hyp(idx), cur_rect, train_object_height, train_object_width);
    rects.push_back(cur_rect);
    rect_scores.push_back(hypothesis_list.hyp(idx).score());
  }

  nms_recursive(rects, rect_scores, 
                nms, ellipse_size);
}

// todo: nms_recursive for Annotation ...
void nms_recursive(const Annotation &a,
                   vector<bool> &nms, 
		   double ellipse_size)
{
  // convert hypothesis to rectangles 
  vector<QVector<QPointF> > rects;
  vector<double> rect_scores;

  for (uint idx = 0; idx < a.size(); ++idx) {
    QVector<QPointF> cur_rect; 


  /** 
      MA: 2011-08-13, support only upright rectangles for now
   */

      cur_rect.push_back(QPointF(a[idx].x1(), a[idx].y1()));
      cur_rect.push_back(QPointF(a[idx].x2(), a[idx].y1()));
      cur_rect.push_back(QPointF(a[idx].x2(), a[idx].y2()));
      cur_rect.push_back(QPointF(a[idx].x1(), a[idx].y2()));


//     if (a[idx].x3() != -1 || a[idx].x4() != -1 || a[idx].y3() != -1 || a[idx].y4() != -1) {
//       cur_rect.push_back(QPointF(a[idx].x1(), a[idx].y1()));
//       cur_rect.push_back(QPointF(a[idx].x2(), a[idx].y2()));
//       cur_rect.push_back(QPointF(a[idx].x3(), a[idx].y3()));
//       cur_rect.push_back(QPointF(a[idx].x4(), a[idx].y4()));
//     } else {
//       cur_rect.push_back(QPointF(a[idx].x1(), a[idx].y1()));
//       cur_rect.push_back(QPointF(a[idx].x2(), a[idx].y1()));
//       cur_rect.push_back(QPointF(a[idx].x2(), a[idx].y2()));
//       cur_rect.push_back(QPointF(a[idx].x1(), a[idx].y2()));
//     }

    rects.push_back(cur_rect);
    rect_scores.push_back(a[idx].score());
  }

  nms_recursive(rects, rect_scores, 
                nms, ellipse_size);
}

QPointF get_rect_center(const QVector<QPointF> &rect)
{
  assert(rect.size() == 4);

  return QPointF(0.25*(rect[0].x() + rect[1].x() + rect[2].x() + rect[3].x()), 
                 0.25*(rect[0].y() + rect[1].y() + rect[2].y() + rect[3].y()));
}

void nms_recursive(const vector<QVector<QPointF> > &rects, const vector<double> &rect_scores, 
                   vector<bool> &nms, double ellipse_size)
{

  double dist_threshold = 1.0;

  /** used for comparison with tracklets detector, used in supplementary material to CVPR'09 paper */
  //double ellipse_size = 0.3;
    
  /** used in all other experiments */
  //double ellipse_size = 0.5;

  int nHypothesis = rects.size();
  assert(rect_scores.size() == (uint)nHypothesis);

  cout << "nms_recursive" << endl;
  cout << "number of hypothesis: " << nHypothesis << endl;

  nms.clear();
  nms.resize(nHypothesis, false);

  if (nHypothesis == 0)
    return;

  //assert(nHypothesis > 0);

  for (int i = 0; i < nHypothesis; ++i) {
    if (i != nHypothesis - 1) {
      if (rect_scores[i] < rect_scores[i+1]) {
        cout << i << " " << rect_scores[i] << " " << rect_scores[i+1] << endl;
        assert(false && "unsorted hypothesis vector");
      }
    }

    // double e1 = square(scale*train_object_width);
    // double e2 = square(scale*train_object_height);

    // double c = cos(hypothesis_list.hyp(i).rad_rot());
    // double s = sin(hypothesis_list.hyp(i).rad_rot());
    
    // QPointF v1(c, s);
    // QPointF v2(-s, c);

    QPointF v1 = rects[i][1] - rects[i][0];
    QPointF v2 = rects[i][1] - rects[i][2];
    double e1 = square(v1.x()) + square(v1.y());
    double e2 = square(v2.x()) + square(v2.y());
    v1 /= sqrt(e1);
    v2 /= sqrt(e2);

    QPointF c1 = get_rect_center(rects[i]);
    
    for (int j = i+1; j < nHypothesis; ++j) {

      /** matching based on Mahalanobis distance */
      QPointF c2 = get_rect_center(rects[j]);
      double dx, dy;

      QPointF p = c2 - c1;
      dx = v1.x()*p.x() + v1.y()*p.y();
      dy = v2.x()*p.x() + v2.y()*p.y();
          
      if (dx*dx/e1 + dy*dy/e2 < square(ellipse_size)*dist_threshold) {
        nms[j] = true;
      }
    }// rects
  }// rects 

  cout << "done." << endl;
}

















void nms_recursive_old(const HypothesisList hypothesis_list, 
                   vector<bool> &nms, 
                   double train_object_width, 
                   double train_object_height, 
		   double ellipse_size)
{
  double dist_threshold = 1.0;

  /** used for comparison with tracklets detector, used in supplementary material to CVPR'09 paper */
  //double ellipse_size = 0.3;
    
  /** used in all other experiments */
  //double ellipse_size = 0.5;

  int nHypothesis = hypothesis_list.hyp_size();

  cout << "nms_recursive" << endl;
  cout << "number of hypothesis: " << nHypothesis << endl;

  nms.clear();
  nms.resize(nHypothesis, false);

  // convert hypothesis to rectangles (needed to evaluate IOU criteria)  
//   vector<QVector<QPointF> > rect_points;

//   if (iou_rot_nms) {
//     for (int idx = 0; idx < nHypothesis; ++idx) {
//       QVector<QPointF> cur_rect; 
//       rect_from_hyp(hypothesis_list.hyp(idx), cur_rect, train_object_height, train_object_width);
//       rect_points.push_back(cur_rect);
//     }
//   }

  assert(nHypothesis > 0);
  //bool bRotRects = hypothesis_list.hyp(0).has_rad_rot();
  bool bRotRects = false;

  //assert(bRotRects);
  // 

  for (int i = 0; i < nHypothesis; ++i) {
    if (i != nHypothesis - 1) {
      if (hypothesis_list.hyp(i).score() < hypothesis_list.hyp(i+1).score()) {
        cout << i << " " << hypothesis_list.hyp(i).score() << " " << hypothesis_list.hyp(i+1).score() << endl;
        assert(false && "unsorted hypothesis vector");
      }
    }

    double scale = hypothesis_list.hyp(i).scale();

    for (int j = i+1; j < nHypothesis; ++j) {

      /** matching based on mahalanobis distance */
      //if (!iou_rot_nms) {
        double e1 = square(ellipse_size*scale*train_object_width);
        double e2 = square(ellipse_size*scale*train_object_height);

        double dx, dy;
        if (bRotRects) {
	  assert(false);

//           double c = cos(hypothesis_list.hyp(i).rad_rot());
//           double s = sin(hypothesis_list.hyp(i).rad_rot());

//           QPointF v1(c, s);
//           QPointF v2(-s, c);
//           QPointF p(hypothesis_list.hyp(i).x() - hypothesis_list.hyp(j).x(), hypothesis_list.hyp(i).y() - hypothesis_list.hyp(j).y());
//           dx = v1.x()*p.x() + v1.y()*p.y();
//           dy = v2.x()*p.x() + v2.y()*p.y();
        }
        else {
          dx = hypothesis_list.hyp(i).x() - hypothesis_list.hyp(j).x();
          dy = hypothesis_list.hyp(i).y() - hypothesis_list.hyp(j).y();
        }
          
        if (dx*dx/e1 + dy*dy/e2 < dist_threshold) {
          nms[j] = true;
        }

        //      }
      /** matching based on intersection-over-union */
//       else {
//         assert((uint)i < rect_points.size() && (uint)j < rect_points.size());

//         if (get_iou(rect_points[i], rect_points[j]) > iou_rot_nms_threshold) {
//           nms[j] = true;
//         }
//       }


    }
  }
  cout << "done." << endl;
}

