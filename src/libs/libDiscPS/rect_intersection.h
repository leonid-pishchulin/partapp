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

#ifndef _RECT_INTERSECTION_H_
#define _RECT_INTERSECTION_H_

#include <QVector>
#include <QPointF>

inline float norm(const QPointF &p) {return sqrt(square(p.x()) + square(p.y())); }

double intersection_area(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2);

double get_iou(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2);

bool match_rect(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2, 
                double c_thres, double o_thresh, double d_thresh, double a_thresh);

double get_point_distance(const QVector<QPointF> &rect, const QPointF &pt);

double rect_area(const QVector<QPointF> &rect);

void get_intersection_points(const QVector<QPointF> &rect1, const QVector<QPointF> &rect2, QVector<QPointF> &points);

void points_in_rect(const QVector<QPointF> &points, const QVector<QPointF> &rect, QVector<QPointF> &points_inside);

double tri_area(const QPointF &p1, const QPointF &p2, const QPointF &p3);

bool segment_intersect(const QPointF &p1, const QPointF &p2, 
                       const QPointF &p3, const QPointF &p4, QPointF &result);

bool rect_circle_intersect(const QPointF &rect_center, const QPointF &rect_px, const QPointF &rect_py,
                           const QPointF &circle_center, float circle_radius);
#endif
