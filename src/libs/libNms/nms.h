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

#include <vector>

#include <QVector>
#include <QPointF>

#include <libAnnotation/annotation.h>

#include <libPartApp/ExpParam.pb.h>
#include <libPictStruct/HypothesisList.pb.h>

void rect_from_hyp(const HypothesisList::ObjectHypothesis &h, QVector<QPointF> &rect, 
                   double reference_height, double reference_width);

void nms_recursive(const ExpParam &exp_param, 
                   const HypothesisList hypothesis_list, 
                   std::vector<bool> &nms, 
                   double train_object_width, 
                   double train_object_height);

void nms_recursive(const HypothesisList hypothesis_list, 
                   std::vector<bool> &nms, 
                   double train_object_width, 
                   double train_object_height, 
		   double ellipse_size);

void nms_recursive(const std::vector<QVector<QPointF> > &rects, const std::vector<double> &rect_scores, 
                   std::vector<bool> &nms, double ellipse_size);

void nms_recursive(const Annotation &a,
                   std::vector<bool> &nms, 
		   double ellipse_size);

