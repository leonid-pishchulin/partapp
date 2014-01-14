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

#ifndef XMLHELPERS_H
#define XMLHELPERS_H

#include <vector>
#include <iostream>

std::string getElementDataString(const std::string& name, const std::string& doc);
int getElementDataInt(const std::string& name, const std::string& doc);
float getElementDataFloat(const std::string& name, const std::string& doc);
std::vector<std::string> getElements(const std::string& name, const std::string& doc);


#endif

