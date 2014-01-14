/*
* "Shape Context" descriptor code (by Krystian Mikolajczyk <http://personal.ee.surrey.ac.uk/Personal/K.Mikolajczyk> ) based on SIFT (by David Lowe)
*
* This code is granted free of charge for non-commercial research and
* education purposes. However you must obtain a license from the author
* to use it for commercial purposes.

* The software and derivatives of the software must not be distributed
* without prior permission of the author.
*
*  If you use this code you should cite the following paper:
*  K. Mikolajczyk, C. Schmid,  A performance evaluation of local descriptors. In PAMI 27(10):1615-1630
*/

#ifndef _feature_h_
#define _feature_h_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "../ImageContent/imageContent.h"

#define PATCH_SIZE 21
#define SCALE_FACTOR  1

#define FALSE 0
#define TRUE 1

extern DARY *patch_mask;
extern float PATCH_SUM;
void initPatchMask(int size);

inline float square(float a){return a*a;}

void normalize(DARY * img,int x, int y, float radius);

void cannyEdges(DARY *img, DARY *edge,  float scale, float lower_threshold, float higher_threshold);
void cannyEdges(DARY *dx, DARY *dy, DARY *edge,  float lower_threshold, float higher_threshold);
void cannyEdges(DARY *img, DARY *edge,  float lower_threshold, float higher_threshold);
void cannyEdges(DARY *dx, DARY *dy, DARY *grad, DARY *edge,  float lower_threshold, float higher_threshold);

#endif
