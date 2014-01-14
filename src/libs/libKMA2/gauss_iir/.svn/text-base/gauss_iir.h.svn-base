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

 
#ifndef _gauss_
#define _gauss_

#ifndef M_2PI
//#define M_PI  3.1415926537
#define M_2PI 6.2831853072
#endif

#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "../ImageContent/imageContent.h"

/************************************************************************
   structure used to store the coefficients nii+, nii-, dii+, dii- 
   and the normalisation factor 'scale'
   see [1] p. 9 - 11; p. 13, 14 
*************************************************************************/

#define  GAUSS_CUTOFF 3

void  dXY9(DARY* image_in, DARY* smooth_image);
void  dXX9(DARY* image_in, DARY* smooth_image);
void  dYY9(DARY* image_in, DARY* smooth_image);
void  smooth9(DARY* image_in, DARY* smooth_image);
void  dXY7(DARY* image_in, DARY* smooth_image);
void  dXX7(DARY* image_in, DARY* smooth_image);
void  dYY7(DARY* image_in, DARY* smooth_image);
void  dX6(DARY* image_in, DARY* smooth_image);
void  dY6(DARY* image_in, DARY* smooth_image);
void  dX4(DARY* image_in, DARY* smooth_image);
void  dY4(DARY* image_in, DARY* smooth_image);
void dX2(DARY* image_in, DARY* dximage);
void dY2(DARY* image_in, DARY* dyimage);
void HorConv3(DARY *image,  DARY *result);
void VerConv3(DARY *image,  DARY *result);
void grad2(DARY* image_in, DARY* dyimage);
void dXX3(DARY* image_in, DARY* dximage);
void dYY3(DARY* image_in, DARY* dyimage);
void dXX5(DARY* image_in, DARY* dximage);
void dYY5(DARY* image_in, DARY* dyimage);
void dXX_YY3(DARY* image_in, DARY* dyimage);
void  smooth3(DARY* image_in, DARY* smooth_image);
void  smooth5(DARY* image_in, DARY* smooth_image);
void  smoothSqrt(DARY* image_in, DARY* smooth_image);

void gradAngle(DARY *im, DARY *grad, DARY *ori);
void gradAngle(DARY *dx, DARY *dy, DARY *grad, DARY *ori);
void gradAngle(DARY *img, DARY *dx, DARY *dy, DARY *grad, DARY *ori);

 float smooth(int x, int y, DARY* image_in, float scale);
 float dX(int x, int y, DARY* image_in, float scale);
 float dY(int x, int y, DARY* image_in, float scale);
 float dXX(int x, int y, DARY* image_in, float scale);
 float dYY(int x, int y, DARY* image_in, float scale);
 float dXY(int x, int y, DARY* image_in, float scale);

 void smooth (DARY* image_in, DARY* out_image, float scale);
 void dX (DARY* image_in, DARY* out_image, float scale);
 void dY (DARY* image_in, DARY* out_image, float scale);
 void dXX (DARY* image_in, DARY* out_image, float scale);
 void dXY (DARY* image_in, DARY* out_image, float scale);
 void dYY (DARY* image_in, DARY* out_image, float scale);
 void dXX_YY (DARY* image_in, DARY* out_image, float scale);
 void dX (DARY* image_in,  DARY* image_out, float scalex, float scaley);	
 void dY (DARY* image_in,  DARY* image_out, float scalex, float scaley);	
 float smooth(int x, int y, DARY* image_in, float scalex, float scaley);
 void  smooth(DARY* image_in, DARY* smooth_image, float scalex, float scaley);


 float smoothf(int x, int y, DARY* image_in, float scale);
 float dXf(int x, int y, DARY* image_in, float scale);
 float dYf(int x, int y, DARY* image_in, float scale);
 float dXXf(int x, int y, DARY* image_in, float scale);
 float dXYf(int x, int y, DARY* image_in, float scale);
 float dYYf(int x, int y, DARY* image_in, float scale);
 float dXX_YYf(int x, int y, DARY* image_in, float scale);
 float dXXXf(int x, int y, DARY* image_in, float scale);
 float dXXYf(int x, int y, DARY* image_in, float scale);
 float dXYYf(int x, int y, DARY* image_in, float scale);
 float dYYYf(int x, int y, DARY* image_in, float scale);
 float dXXXXf(int x, int y, DARY* image_in, float scale);
 float dXXXYf(int x, int y, DARY* image_in, float scale);
 float dXXYYf(int x, int y, DARY* image_in, float scale);
 float dXYYYf(int x, int y, DARY* image_in, float scale);
 float dYYYYf(int x, int y, DARY* image_in, float scale);
void drawGauss(DARY* image_in,int x, int y, float scale);



#endif




