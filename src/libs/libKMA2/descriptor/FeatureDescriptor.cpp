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

#include "feature.h"
#include "../gauss_iir/gauss_iir.h"
#define ORI_THRESHOLD 0.80
//#define DESC_SCALE 3
//

using namespace std;
using namespace kma;

// FeatureDescriptor::~FeatureDescriptor(){
//   if( (vec!=NULL) && (!preallocated) )delete[] vec;
//   if(imagename!=NULL)delete[] imagename; 
// }

// FeatureDescriptor::FeatureDescriptor(float xin, float yin, float scale_in, float featureness_in){
//   init();
//   x=xin;y=yin;c_scale=scale_in;featureness=featureness_in;
// }

// void FeatureDescriptor::init(){
// 	x=0; 
// 	y=0;
// 	type=0;
// 	obj=0;
// 	lap=0;
// 	featureness=1;
// 	angle=1000;
// 	c_scale=1;
// 	der_sig=0;
// 	int_sig=0;
// 	extr=0;
// 	l1=1;
// 	l2=0;
// 	eangle=0;
// 	mi11=1;
// 	mi12=0;
// 	mi21=0;     
// 	mi22=1;
// 	nbf=1;
// 	radius=0;
// 	weight=0;
// 	tree_lev=0;
// 	size=0;
// 	var=0;
// 	obj=0;
// 	area=0;	
// 	vec=NULL;
// 	preallocated=false;
// 	imagename=NULL;
// }

// /**********************************************/
// void FeatureDescriptor::allocVec(int size_in){
// 	if( (!preallocated) && (size_in >0) ){
// 		delete [] vec;
// 		size=size_in;
// 		vec = new float[size];
// 		for(int i=0;i<size;i++)vec[i]=0;
//     }
// }


//void normalize(DARY * img, int x, int y, float radius){ // Gyuri: Why is this function have 4 parameters? Only one is used!
void normalize(DARY * img, int, int, float){
  float sum=0;
  float gsum=0; 

  for(uint j=0;j<img->y();j++){ 
    for(uint i=0;i<img->x();i++){ 
      if(patch_mask->fel[j][i]>0){
        sum+=img->fel[j][i]; 
        gsum++;
      }
    } 
  }    
  sum=sum/gsum;
  float var=0;
  for(uint j=0;j<img->y();j++){ 
    for(uint i=0;i<img->x();i++){ 
      if(patch_mask->fel[j][i]>0){	
        var+=(sum-img->fel[j][i])*(sum-img->fel[j][i]);	
      }
    }
  }     
  var=sqrt(var/gsum);    

  //  cout << "mean "<<sum<< " " <<img->fel[y][x] << " var " << var << endl;
  float fac=50.0/var;
  float max=0,min=1000;
  for(uint j=0;j<img->y();j++){ 
    for(uint i=0;i<img->x();i++){ 
      img->fel[j][i]=128+fac*(img->fel[j][i]-sum);
      if(max<img->fel[j][i])max=img->fel[j][i];
      if(min>img->fel[j][i])min=img->fel[j][i];
      if(img->fel[j][i]>255)img->fel[j][i]=255;
      if(img->fel[j][i]<0)img->fel[j][i]=0;
    }
  }   
  //    // cout << "max " << max << " min "<< min <<endl;
}



/************NORMALIZATION PATCH****************/

DARY *patch_mask = new DARY(PATCH_SIZE,PATCH_SIZE);
//float PATCH_SUM;
void initPatchMask(int size){ 
	//cout << "DEBUGA" << endl << flush;
	int center=size>>1;
	float radius = center*center;
	float sigma=0.9*radius;
	float disq;
	for(int i=0;i<size;i++)
	  for(int j=0;j<size;j++){
	    disq=(i-center)*(i-center)+(j-center)*(j-center);
	    if(disq < radius){
	      patch_mask->fel[j][i]= exp(- disq / sigma);
	      //mask->fel[j][i]= 255*exp(- disq / sigma);   
	      //cout << patch_mask->fel[j][i]<< endl; 
	      //PATCH_SUM+=patch_mask->fel[j][i];
	    }else { 
	      patch_mask->fel[j][i]=0;
	    }		
	  } 
	
	//patch_mask->normalize(0,1);patch_mask->write("mask.pgm");cout << "mask "<< endl;getchar();
} 


