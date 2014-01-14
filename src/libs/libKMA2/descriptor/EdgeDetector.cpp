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

#include <vector>

using namespace std;

void cannyEdges(DARY *img, DARY *edge,  float scale, float lower_threshold, float higher_threshold){
  
  DARY *dx = new DARY(img->y(),img->x());
  DARY *dy = new DARY(img->y(),img->x());
  DARY *grad = new DARY(img->y(),img->x());

  dX(img,dx,scale);
  dY(img,dy,scale);
  for(uint j=0;j<grad->y();j++){
    for(uint i=0;i<grad->x();i++){
      grad->fel[j][i]=sqrt(dx->fel[j][i]*dx->fel[j][i]+dy->fel[j][i]*dy->fel[j][i]); 	
    }
  } 
  cannyEdges(dx, dy, grad, edge, lower_threshold, higher_threshold); 

  delete dx;delete dy;delete grad;
}

void cannyEdges(DARY *dx,DARY *dy, DARY *edge,  float lower_threshold, float higher_threshold){
  
  DARY *grad = new DARY(dx->y(),dx->x());

  for(uint j=0;j<grad->y();j++){
    for(uint i=0;i<grad->x();i++){
      grad->fel[j][i]=sqrt(dx->fel[j][i]*dx->fel[j][i]+dy->fel[j][i]*dy->fel[j][i]); 	
    }
  } 
  cannyEdges(dx, dy, grad, edge, lower_threshold, higher_threshold); 

  delete grad;
}

void cannyEdges(DARY *img, DARY *edge,  float lower_threshold, float higher_threshold){
  
  DARY *dx = new DARY(img->y(),img->x());
  DARY *dy = new DARY(img->y(),img->x());
  DARY *grad = new DARY(img->y(),img->x());

  dX2(img,dx);
  dY2(img,dy);
  for(uint j=0;j<grad->y();j++){
    for(uint i=0;i<grad->x();i++){
      grad->fel[j][i]=sqrt(dx->fel[j][i]*dx->fel[j][i]+dy->fel[j][i]*dy->fel[j][i]); 	
    }
  } 
  cannyEdges(dx, dy, grad, edge, lower_threshold, higher_threshold); 

  delete dx;delete dy;delete grad;
}

void cannyEdges(DARY *dx, DARY *dy, DARY *grad, DARY *edge,  float lower_threshold,
	   float higher_threshold){

  DARY *tmp_edge = new DARY(dx->y(),dx->x(),0.0);
  vector<int> cor_edge;cor_edge.push_back(0);
  float color=255;
  float x1,y1,x2,y2,ux,uy,g,g1,g2,nb=1;
  for(uint j=1;j<grad->y()-2;j++){
    for(uint i=1;i<grad->x()-2;i++){
      g=grad->fel[j][i];
      if(g<lower_threshold)continue;
      ux=dx->fel[j][i];
      uy=dy->fel[j][i];
      x1=i+ux/g;
      y1=j+uy/g;
      x2=i-ux/g;
      y2=j-uy/g;
      g1=grad->getValue(x1,y1);
      g2=grad->getValue(x2,y2);    
      if(g<=g1 || g<=g2)continue;
      else if(g>higher_threshold)edge->fel[j][i]=color;
      else if(g>lower_threshold && (edge->fel[j-1][i-1]==color  ||
				    edge->fel[j-1][i]==color ||
				    edge->fel[j-1][i+1]==color ||
				    edge->fel[j][i-1]==color)){	
	edge->fel[j][i]=color;}
      else if(g>lower_threshold){
	if(tmp_edge->fel[j-1][i-1]>0){
	  tmp_edge->fel[j][i]=tmp_edge->fel[j-1][i-1];
	}
	else if(tmp_edge->fel[j-1][i]>0){
	  tmp_edge->fel[j][i]=tmp_edge->fel[j-1][i];
	}
	else if(tmp_edge->fel[j-1][i+1]>0){
	  tmp_edge->fel[j][i]=tmp_edge->fel[j-1][i+1];
	}
	else if(tmp_edge->fel[j][i-1]>0){
	  tmp_edge->fel[j][i]=tmp_edge->fel[j][i-1];
	}
	else if(tmp_edge->fel[j][i+1]>0){
	  tmp_edge->fel[j][i]=tmp_edge->fel[j][i+1];
	}else {
	  tmp_edge->fel[j][i]=nb;cor_edge.push_back(0);nb++;
	}
      }
    }
  }
  //edge->write("edge.pgm");getchar();
  
  for(int j=edge->y()-2;j>0;j--){
    for(int i=edge->x()-2;i>0;i--){
	if((edge->fel[j-1][i-1]==color ||
	    edge->fel[j-1][i]==color ||
	    edge->fel[j-1][i+1]==color ||
	    edge->fel[j][i-1]==color ||
	    edge->fel[j][i+1]==color ||
	    edge->fel[j+1][i-1]==color ||
	    edge->fel[j+1][i]==color ||
	    edge->fel[j+1][i+1]==color))cor_edge[(int)tmp_edge->fel[j][i]]=1;	
    }
  }  
  for(int j=edge->y()-2;j>0;j--){
    for(int i=edge->x()-2;i>0;i--){
      if(tmp_edge->fel[j][i]>0 && cor_edge[(int)tmp_edge->fel[j][i]]==1){	
	edge->fel[j][i]=color;
      }
    }
  } 

  for(uint j=0;j<grad->y();j++){
    for(uint i=0;i<grad->x();i++){
      if(edge->fel[j][i]>0)edge->fel[j][i]=grad->fel[j][i];
      else edge->fel[j][i]=0;
    }
  }
  delete tmp_edge;
}

