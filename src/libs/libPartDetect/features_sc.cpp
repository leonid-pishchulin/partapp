#include "FeatureGrid.h"
#include <libKMA2/kmaimagecontent.h>
#include <libKMA2/ShapeDescriptor.h>
#include "descriptorGrid.h"
#include "mex.h"

/*
  F = features(image,window_step,desc_size)
  image:       double RGB image
  window_step: step to extract descriptors
  desc_size:   descriptor size = round(scale*18),
               where scale 1 corresponds to 
	       200 px high people
*/ 
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]) {
      
  const mxArray *mximage = prhs[0];
  double* img = (double *)mxGetPr(mximage);
  const int *dims = mxGetDimensions(mximage);
  if (mxGetNumberOfDimensions(mximage) != 3 ||
      dims[2] != 3 ||
      mxGetClassID(mximage) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input");
  
  int window_step = mxGetScalar(prhs[1]);
  int desc_size   = mxGetScalar(prhs[2]);
  
  uint img_height = dims[0];
  uint img_width = dims[1];
  
  // create a data structure
  kma::ImageContent* kmaimg = new kma::ImageContent::ImageContent(img_height, img_width, "3float", 0);
  for (int ix = 0; ix < img_width; ++ix) 
    for (int iy = 0; iy < img_height; ++iy){ 
      kmaimg->felr[iy][ix] = img[iy+ix*img_height                         ];
      kmaimg->felg[iy][ix] = img[iy+ix*img_height +   img_height*img_width];
      kmaimg->felb[iy][ix] = img[iy+ix*img_height + 2*img_height*img_width];
    }
  kmaimg->toGRAY();
  
  // no rotation
  boost_math::double_vector ax(2);boost_math::double_vector ay(2);
  ax(0) = 1.0; ax(1) = 0.0; ay(0) = -0.0; ay(1) = 1.0;
  
  // compute features
  FeatureGrid feature_grid(kmaimg->x(), kmaimg->y(), ax, ay, window_step, desc_size);
  part_detect::computeDescriptorGrid(kmaimg, feature_grid);
    
  delete kmaimg;
  
  int descSize = kma::shape::SrSize * kma::shape::ScSize * kma::shape::SOriSize;

  /* Create an mxArray for the output data */
  //plhs[0] = mxCreateDoubleMatrix(feature_grid.ny*feature_grid.nx, descSize, mxREAL);
  int ndim = 1, adims[1] = {feature_grid.ny*feature_grid.nx*descSize};
  plhs[0] = mxCreateNumericArray(ndim, adims, mxDOUBLE_CLASS, mxREAL);
  
  /* Create a pointer to the output data */
  double* features = mxGetPr(plhs[0]);
  for (int ix = 0; ix < feature_grid.nx; ++ix) 
    for (int iy = 0; iy < feature_grid.ny; ++iy) 
      if (feature_grid.desc[iy][ix].size() == 0)
	for (int d = 0; d < descSize; ++d) 
	  features[d+descSize*(iy+feature_grid.ny*ix)] = 0;
      else
	for (int d = 0; d < descSize; ++d) 
	  features[d+descSize*(iy+feature_grid.ny*ix)] = feature_grid.desc[iy][ix][d];

  /*
  int imgNameLen = (mxGetM(prhs[2]) * mxGetN(prhs[2])) + 1;
  char *imgName = (char*) malloc (imgNameLen);
  int status = mxGetString(prhs[2], imgName, imgNameLen); 
  if (status != 0){
    mexErrMsgTxt("image name is wrong\n");
  }
  
  // load image
  kma::ImageContent *kmaimg2 = kma::load_convert_gray_image(imgName);
  for (int iy = 0; iy < 10; ++iy){ 
    for (int ix = 0; ix < 10; ++ix){ 
      mexPrintf("(%1.0f %1.0f %1.0f) ", kmaimg->felr[iy][ix], kmaimg->felg[iy][ix], kmaimg->felb[iy][ix]);
    }
    mexPrintf("\n");
  }
  
  mexPrintf("Compare images\n");
  for (int iy = 0; iy < 10; ++iy){ 
    for (int ix = 0; ix < 10; ++ix){ 
      mexPrintf("(%1.0f %1.0f) ", kmaimg->fel[iy][ix], kmaimg2->fel[iy][ix]);
    }
    mexPrintf("\n");
  }
  
  delete kmaimg2;
  */
}
