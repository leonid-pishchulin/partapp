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

#include <libBoostMath/boost_math.h>

#include "ShapeDescriptor.h"
#include "gauss_iir/gauss_iir.h"
#include "descriptor/feature.h"

using namespace std;

namespace kma {

  /* Normalize length of vec to 1.0.
   */
  void KMANormalizeVect(vector<float> &vec)
  {
    int i;
    float val, fac, sqlen = 0.0;
    int len = vec.size();

    for (i = 0; i < len; i++) {
      val = vec[i];
      sqlen += val * val;
    }
    fac = 1.0 / sqrt(sqlen);
    for (i = 0; i < len; i++)
      vec[i] *= fac;
  }

  void precompute_patch_idx(boost_math::double_matrix &patch_lcpos, boost_math::double_matrix &patch_lrpos, 
                            int iradius, int rSize, int cSize)
  {
    patch_lcpos.resize(2*iradius + 1, 2*iradius + 1);
    patch_lrpos.resize(2*iradius + 1, 2*iradius + 1);

    float cspacing = (cSize) / (M_2PI);

    for (int i = -iradius; i <= iradius; i++)
      for (int j = -iradius; j <= iradius; j++) {
       
        int rpos = i;
        int cpos = j;

        double lcpos=(M_PI+atan2(rpos,cpos))*cspacing;
        double lrpos=log(1+sqrt((float)i*i+j*j)/iradius)*rSize;
    
        patch_lcpos(iradius + j, iradius + i) = lcpos;
        patch_lrpos(iradius + j, iradius + i) = lrpos;

      }// patch 
    
  }



  /* Increment the appropriate locations in the index to incorporate
     this image sample.  The location of the sample in the index is (rx,cx). 
  */

  void KMAPlaceInLogPolIndex(vector<float> &index,
                             float mag, float ori, float rx, float cx, int oriSize, int rSize, int cSize, 
                             bool bFullOrientation)
  {
    int r, c, ort, ri, ci, oi, rindex, cindex, oindex, rcindex;
    float oval, rfrac, cfrac, ofrac, rweight, cweight, oweight;
   
    if (bFullOrientation)
      oval = oriSize * ori / (M_2PI);
    else
      oval = oriSize * ori / (M_PI);
   
    ri = (int)((rx >= 0.0) ? rx : rx - 1.0);  /* Round down to next integer. */
    ci = (int)((cx >= 0.0) ? cx : cx - 1.0);
    oi = (int)((oval >= 0.0) ? oval : oval - 1.0);
    rfrac = rx - ri;         /* Fractional part of location. */
    cfrac = cx - ci;
    ofrac = oval - oi; 
    /*   assert(ri >= -1  &&  ri < IndexSize  &&  oi >= 0  &&  oi <= OriSize  && rfrac >= 0.0  &&  rfrac <= 1.0);*/
    //cout << ri << " " << ci << " " << oi << endl;
    /* Put appropriate fraction in each of 8 buckets around this point
       in the (row,col,ori) dimensions.  This loop is written for
       efficiency, as it is the inner loop of key sampling. */
    for (r = 0; r < 2; r++) {
      rindex = ri + r;
      if (rindex >=0 && rindex < rSize) {
        rweight = mag * ((r == 0) ? 1.0 - rfrac : rfrac);
         
        for (c = 0; c < 2; c++) {
          cindex = ci + c;
          if(cindex >= cSize)
            cindex=0;	    
          if (cindex >=0 && cindex < cSize) {
            cweight = rweight * ((c == 0) ? 1.0 - cfrac : cfrac);
            rcindex=(rindex*cSize+cindex)<<3;//remember when you change the orientation number
            for (ort = 0; ort < 2; ort++) {
              oindex = oi + ort;
              if (oindex >= oriSize)  /* Orientation wraps around at PI. */
                oindex = 0;
              oweight = cweight * ((ort == 0) ? 1.0 - ofrac : ofrac);
              //cout << rcindex+oindex<< endl;
              index[rcindex+oindex]+=oweight;
            }
          }  
        }
      }
    } 
  }


  /* Given a sample from the image gradient, place it in the index array.
   */
  void KMAAddLogPolSample(vector<float> &index,
                         DARY *grad, DARY *orim, float angle, int r, int c, float rpos, float cpos,
                         float rx, float cx, int oriSize, int rSize, int cSize)
  {

    /**
       at this point grad should be a binary edge image returned by cannyEdges
     */
    if (grad->fel[r][c] < 1e-10)
      return;

    float mag, ori;
    
    /* Clip at image boundaries. */
    if (r < 0  ||  r >= (int)grad->y()  ||  c < 0  ||  c >= (int)grad->x())
      return;
    
    mag = patch_mask->fel[r][c] * grad->fel[r][c];
    /* Subtract keypoint orientation to give ori relative to keypoint. */
    ori = orim->fel[r][c]-angle;
    
    /* Put orientation in range [0, 2*PI].  If sign of gradient is to
       be ignored, then put in range [0, PI]. */
   
    bool bFullOrientation = false;

    if (bFullOrientation) {
      while (ori > M_2PI)
        ori -= M_2PI;
      
      while (ori < 0.0)
        ori += M_2PI;     
    }
    else {
      while (ori > M_PI)
        ori -= M_PI;
      
      while (ori < 0.0)
        ori += M_PI;     
    }

    KMAPlaceInLogPolIndex(index, mag, ori, rx, cx, oriSize, rSize, cSize, bFullOrientation);
  } 


  void KMAcomputeShape(const boost_math::double_matrix &patch_lcpos, 
                       const boost_math::double_matrix &patch_lrpos, 
                       DARY *img, vector<float> &vec)
  {
    const int _ShapeSize = kma::shape::SrSize * kma::shape::ScSize * kma::shape::SOriSize;
    assert((int)vec.size() == _ShapeSize);

    uint patch_width = PATCH_SIZE;
    uint patch_height = PATCH_SIZE;

    assert(img->x() == patch_width && img->y() == patch_height);
    assert(patch_lcpos.size1() == patch_height && patch_lcpos.size2() == patch_width); 
    assert(patch_lrpos.size1() == patch_height && patch_lrpos.size2() == patch_width);

    int oriSize = kma::shape::SOriSize;
    int rSize = kma::shape::SrSize;
    int cSize = kma::shape::ScSize;

    DARY *grad = new DARY(PATCH_SIZE,PATCH_SIZE);
    DARY *ori = new DARY(PATCH_SIZE,PATCH_SIZE);    
    DARY *dx = new DARY(img->y(),img->x());
    DARY *dy = new DARY(img->y(),img->x());
    DARY *edge = new DARY(img->y(),img->x());

    dX2(img,dx);
    dY2(img,dy);
    for(uint j=0;j<grad->y();j++){
      for(uint i=0;i<grad->x();i++){
        grad->fel[j][i]=sqrt(dx->fel[j][i]*dx->fel[j][i]+dy->fel[j][i]*dy->fel[j][i]); 
        ori->fel[j][i]=atan2(dy->fel[j][i],dx->fel[j][i]);
      }
    } 

    /* initialize edge image */
    memset(edge->fel[0], 0, edge->x()*edge->y()*sizeof(float));

    cannyEdges(dx, dy, grad, edge, 5, 15);
    delete dx; delete dy; delete grad;

    /* begin of KeyLogPolSample(vec, edge, ori,  angle, SOriSize, SrSize, ScSize); */
    int iradius = (int)floor(PATCH_SIZE/2);

    for (int i = -iradius; i <= iradius; i++)
      for (int j = -iradius; j <= iradius; j++) {
        //         lcpos=(M_PI+atan2(rpos,cpos))*cspacing;
        //         lrpos=log(1+sqrt((float)i*i+j*j)/iradius)*rSize;

        double lcpos = patch_lcpos(iradius + j, iradius + i);
        double lrpos = patch_lrpos(iradius + j, iradius + i);
        double rx = lrpos;// + (rSize - 1) / 2.0;
        double cx = lcpos;// + (cSize - 1) / 2.0;

        if (rx > -1.0 && rx < (float) rSize  &&
            cx > -1.0 && cx < (float) cSize) {
          //cout << "in" << cpos << " " << rpos << endl;
          KMAAddLogPolSample(vec, edge, ori, 0.0, 
                            iradius + i, iradius + j, 
                            lrpos, lcpos,  /* not used ? */ 
                            lrpos, lcpos,  /* rx, cx */
                            oriSize, rSize, cSize);

        }

        

      }
    /* end of KeyLogPolSample */

    delete edge; delete ori;
    KMANormalizeVect(vec);

    int intval, changed = FALSE;
    //for (int i = 0; i < kma::shape::ShapeSize; i++)
    for (int i = 0; i < _ShapeSize; i++)
      if (vec[i] > kma::shape::MaxIndexVal) { 
        vec[i] = kma::shape::MaxIndexVal;
        changed = TRUE;
      }

    if (changed) {

      KMANormalizeVect(vec);
    }

    /* Convert float vector to integer. 
       Assume largest value in normalized
       vector is likely to be less than 0.5. */
    //for (int i = 0; i < kma::shape::ShapeSize; i++) {
    for (int i = 0; i < _ShapeSize; i++) {
      intval = (int) (512.0 * vec[i]);
      vec[i] = (255 < intval) ? 255 : intval;
    }
   
  }



}
