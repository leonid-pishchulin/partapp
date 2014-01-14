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

#ifndef _SHAPE_DESCRIPTOR_H
#define _SHAPE_DESCRIPTOR_H

#include <vector>

#include <libBoostMath/boost_math.h>

#include "kmaimagecontent.h"
#include "ImageContent/imageContent.h"

namespace kma{

  void precompute_patch_idx(boost_math::double_matrix &patch_lcpos, boost_math::double_matrix &patch_lrpos, 
                            int iradius, int rSize, int cSize);

  void KMAcomputeShape(const boost_math::double_matrix &patch_lcpos, 
                       const boost_math::double_matrix &patch_lrpos, 
                       DARY *img, std::vector<float> &vec);

  namespace shape {
    const int SrSize=3; 
    const int ScSize=4; 
    const int SOriSize = 8;
    const float MaxIndexVal = 0.2;  /* Good value is 0.2 */
  };

};


#endif
