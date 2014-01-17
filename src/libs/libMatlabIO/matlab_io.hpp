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

#ifndef _MATLAB_IO_HPP
#define _MATLAB_IO_HPP

/* MATLAB header file */
#include <mat.h>

//#include <boost/multi_array.hpp>
#include <libMultiArray/multi_array_def.h>

#include <iostream>

#include <QString>

#include <libMatlabIO/matlab_io.h>

namespace matlab_io {

  using std::cout;
  using std::endl;

  template <typename T> struct get_matlab_class_id 
  {
    static mxClassID id() {
      assert(false && "unknown type"); 
      return mxUNKNOWN_CLASS; 
    }
  };

  template<> struct get_matlab_class_id<float>
  {
    static mxClassID id() {
      return mxSINGLE_CLASS; 
    }
  };

  template<> struct get_matlab_class_id<double>
  {
    static mxClassID id() {
      return mxDOUBLE_CLASS; 
    }
  };

  
  /**
     save Boost MultiArray in Matlab format
   */
  template <typename Array>
  bool mat_save_multi_array(MATFile *f, QString qsVarName, const Array &A) 
  {
    assert(f != 0);
    const typename Array::size_type *shape_ptr = A.shape();
    boost::array<typename Array::size_type, Array::dimensionality> array_shape;
    for (uint i = 0; i < Array::dimensionality; ++i) 
      array_shape[i] = shape_ptr[i];

    // Matlab needs fortran storage order
    Array B(array_shape, boost::fortran_storage_order());  
    //Array B(array_shape, boost::fortran_storage_order);  
    B = A;  
   
    //  const typename Array::element *data1 = A.data(); 
    const typename Array::element *data2 = B.data();

    size_t nElements = A.num_elements();
    /*   for (uint i = 0; i < nElements; ++i)  */
    /*     cout << *(data1 + i) << " " << *(data2 + i) << endl;  */
  
    /* initialize matlab array */
    mwSize dims[Array::dimensionality];
    for (uint i = 0; i < Array::dimensionality; ++i) {
      dims[i] = array_shape[i];
      //cout << "dims " << i << ": " <<  dims[i] << endl;
    }

    mxArray *MA = mxCreateNumericArray(Array::dimensionality, dims, 
                                       get_matlab_class_id<typename Array::element>::id(), mxREAL);
    assert(MA != 0);

    if (A.num_elements() > 0) {
      typename Array::element *pMA = (typename Array::element *)mxGetPr(MA);
      assert(pMA != 0);
      for (uint i = 0; i < nElements; ++i) {
        *pMA = *(data2 + i);
        ++pMA;
      }
    }


//     mxArray *MA = mxCreateNumericArray(Array::dimensionality, dims, mxSINGLE_CLASS, mxREAL);
//     assert(MA != 0);

//     if (A.num_elements() > 0) {
//       float *pMA = (float *)mxGetPr(MA);
//       assert(pMA != 0);
//       for (uint i = 0; i < nElements; ++i) {
//         *pMA = *(data2 + i);
//         ++pMA;
//       }
//     }

    //cout << "matPutVariable" << endl;
    //matPutVariable(f, qsVarName, MA);
    matPutVariable(f, qsVarName.toStdString().c_str(), MA);
    //cout << "mxDestroyArray" << endl;
    mxDestroyArray(MA);
    //cout << "done" << endl;

    return true;
  }

  /**
     save Boost MultiArray in Matlab format
   */
  template <typename Array> 
  bool mat_save_multi_array(QString qsFilename, QString qsVarName, const Array &array)
  {
    cout << "mat_save_multi_array, qsFilename " << qsFilename.toStdString() << 
      ", qsVarName = " << qsVarName.toStdString() << endl;

    //MATFile *f = mat_open(qsFilename, "w");
    MATFile *f = mat_open(qsFilename, "wz");
    bool res = false;
    if (f != 0) {
      res = mat_save_multi_array(f, qsVarName, array);
      mat_close(f);
    }

    cout << "done" << endl;
    return res;
  }


  /**
     load Boost MultiArray from file in Matlab format 

   */
  template <typename Array>
  Array mat_load_multi_array(MATFile *f, QString qsVarName)
  {
    int verbosity = 0;
    if (verbosity > 0)
      cout << "loading variable: " << qsVarName.toStdString() << endl;
    const char *name = 0;

    if (f != 0) {
      mxArray *matlab_mat = matGetNextVariable(f, &name);
      if (verbosity > 0)
	cout << "\tnext variable: " << name << endl;

      while (matlab_mat != 0) {
        if (qsVarName == name) {
	  if (verbosity > 0)
	    cout << "\tfound variable: " << name << endl;
          mwSize ndim = mxGetNumberOfDimensions(matlab_mat);
          const mwSize *dims = mxGetDimensions(matlab_mat);
          //cout << "number of dimensions: " << ndim << endl;

          assert(ndim == Array::dimensionality);
          boost::array<typename Array::size_type, Array::dimensionality> array_shape;
          for (uint didx = 0; didx < Array::dimensionality; ++didx) {
            //cout << "extent of dim " << didx << ": " << dims[didx] << endl;
            array_shape[didx] = dims[didx];
          }

          Array grid(array_shape);
          Array B(array_shape, boost::fortran_storage_order());

          typename Array::element *pDataFortran = B.data();
          int nElements = B.num_elements();

          if (mxIsSingle(matlab_mat)) {
            float *pA = (float *)mxGetPr(matlab_mat);
            assert(pA != 0);
            for (int i = 0; i < nElements; ++i) {
              pDataFortran[i] = *pA;
              ++pA;
            }
          }
          else {
            double *pA = (double *)mxGetPr(matlab_mat);
            assert(pA != 0);
            for (int i = 0; i < nElements; ++i) {
              pDataFortran[i] = *pA;
              ++pA;
            }
          }

          //cout << "done reading " << nElements << " elements " << endl;

          /* automatically convert to c_storage_order */
          assert(grid.storage_order() == boost::c_storage_order());
          grid = B; 

          mxDestroyArray(matlab_mat);
          return grid;
        }
        
        mxDestroyArray(matlab_mat);
        matlab_mat = matGetNextVariable(f, &name);
      }// variables
    }// if loaded

    assert(false && "variable not found or could not open file");
    Array grid;
    return grid;
  }

  /**
     load Boost MultiArray from file in Matlab format 

     currently the whole array is copied (twice) on function return,
     since pass by reference needs resize() method which works on
     ExtentList. Such method is missing in current implementation of
     MultiArray, but boost 1.34 should have it.

     note: array type should be provided when calling a function: "mat_load_multi_array<ArrayType>(...);"

   */

  template <typename Array>
  Array mat_load_multi_array(QString qsFilename, QString qsVarName) 
  {
    MATFile *f = mat_open(qsFilename, "r");
    if (f == 0) {
      cout << "error opening " << qsFilename.toStdString() << endl;
      assert(false);
    }
      
    //assert(f != 0);
    Array ar = mat_load_multi_array<Array>(f, qsVarName);
    mat_close(f);

    return ar;
  }

  template <typename T> bool mat_save_stdcpp_vector(QString qsFilename, QString qsVarName, const std::vector<T> &v)
  {
    boost_math::double_vector v2(v.size());
    
    for (uint idx = 0; idx < v.size(); ++idx)
      v2(idx) = v[idx];

    return mat_save_double_vector(qsFilename, qsVarName, v2);
  }

  template <typename T> bool mat_save_stdcpp_vector(MATFile *f, QString qsVarName, const std::vector<T> &v)
  {
    assert(f != 0);

    boost_math::double_vector v2(v.size());
    
    for (uint idx = 0; idx < v.size(); ++idx)
      v2(idx) = v[idx];

    return mat_save_double_vector(f, qsVarName, v2);
  }


  template <typename T> bool mat_load_stdcpp_vector(QString qsFilename, QString qsVarName, std::vector<T> &v)
  {
    boost_math::double_vector v2(v.size());
    
    bool res = mat_load_double_vector(qsFilename, qsVarName, v2);

    if (res) {
      v.clear();
      v.resize(v2.size());

      for (uint idx = 0; idx < v2.size(); ++idx) {
        v[idx] = static_cast<T>(v2(idx));
      }
    }

    return res;
      
  }



}// namespace 

#endif
