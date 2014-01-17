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

#ifndef _MATLAB_CELL_IO_HPP_
#define _MATLAB_CELL_IO_HPP_

#include <libMultiArray/multi_array_def.h>

#include <libMatlabIO/matlab_io.h>

namespace matlab_io
{
  template <typename Array>
  bool mat_load_multi_array_vec2(MATFile *f, QString qsVarName, std::vector<std::vector<Array> > &vvA)
  {
    const char *name = 0;
    bool bRes = false;
    vvA.clear();

    if (f != 0) {
      mxArray *matlab_mat = matGetNextVariable(f, &name);
      std::cout << name << std::endl;

      bool bLoaded = false;

      while (matlab_mat != 0 && !bLoaded) {
        if (qsVarName == name) {
          if (mxIsCell(matlab_mat)) {
            mwSize cell_ndim = mxGetNumberOfDimensions(matlab_mat);
            const mwSize *cell_dims = mxGetDimensions(matlab_mat);
            //std::cout << "number of cell dimensions: " << cell_ndim << endl;

            assert(cell_ndim == 2);
            for (int dim_idx = 0; dim_idx < 2; ++dim_idx) {
              std::cout << "extent of cell dim " << dim_idx << ": " << cell_dims[dim_idx] << std::endl;
            }            

            uint N1 = cell_dims[0];
            uint N2 = cell_dims[1];            
            
            //vvA.resize(N1, std::vector<Array>(N2, Array()));

            for (uint i1 = 0; i1 < N1; ++i1) {
              vvA.push_back(std::vector<Array>());
              for (uint i2 = 0; i2 < N2; ++i2) {

                /* get dimensions of the current cell */
                mwIndex subs[2];
                subs[0] = i1;
                subs[1] = i2;
                mwIndex cell_idx = mxCalcSingleSubscript(matlab_mat, cell_ndim, subs);
                mxArray *E = mxGetCell(matlab_mat, cell_idx);

                mwSize mat_ndim = mxGetNumberOfDimensions(E);
                const mwSize *mat_dims = mxGetDimensions(E);
                //std::cout << "number of cell element dimensions: " << mat_ndim << endl;
                assert(mat_ndim == Array::dimensionality);

                /* copy from matlab matrix to Array */
                boost::array<typename Array::size_type, Array::dimensionality> array_shape;
                for (uint i = 0; i < Array::dimensionality; ++i) 
                  array_shape[i] = mat_dims[i];

                Array B(array_shape, boost::fortran_storage_order());  
                uint nElements = B.num_elements();

                if (nElements > 0) {
                  typename Array::element *data2 = B.data();
                  if (mxIsSingle(E)) {
                    float *pE = (float *)mxGetPr(E);
                    assert(pE != 0);
                    for (uint i = 0; i < nElements; ++i) {
                      *(data2 + i) = *pE;
                      ++pE;
                    }
                  }
                  else {
                    double *pE = mxGetPr(E);
                    assert(pE != 0);
                    for (uint i = 0; i < nElements; ++i) {
                      *(data2 + i) = *pE;
                      ++pE;
                    }
                  }

                  //vvA[i1][i2] = B;

                  /* copy to array with normal storage order */
                  // Array A = B; // this does not work!!! (storage order is of course copied at construction :)
                  Array A(array_shape);
                  A = B;
                  
                  vvA.back().push_back(A);
                }

              }
            }// cell elements

            bRes = true;
          }// is cell
          else {
            std::cout << "variable is not a cell array" << std::endl;
          }

          bLoaded = true;
        }

        mxDestroyArray(matlab_mat);
        matlab_mat = 0;

        if (!bLoaded)
          matlab_mat = matGetNextVariable(f, &name);

      }// variables
    }

    assert(bRes && "variable not found or could not open file");
    return bRes;
  }

  template <typename Array>
  bool mat_load_multi_array_vec2(QString qsFilename, QString qsVarName, std::vector<std::vector<Array> > &vvA)
  {
    std::cout << "mat_load_multi_array_vec2 " << qsFilename.toStdString() << std::endl;
    MATFile *f = mat_open(qsFilename, "r");
    assert(f != 0);
    mat_load_multi_array_vec2(f, qsVarName, vvA);
    mat_close(f);
    return true;
  }

  template <typename Array>
  bool mat_save_multi_array_vec2(MATFile *f, QString qsVarName, const std::vector<std::vector<Array> > &vvA)
  {
    assert(f != 0);
    
    mwSize cell_ndim = 2;
    mwSize cell_dims[2];
    
    assert(vvA.size() > 0);
    uint N1 = vvA.size();
    uint N2 = vvA[0].size();

    cell_dims[0] = N1;
    cell_dims[1] = N2;

    mxArray *CA = mxCreateCellArray(cell_ndim, cell_dims);
    assert(CA != 0);

    for (uint i1 = 0; i1 < N1; ++i1) {
      assert(vvA.size() > i1);
      assert(vvA[i1].size() == N2);

      for (uint i2 = 0; i2 < N2; ++i2) {

        /* init shape array */
        const typename Array::size_type *shape_ptr = vvA[i1][i2].shape();
        assert(shape_ptr != 0);

        boost::array<typename Array::size_type, Array::dimensionality> array_shape;
        for (uint i = 0; i < Array::dimensionality; ++i) 
          array_shape[i] = shape_ptr[i];

        /* create array of the same shape as initial array with data stored in fortran storage order */
        Array B(array_shape, boost::fortran_storage_order());  
        B = vvA[i1][i2];  
    
        const typename Array::element *data2 = B.data();
        assert(data2 != 0);

        /* shape again, this time as mwSize array */
        mwSize dims[Array::dimensionality];
        for (uint i = 0; i < Array::dimensionality; ++i) 
          dims[i] = array_shape[i];

        mxArray *E = mxCreateNumericArray(Array::dimensionality, dims, mxSINGLE_CLASS, mxREAL);    
        assert(E != 0);
    
        /* copy elements to matlab array */
        size_t nElements = vvA[i1][i2].num_elements();
        if (nElements > 0) {
          float *pE = (float *)mxGetPr(E);
          assert(pE != 0);
          for (uint idx = 0; idx < nElements; ++idx) {
            *pE = *(data2 + idx);
            ++pE;
          }
        }

        /* add element to cell array */
        mwIndex subs[2];
        subs[0] = i1;
        subs[1] = i2;
        mwIndex cell_idx = mxCalcSingleSubscript(CA, cell_ndim, subs);
        //std::cout << "cell index: " << cell_idx << std::endl;
        mxSetCell(CA, cell_idx, E);
      }
    }

    matPutVariable(f, qsVarName.toStdString().c_str(), CA);
    
    mxDestroyArray(CA);
    return true;
  }

  template <typename Array>
  bool mat_save_multi_array_vec2(QString qsFilename, QString qsVarName, const std::vector<std::vector<Array> > &vvA)
  {
    MATFile *f = mat_open(qsFilename, "wz");
    assert(f != 0);
    mat_save_multi_array_vec2(f, qsVarName, vvA);
    mat_close(f);
    return true;
  }

}


#endif

