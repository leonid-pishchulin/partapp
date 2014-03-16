PARTAPP - Human Pose Estimation
=====

This short documentation describes steps necessary to compile and run the human pose estimation model presented in the paper:

**Leonid Pishchulin, Micha Andriluka, Peter Gehler and Bernt Schiele  
Strong Appearance and Expressive Spatial Models for Human Pose Estimation  
In _IEEE International Conference on Computer Vision (ICCV'13)_, Sydney, Australia, December 2013**

Required Libraries
---

The following libraries are required to compile and run the code:

   - [Qt] (http://qt-project.org/) (tested with version 4.8.2)
   - [Boost] (http://www.boost.org/) (tested with version 1.49)
   - Matlab (tested with Matlab 2013a)
   - Matlab compiler mcc
   - MATLAB Compiler Runtime (MCR) (tested with R2013a (8.1))
   - Goolgle's [Protocol Buffers] (http://code.google.com/p/protobuf/) (tested with version 2.0.1rc1)

Compiling
---

This code was developed under Linux (Debian _wheezy_, 64 bit) and was tested only in this environment.  

1. Switch to the top level directory of the source code (the one with the script `run_partapp.sh`), issue commands:

    ```
    ln -s ./external_include/matlab-2008b include_mat  
    ln -s ./external_lib/matlab-2008b-glnxa64-nozlib lib_mat  
    ln -s ./external_include include_pb  
    ln -s ./external_lib lib_pb
    ```

2. Add the full path of folders `lib_mat` and `lib_pb` to the `LD_LIBRARY_PATH` environment variable  
3. Download and install Matlab Compiler Runtime (MCR)
4. Point to your MCR by editing the file `src/libs/libPrediction/matlab_runtime.h`  
5. Issue commands `qmake-qt4 -recursive; make; ./compileMatlab.sh` in the `src/libs` directory
6. Issue commands `qmake-qt4 -recursive; make` in the `src/apps` directory

Testing
---

1. Issue the following commands in the `code_test` subdirectory:

   ```
   unzip code_test.zip 
   ../run_partapp.sh --expopt ./expopt/exp-code-test-local-app-model.txt --head_detect_dpm --part_detect_dpm --find_obj 
   ../run_partapp.sh --expopt ./expopt/exp-code-test-poselets.txt --save_resp_test 
   ../run_partapp.sh --expopt ./expopt/exp-code-test-full-model.txt --find_obj --eval_segments --vis_segments
   ```

This will run local appearance model, compute poselet responses and finally run full model to estimate body parts on a provided image and visualize the results. Compare the image in the `./log_dir/exp-code-test-full-model/part_marginals/seg_eval_images` with the image in `./images_result`

Running pose estimation experiments
---

   Our model requires that training and testing images contain single persons being roughy 200 px high.
   Download the experiments package from our project webpage: https://www.d2.mpi-inf.mpg.de/poselet-conditioned-ps
   Unpack the package in the separate directory `EXP_DIR`. Here is a short description of the contents:

   - `./images/LSP` - upscaled images from the LSP dataset, so that every person is roughly 200 px high
   - `./expopt` - configuration files which control diverse parameters of the system
   - `./log_dir/exp-lsp-local-app-model/class` - pretrained AdaBoost models (classifiers)
   - `./log_dir/exp-lsp-local-app-model/dpm_model` - pretrained DPM models
   - `./log_dir/exp-lsp-local-app-model/sparial` - pretrained generic spatial model
   - `./log_dir/exp-lsp-local-app-model/test_dpm_unary/torso` - responses of the DPM torso detector
   - `./log_dir/exp-lsp-train-torso/part_marginals` - torso detections on the train set
   - `./log_dir/exp-lsp-poselets/class` - pretrained AdaBoost poselet models
   - `./log_dir/exp-lsp-poselets/resp_train` - poselet responses on train images
   - `./log_dir/exp-lsp-poselets/pred_data` - pretrained LDA classifiers to predict poselet conditioned parameters
   - `./log_dir/exp-lsp-full-model/spatial` - precomputed poselet conditioned spatial model

### Test
Assuming that `PARTAPP_DIR` is the directory where you have unpacked and compiled the source code, you can run the system on a single image by issuing the commands:

    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-local-app-model.txt --head_detect_dpm --part_detect_dpm --find_obj --first <IMGIDX> --numimgs 1
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-poselets.txt --save_resp_test --first <IMGIDX> --numimgs 1
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --find_obj --first <IMGIDX> --numimgs 1

where `IMGIDX` is index of the image (if `--first` and `--numimgs` parameters are omitted the whole dataset will be processed).

In order to evaluate the number of correctly detected parts and visualize the results run:

     <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --eval_segments --vis_segments --first <IMGIDX> --numimgs 1

WARNING: this model evaluates AdaBoost detectors at every 4th pixel to speed up the computation. This results into overall performance of _68.7%_ vs _69.2% PCP_ reported in the paper. To run slower dense classifiers, set `window_desc_step_ratio: 0.0125` in `./expopt/abcparams_rounds500_dense05_2000.txt`

### Train + Test

1. **local appearance model**

 1) train AdaBoost model for part `PARTIDX` 0 - 21 (22 parts total)
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-local-app-model.txt --train_class --pidx <PARTIDX>
```
 2) train generic spatial model
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-local-app-model.txt --pc_learn
```
 3) compute torso position prior
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-local-app-model.txt --compute_pos_prior
```
 4) train DPM part detectors (_in matlab_)
```
     cd <PARTAPP_DIR>/src/libs/libDPM
     trainDPM(<PARTIDX>, <EXP_DIR>)
```
    where `EXP_DIR` is the root of the experiments package

 5) train DPM head detector (in matlab)
```
     cd <PARTAPP_DIR>/src/libs/libDPM
     trainDPM(11, <EXP_DIR>, true)
```
 6) run local appearance model
```
     <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-local-app-model.txt --head_detect_dpm --part_detect_dpm --find_obj
```

2. **poselets**

  1) run torso detector on training images
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-train-torso.txt --find_obj --first <IMGIDX> --numimgs 1
```
  2) train poselet AdaBoost detectors
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-poselets.txt --train_class --pidx <PARTIDX> --tidx <TYPEIDX>
```
    where `PARTIDX` is poselet id, [0, 20]; `TYPEIDX` is poselet mixture type, [0, 100)

  3) collect poselet responses on training and testing images
```
     <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-poselets.txt --save_resp_train --first <IMGIDX> --numimgs 1
     <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-poselets.txt --save_resp_test --first <IMGIDX> --numimgs 1
```
  4) train LDA classifiers
```
     <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-poselets.txt  --train_lda_urot --train_lda_upos --train_lda_pwise
```
3. **full model**

  1) compute poselet conditioned mixtures
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --pc_learn_types
```
  2) run full model
```
    PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --find_obj --first <IMGIDX> --numimgs 1
```
  3) evaluate and visualize the results
```
    <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --find_obj --first <IMGIDX> --numimgs 1
```

### Managing annotations

Loading/saving annotation files in matlab:

    cd <PARTAPP_DIR>/src/scripts/matlab
    annotations = loadannotations(<annotations.al>);
    saveannotations(annotations, <annotations.al>);


### Running on your data

1. rescale images

As our model runs on a single scale, you need to rescaled the images first. After rescaling each shown person should be roughly 200 px high, if the person were staying upright. This can be done by using head size as a reference.

2. save images as png

The best way is to use matlab to convert the images.

3. create annotation list in .al format

    cd <PARTAPP_DIR>/src/scripts/matlab
    prepare_data_file(<path/to/images/>);

**We may also run our model on your data**

For more information visit our project web page https://www.d2.mpi-inf.mpg.de/poselet-conditioned-ps  
If you have any questions, send an email to leonid@mpi-inf.mpg.de with a topic "partapp code".  
