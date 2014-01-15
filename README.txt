This short documentation describes steps necessary to compile and run the human pose estimation model
presented in the paper:

L. Pishchulin, M. Andriluka, P. Gehler and B. Schiele
Strong Appearance and Expressive Spatial Models for Human Pose Estimation
IEEE Conference on Computer Vision and Pattern Recognition (ICCV'13), Sydney, Australia, December 2013

This code was developed under Linux (Debian wheezy, 64 bit), and was tested only in this environment.

1. Required Libraries

   The following libraries are required to compile and run the code:
   
   - Qt (tested with version 4.8.2)
   - Boost (tested with version 1.34), http://www.boost.org/
   - Matlab (tested with Matlab 2008b)
   - Goolgle's Protocol Buffers (tested with version 2.0.1rc1), http://code.google.com/p/protobuf/

2. Compiling the code

   - Switch to the top level directory of the source code (the one with the script "run_partapp.sh").
   - Issue commands:
   
     ln -s ./external_include/matlab-2008b include_mat	
     ln -s ./external_lib/matlab-2008b-glnxa64-nozlib lib_mat		
     ln -s ./external_include include_pb
     ln -s ./external_lib lib_pb
     
   - add the full path of folder "include_mat" to the LD_LIBRARY_PATH environment variable
   - point to your matlab runtime environment by editing the file src/libs/libPrediction/matlab_runtime.h
   - issue commands "qmake-qt4 -recursive; make; ./compileMatlab.sh" in the "src/libs" directory
   - issue commands "qmake-qt4 -recursive; make" in the "src/apps" directory;

3. Test the compiled code  

   - Issue the following commands in the code_test subdirectory:
   
   unzip code_test.zip
   ../run_partapp.sh --expopt ./expopt/exp-code-test-local-app-model.txt --head_detect_dpm --part_detect_dpm --find_obj
   ../run_partapp.sh --expopt ./expopt/exp-code-test-poselets.txt --save_resp_test
   ../run_partapp.sh --expopt ./expopt/exp-code-test-full-model.txt --find_obj --eval_segments --vis_segments

   This will run local appearance model, compute poselet responses and finally run full model to estimate body parts 
   on a provided image and visualize the results. 
   Compare the image in the ./log_dir/exp-code-test-full-model/part_marginals/seg_eval_images with the image
   in ./images_result

4. Running pose estimation experiments

   Download the experiments package from our homepage: https://www.d2.mpi-inf.mpg.de/poselet-conditioned-ps
   Unpack the package in the separate directory. Here is a short description of the contents:
 
   ./images/LSP - upscaled images from the LSP dataset, so that every person is roughly 200px high
   ./expopt - configuration files which control diverse parameters of the system
   
   ./log_dir/exp-lsp-local-app-model/class - pretrained AdaBoost models (classifiers)
   ./log_dir/exp-lsp-local-app-model/dpm_model - pretrained DPM models
   ./log_dir/exp-lsp-local-app-model/sparial - pretrained generic spatial model
   ./log_dir/exp-lsp-local-app-model/test_dpm_unary/torso - responses of DPM torso detector

   ./log_dir/exp-lsp-train-torso/part_marginals - torso detections on the train set
   
   ./log_dir/exp-lsp-poselets/class - pretrained AdaBoost poselet models
   ./log_dir/exp-lsp-poselets/resp_train - poselet responses on train images
   ./log_dir/exp-lsp-poselets/pred_data - pretrained LDA classifiers to predict poselet conditioned parameters
   
   ./log_dir/exp-lsp-full-model/spatial - precomputed poselet conditioned spatial model
   
   Assuming that <PARTAPP_DIR> is the directory where you have unpacked and compiled the source code, you can run 
   the system on a single image by issuing the commands:
       	      
   <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-local-app-model.txt --head_detect_dpm --part_detect_dpm --find_obj --first <IMGIDX> --numimgs 1
   <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-poselets.txt --save_resp_test --first <IMGIDX> --numimgs 1
   <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --find_obj --first <IMGIDX> --numimgs 1
      
    where <IMGIDX> is index of the image (if "first" and "numimgs" parameters are omitted the whole dataset will be processed).	 

    In order to evaluate the number of correctly detected parts and visualize the results run:
        
   <PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/exp-lsp-full-model.txt --eval_segments --vis_segments --first <IMGIDX> --numimgs 1
   
   WARNING: this model evaluates AdaBoost detectors at every 4th pixel
   to speed up the computation. This results into overall performance
   of 68.7% PCP vs 69.2% PCP reported in the paper. To run the dense
   classifiers, please, set the value of "window_desc_step_ratio" to
   "0.125" in abcparams_rounds500_dense05_2000.txt
