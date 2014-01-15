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
   - issue commands "qmake-qt4 -recursive; make; ./compileMatlab.sh" in the "src/libs" directory
   - issue commands "qmake-qt4 -recursive; make" in the "src/apps" directory;
   
3. Test the compiled code  

   - Issue the following commands in the code_test subdirectory:
   
   unzip code_test.zip
   ../run_partapp.sh --expopt ./expopt/exp-code-test-local-app-model.txt --head_detect_dpm --part_detect_dpm --find_obj
   ../run_partapp.sh --expopt ./expopt/exp-code-test-poselets.txt --save_resp_test
   ../run_partapp.sh --expopt ./expopt/exp-code-test-full-model.txt --find_obj --eval_segments --vis_segments

   This will run local appearance model, compute poselet responces and finally run full model to estimate body parts 
   on a provided image and visualize the results. 
   Compare the image in the ./log_dir/exp-code-test-full-model/part_marginals/seg_eval_images with the image
   in ./images_result

4. Running pose estimation experiments

   Download the experiments package from our homepage: https://www.d2.mpi-inf.mpg.de/poselet-conditioned-ps
   Unpack the package in the separate directory. Here is a short description of the contents:
 
   ./buffy_detections - prescaled and cropped correct HOG detections obtained on the Buffy dataset 
		      using the upper body detector from http://www.robots.ox.ac.uk/~vgg/software/UpperBody/index.html

   ./ramanan_people - directory where "People" dataset should be placed after downloading it from Dava Ramanan's homepage.
		      The dataset can be obtained from http://www.ics.uci.edu/~dramanan/papers/parse/people.zip
		      Prior to running the experiments the images should be converted from jpg to png format. If you have ImageMagick installed 
		      this can be done with the shell command:

		           for i in ./*jpg; do convert $i `basename $i jpg`png; done 

   ./ramanan_people_train_h200 - training set for full body pose estimation (first 100 images from "People" dataset + mirrored versions)

   ./ramanan_train_upperbody_h180 - training set for upper body pose estimation ( same as ramanan_people_train_h200 but different scaling and annotation)
			            Note that in the provided model only classifiers were trained on this data while joint parameters 
				    where estimated on the episode 4 of Buffy dataset, which was not for evaluation.

   ./tud_pedestrians_train - training data for people detection model. This is the same training dataset as was originally distributed with "TUD Pedestrians". 
			     You can download the data from http://tahiti.mis.informatik.tu-darmstadt.de/datasets/tracking-by-detection/train-400.tar.gz
			     if you want to retrain the model.

   ./tud_upright_people - TUD Upright People dataset. This is a smaller version of the "TUD Pedestrians" dataset, which contains more people and 
			  less background. The same pretrained model which is provided for this dataset can also be applied to "TUD Pedestrians".

   ./expopt - configuration files which control diverse parameters of the system

   ./log_dir/<EXP_NAME>/class - pretrained model (classifiers and joint parameters)

   At runtime the following directories will be created:				      
     ./log_dir/<EXP_NAME>/test_scoregrid - location where part detections will be stored 
     ./log_dir/<EXP_NAME>/part_marginals - location where part marginals will be stored
     
   Make sure that you have sufficient disc space before running the experiments on many images. 
   The code is currently storing part detections and part posteriors on disc, which for example for the "People" 
   dataset might requires around 25 MB per image.     

   Assuming that <PARTAPP_DIR> is the directory where you have unpacked and compiled the source code, you can run 
   part detectors and compute part posteriors for a single image by issuing the command:

   	<PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/<EXP_FILENAME> --part_detect --find_obj --first <IMGIDX> --numimgs 1

    where <EXP_FILENAME> is one of the experiment configuration files:

	  ./expopt/exp_buffy_hog_detections.txt
	  ./expopt/exp_ramanan_075.txt
	  ./expopt/exp_tud_upright_people.txt

    and <IMGIDX> is index of the image (if "first" and "numimgs" parameters are omitted the whole dataset will be processed).	 

    In order to evaluate the number of correctly detected parts run:
        
	<PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/<EXP_FILENAME> --eval_segments --first <IMGIDX> --numimgs 1     
	
    This command will also produce visualization of the max-marginal part estimates in the "part_marginals/seg_eval_images" directory.
    
    Object hypothesis can be extracted using the command:

	<PARTAPP_DIR>/run_partapp.sh --expopt ./expopt/<EXP_FILENAME> --save_res     
     	
    This will produce annotation files in the same format as training and test data. 

    
