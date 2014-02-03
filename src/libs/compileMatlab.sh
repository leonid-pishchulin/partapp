#!/bin/bash

cd ../../
PARTAPP_DIR=$PWD
EXT_MAT_DIR=$PARTAPP_DIR/external_mat
echo $PARTAPP_DIR
echo $EXT_MAT_DIR
cd - 
MATLABPATH=$EXT_MAT_DIR/sparseLDA_v2/:$EXT_MAT_DIR/larsen/:$PARTAPP_DIR/src/libs/libPrediction/:$PARTAPP_DIR/src/libs/libDPM/:${MATLABPATH}
echo $MATLABPATH

cd libPrediction
echo "mcc -m trainClass.m -v -a ./ -a $EXT_MAT_DIR/sparseLDA_v2/ -a $EXT_MAT_DIR/larsen/ -a ../../scripts/matlab/ -N -p images/ -p stats/ ; quit" | matlab -nojvm -nosplash -nodisplay
echo "mcc -m predictFactorsImg.m -v -a ./ -a ../../scripts/matlab/ -N -p images/  -p stats/; quit" | matlab -nojvm -nosplash -nodisplay
cd ..

cd libDPM
echo "compile;  quit" | matlab -nojvm -nosplash -nodisplay
make;
echo "mcc -m partdetect_dpm.m -v -a ./ -a ../../scripts/matlab/  -N -p images/  -p stats/;  quit" | matlab -nojvm -nosplash -nodisplay
echo "mcc -m partdetect_dpm_all.m -v -a ./ -a ../../scripts/matlab/ -N -p images/  -p stats/; quit" | matlab -nojvm -nosplash -nodisplay
cd ..
