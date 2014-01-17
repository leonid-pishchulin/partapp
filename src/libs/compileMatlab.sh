#!/bin/bash

cd ../../
EXT_MAT_DIR=$PWD/external_mat
cd - 
MATLABPATH=$EXT_MAT_DIR/sparseLDA_v2/:$EXT_MAT_DIR/larsen/:${MATLABPATH}
#echo $MATLABPATH

cd libPrediction
#echo "mcc -m trainClass.m -v -a ./ -a " $EXT_MAT_DIR/sparseLDA_v2/  " -a " $EXT_MAT_DIR/larsen/  " -a ../../scripts/matlab/ -N -p images/  -p stats/ ; quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
#echo "mcc -m predictFactorsImg.m -v -a ./ -N -p images/  -p stats/ ; quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
cd ..

cd libDPM
echo "compile;  quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
#echo "mcc -m partdetect_dpm.m -v -a ./ -a ../../scripts/matlab/  -N -p images/  -p stats/; quit" | /usr/bin/matlab -nojvm -nosplash -nodisp
#echo "mcc -m partdetect_dpm_all.m -v -a ./ -a ../../scripts/matlab/ -N -p images/  -p stats/; quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
make
cd ..
