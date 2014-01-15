#!/bin/bash

cd libPrediction
echo "mcc -m trainClass.m -v -a ./ -a ./sparseLDA_v2 -a ../../scripts/matlab/ -N -p images/  -p stats/ ; quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
echo "mcc -m predictFactorsImg.m -v -a ./ -a ./sparseLDA_v2 -a ../../scripts/matlab/ -N -p images/  -p stats/ ; quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
cd ..

cd libDPM
echo "compile;  quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
echo "mcc -m partdetect_dpm.m -v -a ./ -a ../../scripts/matlab/  -N -p images/  -p stats/; quit" | /usr/bin/matlab -nojvm -nosplash -nodisp
echo "mcc -m partdetect_dpm_all.m -v -a ./ -a ../../scripts/matlab/ -N -p images/  -p stats/; quit" | /usr/bin/matlab -nojvm -nosplash -nodisplay
cd ..
