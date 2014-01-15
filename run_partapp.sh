#!/bin/sh

PARTAPP_DIR=`dirname $0`
LD_LIBRARY_PATH=$PARTAPP_DIR/lib/Release:$PARTAPP_DIR/lib_pb:$PARTAPP_DIR/lib_mat
PATH=$PATH:$PARTAPP_DIR/src/libs/libPrediction:$PARTAPP_DIR/src/libs/libDPM
MATLABPATH=${MATLABPATH}:$PARTAPP_DIR/src/libs/libPrediction/sparseLDA_v2/:$PARTAPP_DIR/src/libs/libPrediction/larsen/
if [ $# -lt 1 ]
then
    $PARTAPP_DIR/bin/Release/partapp --help
else
    $PARTAPP_DIR/bin/Release/partapp $@
fi
