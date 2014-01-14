include ( ../begin.pri )

TARGET = PartDetect

###################################################
# HEADERS/SOURCES
###################################################

PB_HEADERS = PartConfig.pb.h PartWindowParam.pb.h AbcDetectorParam.pb.h 
PB_SOURCES = PartConfig.pb.cc PartWindowParam.pb.cc AbcDetectorParam.pb.cc

HEADERS = partdef.h partdetect.h FeatureGrid.h
SOURCES = partdef.cpp partdetect_aux.cpp FeatureGrid.cpp partdetect_train.cpp partdetect_test.cpp partdetect_icps.cpp

HEADERS += $$PB_HEADERS
SOURCES += $$PB_SOURCES

LIBS += $$OBJECTS_DIR/kmeans.o 

###################################################
# INCLUDEPATH
###################################################

# protocol buffers
INCLUDEPATH += ../../../include_pb

# Matlab
include( ../../matlab_include.pri )

include ( ../../protobuf.pri)

QMAKE_PRE_LINK  = ./make-kmeans.sh
