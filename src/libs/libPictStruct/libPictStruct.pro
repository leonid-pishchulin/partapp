include ( ../begin.pri )

TARGET = PictStruct

###################################################
# HEADERS/SOURCES
###################################################

PB_HEADERS = HypothesisList.pb.h 
PB_SOURCES = HypothesisList.pb.cc

HEADERS = objectdetect.h objectdetect_sample.h 
SOURCES = objectdetect_aux.cpp objectdetect_learnparam.cpp objectdetect_findpos.cpp objectdetect_findrot.cpp objectdetect_roi.cpp objectdetect_sample.cpp objectdetect_icps.cpp Timer.cpp

HEADERS += $$PB_HEADERS
SOURCES += $$PB_SOURCES

###################################################
# INCLUDEPATH
###################################################

# protocol buffers
INCLUDEPATH += ../../../include_pb

# Matlab
include( ../../matlab_include.pri )

include ( ../../protobuf.pri)


