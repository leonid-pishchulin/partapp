include ( ../begin.pri )

TARGET = Nms

###################################################
# HEADERS/SOURCES
###################################################

HEADERS = rect_intersection.h nms.h
SOURCES = rect_intersection.cpp nms.cpp

#HEADERS += $$PB_HEADERS
#SOURCES += $$PB_SOURCES

# TUDVision
#include( ../../tudvision_include.pri )

#include( ../../matlab_include.pri )

# protocol buffers 
INCLUDEPATH += ../../../include_pb
include ( ../../protobuf.pri)

#DEFINES += DAI_WITH_BP
