include ( ../begin.pri )

TARGET = PartApp

###################################################
# HEADERS/SOURCES
###################################################

PB_HEADERS = ExpParam.pb.h 
PB_SOURCES = ExpParam.pb.cc

HEADERS = partapp.h
SOURCES = partapp.cpp

HEADERS += $$PB_HEADERS
SOURCES += $$PB_SOURCES

###################################################
# INCLUDEPATH
###################################################

# Matlab
include( ../../matlab_include.pri )

include ( ../../protobuf.pri)


