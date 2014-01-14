include ( ../begin.pri )

TARGET = DiscPS

###################################################
# HEADERS/SOURCES
###################################################

PB_HEADERS = FactorDefs.pb.h
PB_SOURCES = FactorDefs.pb.cc

HEADERS = disc_ps.h factors.h rect_intersection.h disc_sample.hpp unique_vect.h
SOURCES = disc_ps.cpp disc_sample.cpp factors.cpp rect_intersection.cpp disc_sample_with_prior.cpp unique_vect.cpp

HEADERS += $$PB_HEADERS
SOURCES += $$PB_SOURCES

###################################################
# INCLUDEPATH
###################################################

# protocol buffers
INCLUDEPATH += ../../../include_pb

# libDAI 
include( ../../libdai.pri )

# Matlab
include( ../../matlab_include.pri )

include ( ../../protobuf.pri)


