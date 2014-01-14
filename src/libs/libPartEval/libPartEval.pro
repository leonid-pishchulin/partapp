include ( ../begin.pri )

TARGET = PartEval

###################################################
# HEADERS/SOURCES
###################################################

HEADERS = parteval.h
SOURCES = parteval.cpp

###################################################
# INCLUDEPATH
###################################################

INCLUDEPATH += ../../libs
DEPENDPATH += ../../libs

# Matlab
include( ../../matlab_include.pri )

# Protocol Buffers
INCLUDEPATH += ../../../include_pb

# protocol buffers 
include ( ../../protobuf.pri)
