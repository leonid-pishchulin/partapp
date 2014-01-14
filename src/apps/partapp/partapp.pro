TARGET = partapp

###################################################
# CONFIG
###################################################

CONFIG = uic resources qt incremental stl qt_no_framework warn_on
CONFIG += release

CONFIG(release) {
  OBJECTS_DIR = ./Release/
  DESTDIR = ../../../bin/Release/	
} else {
  OBJECTS_DIR = ./Debug/
  DESTDIR = ../../../bin/Debug/
}

message($$CONFIG)	

CONFIG(debug) {
  message(Debug build)
}
CONFIG(release) {
  message(Release build)
}


###################################################
# HEADERS/SOURCES
###################################################

#HEADERS = parteval.h 
SOURCES = main.cpp  


###################################################
# INCLUDEPATH
###################################################


INCLUDEPATH += ../../libs
DEPENDPATH += ../../libs

# Matlab
include( ../../matlab_include.pri )

# Protocol Buffers
INCLUDEPATH += ../../../include_pb

###################################################
# LIBS
###################################################

CONFIG(release) {
LIBS = -L../../../lib/Release/
LIBS += -L../../../lib_pb
}
else {
  LIBS = -L../../../lib/Debug/	
  LIBS += -L../../../lib_pb
}

LIBS += -lMatlabIO -lBoostMath -lFilesystemAux -lPartDetect -lPartApp -lPictStruct -lKMA2 -lDiscPS -lPartEval 

# boost 
LIBS += -lboost_program_options

# protocol buffers
LIBS += -lprotobuf 

# TUDVision
LIBS += -lAnnotation -lAdaBoost

# libDAI
include( ../../libdai.pri )

# blas 
LIBS += -lblas -lm

# needed to correctly handle dependencies on protocol buffers generated files
include ( ../../protobuf.pri)
