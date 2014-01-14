include ( ../begin.pri )

TARGET = MatlabIO

HEADERS = matlab_io.h 
SOURCES = matlab_io.cpp

include( ../../matlab_include.pri )

LIBS += -L../../../lib_mat
LIBS += -lmx -lmat

