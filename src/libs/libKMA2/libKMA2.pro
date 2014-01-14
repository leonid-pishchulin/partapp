include ( ../begin.pri )

TARGET = KMA2

###################################################
# HEADERS/SOURCES
###################################################

HEADERS = descriptor/feature.h ./ImageContent/imageContent.h gauss_iir/gauss_iir.h ShapeDescriptor.h kmaimagecontent.h ShapeDescriptor.h
SOURCES = ImageContent/ImageContent.cpp descriptor/FeatureDescriptor.cpp descriptor/EdgeDetector.cpp gauss_iir/gauss_iir.cpp ShapeDescriptor.cpp
