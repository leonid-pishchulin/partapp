#!/bin/bash

PATH_QT4=/usr/include/qt4/
echo "mex -v -I../ -I$PATH_QT4 -I../../../external_include/ features_sc.cpp FeatureGrid.cpp ../libKMA2/ImageContent/ImageContent.cpp ../libKMA2/descriptor/FeatureDescriptor.cpp ../libKMA2/gauss_iir/gauss_iir.cpp ../libKMA2/ShapeDescriptor.cpp ../libKMA2/descriptor/EdgeDetector.cpp descriptorGrid.cpp -lpng -lQtCore -o features;  quit" | matlab -nojvm -nosplash -nodisplay
