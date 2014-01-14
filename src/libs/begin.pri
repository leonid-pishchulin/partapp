TEMPLATE = lib
VERSION = 1.0.1

CONFIG += release
CONFIG += dll

CONFIG(release) {
  OBJECTS_DIR = ./Release/
  DESTDIR = ../../../lib/Release/	
} else {
  OBJECTS_DIR = ./Debug/
  DESTDIR = ../../../lib/Debug/
}

INCLUDEPATH += ..
DEPENDPATH += ..

