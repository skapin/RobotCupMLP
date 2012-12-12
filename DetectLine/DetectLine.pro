MOC_DIR = .moc
OBJECTS_DIR = .obj
TEMPLATE = app
SOURCES = main.cpp
CONFIG += qt \
    warn_on
INCLUDEPATH += /usr/local/include/opencv2
LIBS += -lopencv_core \
    -lopencv_highgui \
    -lopencv_features2d \
    -lopencv_video
OTHER_FILES += readme.txt