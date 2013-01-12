MOC_DIR = .moc
OBJECTS_DIR = .obj

TEMPLATE = app
SOURCES = trackingBall.cpp


CONFIG += qt warn_on

INCLUDEPATH += /user/local/include/opencv2
LIBS += -lopencv_core\
        -lopencv_highgui\
        -lopencv_video\
