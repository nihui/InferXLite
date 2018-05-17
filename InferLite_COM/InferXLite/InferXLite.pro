TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.c \
    src/app.c \
    src/hash.c \
    src/inferxlite.c \
    src/interface.c \
    src/metafmt.c \
    src/pipe.c \
    src/backends/backend.c \
    src/backends/backend_impl.c \
    src/models/JDminiYOLOv2.c \
    src/models/MOBILENETV2.c \
    src/models/model_init.c \
    src/models/ShipSSD.c \
    src/models/SqueezeNet.c \
    src/models/YOLOtiny.c


#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../OpenBLAS/lib/ -lopenblas
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../OpenBLAS/lib/ -lopenblas
#else:unix: LIBS += -L$$PWD/../../OpenBLAS/lib/ -lopenblas

INCLUDEPATH += $$PWD/OpenBLAS/include
DEPENDPATH += $$PWD/OpenBLAS/include

INCLUDEPATH += $$PWD/include
INCLUDEPATH += $$PWD/include/models
INCLUDEPATH += $$PWD/include/backends

HEADERS += \
    OpenBLAS/include/cblas.h \
    OpenBLAS/include/f77blas.h \
    OpenBLAS/include/lapacke.h \
    OpenBLAS/include/lapacke_config.h \
    OpenBLAS/include/lapacke_mangling.h \
    OpenBLAS/include/lapacke_utils.h \
    OpenBLAS/include/openblas_config.h \
    include/app.h \
    include/dirent.h \
    include/hash.h \
    include/inferxlite.h \
    include/inferxlite_common.h \
    include/interface.h \
    include/metafmt.h \
    include/pipe.h \
    include/backends/backend.h \
    include/backends/backend_impl.h \
    include/models/JDminiYOLOv2.h \
    include/models/MOBILENET.h \
    include/models/MOBILENETV2.h \
    include/models/model_init.h \
    include/models/ShipSSD.h \
    include/models/SqueezeNet.h \
    include/models/YOLOtiny.h






win32:CONFIG(release, debug|release): LIBS += -L$$PWD/OpenBLAS/lib/ -lopenblas
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/OpenBLAS/lib/ -lopenblas
else:unix: LIBS += -L$$PWD/OpenBLAS/lib/ -lopenblas

INCLUDEPATH += $$PWD/OpenBLAS/include
DEPENDPATH += $$PWD/OpenBLAS/include
