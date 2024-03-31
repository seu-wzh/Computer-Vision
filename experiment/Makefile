cc := g++

srcdir := src
bindir := bin
builddir := build
includedir := include /usr/include/opencv4

cppflags := -std=c++11 -Wall $(foreach dir,$(includedir),-I$(dir))

ifeq ("$(debug)", "yes")
	cppflags += -g
endif

ifeq ("$(method)", "customize")
	cppflags += -DCUSTOMIZE
endif

ifeq ("$(verbose)", "yes")
	hide :=
else
	hide := @
endif

target := canny_detection
source := bilateral.cpp canny.cpp main.cpp
object := $(source:.cpp=.o)
library := opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_videoio yaml-cpp

target := $(addprefix $(bindir)/,$(target))
source := $(addprefix $(srcdir)/,$(source))
object := $(addprefix $(builddir)/,$(object))
linkflags := -L/usr/local/lib $(foreach lib,$(library),-l$(lib))

.PHONY:all clean

all: $(target)

$(target): $(object)
	$(hide)$(cc) $^ $(linkflags) -o $@

$(builddir)/%.o: $(srcdir)/%.cpp
	$(hide)$(cc) $(cppflags) -c $< -o $@

clean:
	$(hide)rm $(bindir)/* $(builddir)/*