# Project 4: Calibration and Augmented Reality
# Shivang Patel (shivang2402)
CC = clang++
CXX = $(CC)
CFLAGS = -std=c++17 -Wall -I/opt/homebrew/opt/opencv/include/opencv4 -I../include
CXXFLAGS = $(CFLAGS)
LDFLAGS = -L/opt/homebrew/opt/opencv/lib
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_features2d

BINDIR = ../bin

calibar: main.o calibration.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

features: feature_detect.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~