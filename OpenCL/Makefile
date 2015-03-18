CXX=g++
CXXFLAGS=-I../libimage/include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\NVIDIA GPU Computing SDK\OpenCL\common\inc" -Iinclude -W -Wall -O3 -std=c++11
LDFLAGS=-L../libimage/lib -limage -L"C:\Program Files\NVIDIA GPU Computing Toolkit\NVIDIA GPU Computing SDK\OpenCL\common\lib\Win32" -lOpenCL
BIN=bin/filters_opencl.exe
SRC=$(wildcard src/*.cpp)
OBJ=$(patsubst src/%.cpp, obj/%.o, $(SRC))
BASENAME=$(patsubst src/%.cpp, %, $(SRC))
HDR=$(wildcard include/*.h)

.PHONY: all clean cleanall run

all: $(BIN)

obj:
	mkdir obj

bin:
	mkdir bin

$(BIN): bin obj $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

obj/%.o: src/%.cpp
	$(CXX) $< -o $@ -c $(CXXFLAGS)

clean:
	rm -f depend.d
	rm -rf obj

cleanall: clean
	rm -rf $(BIN)
	rm -rf output-images

output-images:
	mkdir output-images

run: run_nvidia run_intel

run_nvidia: output-images $(BIN)
	$(BIN) invert NVIDIA ../references/input-images/Lenna.tga output-images/invert.tga
	$(BIN) threshold NVIDIA ../references/input-images/Lenna.tga 127 output-images/threshold.tga
	$(BIN) add NVIDIA ../references/input-images/Lenna.tga ../references/input-images/DS.tga output-images/add.tga
	$(BIN) erosion NVIDIA ../references/input-images/binary.tga 3 3 output-images/erosion-binary.tga
	$(BIN) erosion NVIDIA ../references/input-images/binary.tga 10 10 output-images/erosion-binary-10.tga
	$(BIN) erosion NVIDIA ../references/input-images/Lenna.tga 1 1 output-images/erosion-colors.tga

run_intel: output-images $(BIN)
	$(BIN) invert Intel ../references/input-images/Lenna.tga output-images/invert.tga
	$(BIN) threshold Intel ../references/input-images/Lenna.tga 127 output-images/threshold.tga
	$(BIN) add Intel ../references/input-images/Lenna.tga ../references/input-images/DS.tga output-images/add.tga
	$(BIN) erosion Intel ../references/input-images/binary.tga 3 3 output-images/erosion-binary.tga
	$(BIN) erosion Intel ../references/input-images/binary.tga 10 10 output-images/erosion-binary-10.tga
	$(BIN) erosion Intel ../references/input-images/Lenna.tga 1 1 output-images/erosion-colors.tga

run_future:
	$(BIN) convolution ../references/input-images/Lenna.tga ../references/matrices/7x7gaussian.txt output-images/convolution-7x7gaussian-extend.tga
	$(BIN) convolution ../references/input-images/Lenna.tga ../references/matrices/3x3sharpen.txt output-images/convolution-3x3sharpen-extend.tga

depend.d: $(SRC) $(HDR)
	rm -f $@
	for f in $(BASENAME); do $(CXX) src/$$f.cpp $(CXXFLAGS) -MM -E -MT obj/$$f.o >> $@; done

include depend.d
