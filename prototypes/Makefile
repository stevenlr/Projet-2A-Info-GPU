CXX=g++
LDFLAGS=-lpandore -LC:\pandore\lib
CXXFLAGS=-IC:\pandore\include
SRC=$(wildcard src/*.cpp)
BINS=$(patsubst src/%.cpp, bin/%.exe, $(SRC))
IMGSPAN=$(wildcard *.pan)
IMGSPNG=$(patsubst %.pan, img/%.png, $(IMGSPAN))

.PHONY: all clean cleanpan convertimages

all: bin src-Lenna.pan src-DS.pan src-binary.pan $(BINS)

bin:
	mkdir bin

bin/%.exe: src/%.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

%.pan: %.png
	pany2pan $^ $@
#	prgb2gray 0.299 0.587 0.114 $@ $@

clean:
	rm -rf bin $(BINS)

cleanpan:
	rm -f *.pan

img:
	mkdir img

convertimages: img $(IMGSPNG)

img/%.png: %.pan
	pclipvalues 0 255 $^ tmp.pan
	pim2uc tmp.pan tmp.pan
	ppan2png tmp.pan $@
	rm tmp.pan