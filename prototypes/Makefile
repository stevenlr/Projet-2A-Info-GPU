CXX=g++
LDFLAGS=-lpandore -LC:\pandore\lib
CXXFLAGS=-IC:\pandore\include
SRC=$(wildcard src/*.cpp)
BINS=$(patsubst src/%.cpp, bin/%.exe, $(SRC))

.PHONY: all clean

all: bin Lenna.pan DS.pan $(BINS)

bin:
	mkdir bin

bin/%.exe: src/%.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

%.pan: %.png
	pany2pan $^ $@

clean:
	rm -f $(BINS)
