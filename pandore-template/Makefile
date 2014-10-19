CXX=g++
BIN=test
LDFLAGS=-lpandore -L/usr/share/pandore/lib
CXXFLAGS=-I/usr/share/pandore/include

.PHONY: all clean

all: Lenna.pan $(BIN)

$(BIN): main.o
	$(CXX) $^ -o $(BIN) $(LDFLAGS)

Lenna.pan:
	pany2pan Lenna.png Lenna.pan

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS)

clean:
	rm -f *.o $(BIN)
