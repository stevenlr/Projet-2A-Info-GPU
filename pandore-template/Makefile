CXX=g++
BIN=test
LDFLAGS=-lpandore -LC:\pandore\lib
CXXFLAGS=-IC:\pandore\include

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
