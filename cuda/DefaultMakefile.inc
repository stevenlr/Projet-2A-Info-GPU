CC=nvcc
NVCCFLAGS=-ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -m32
CFLAGS=-I../../libimage/include -O3
LDFLAGS=-L../../libimage-vc -limage
BIN=bin/$(BINNAME).exe
SRC=$(wildcard src/*.cu)
BASENAME=$(patsubst src/%.c, %, $(SRC))
RM=rm
MKDIR=mkdir

.PHONY: all clean cleanall run

all: $(BIN)

bin:
	$(MKDIR) -p bin

$(BIN): bin $(SRC)
	$(CC) $(SRC) -o $@ $(LDFLAGS) $(CFLAGS) $(NVCCFLAGS)

clean:
	$(RM) -rf obj

cleanall: clean
	$(RM) -rf bin

output-images:
	$(MKDIR) -p ../output-images

output-profiling:
	$(MKDIR) -p ../output-profiling