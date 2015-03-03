CC=nvcc
NVCCFLAGS=-ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -m32
CFLAGS=-I../../libimage/include -O3
LDFLAGS=-L../../libimage-vc -limage
BIN=bin/invert.exe
SRC=$(wildcard src/*.cu)
BASENAME=$(patsubst src/%.c, %, $(SRC))
RM=del
MKDIR=mkdir

.PHONY: all clean cleanall run

all: $(BIN)

bin:
	mkdir bin

$(BIN): bin $(SRC)
	$(CC) $(SRC) -o $@ $(LDFLAGS) $(CFLAGS) $(NVCCFLAGS)

clean:
	$(RM) -f depend.d
	$(RM) -rf obj

cleanall: clean
	$(RM) -rf bin
	$(RM) -rf output-images

output-images:
	$(MKDIR) output-images

run: output-images $(BIN)
	$(BIN) ../../references/input-images/Lenna.tga output-images/invert.tga

