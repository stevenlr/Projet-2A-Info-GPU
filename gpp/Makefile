CC=gcc
CFLAGS=-I../libimage/include -Iinclude -W -Wall -std=c99
LDFLAGS=-L../libimage/lib -limage
BIN=bin/filters-gpp.exe
#SRC=$(wildcard src/*.c)
SRC=src/main.c src/benchmark.c src/invert.c src/threshold.c src/add.c src/erosion.c
OBJ=$(patsubst src/%.c, obj/%.o, $(SRC))
BASENAME=$(patsubst src/%.c, %, $(SRC))
HDR=$(wildcard include/*.h)

.PHONY: all clean cleanall run

all: $(BIN)

obj:
	mkdir obj

bin:
	mkdir bin

$(BIN): bin obj $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

obj/%.o: src/%.c
	$(CC) $< -o $@ -c $(CFLAGS)

clean:
	rm -f depend.d
	rm -rf obj

cleanall: clean
	rm -rf bin
	rm -rf output-images

output-images:
	mkdir output-images

run: output-images $(BIN)
	$(BIN) invert ../references/input-images/Lenna.tga output-images/invert.tga
	$(BIN) threshold ../references/input-images/Lenna.tga 127 output-images/threshold.tga
	$(BIN) add ../references/input-images/Lenna.tga ../references/input-images/DS.tga output-images/add.tga
	$(BIN) erosion ../references/input-images/binary.tga 3 output-images/erosion-binary.tga
	$(BIN) erosion ../references/input-images/Lenna.tga 1 output-images/erosion-colors.tga
	$(BIN) convolution ../references/input-images/Lenna.tga ../references/matrices/7x7gaussian.txt output-images/convolution-7x7gaussian-extend.tga
	$(BIN) convolution ../references/input-images/Lenna.tga ../references/matrices/3x3sharpen.txt output-images/convolution-3x3sharpen-extend.tga

depend.d: $(SRC) $(HDR)
	rm -f $@
	for f in $(BASENAME); do $(CC) src/$$f.c $(CFLAGS) -MM -E -MT obj/$$f.o >> $@; done

include depend.d
