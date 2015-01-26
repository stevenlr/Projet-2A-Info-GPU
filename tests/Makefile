CC=gcc
CFLAGS=-I../libimage/include -Iinclude -W -Wall -std=c99
LDFLAGS=-L../libimage/lib -limage
BIN=bin/tests.exe
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c, obj/%.o, $(SRC))
BASENAME=$(patsubst src/%.c, %, $(SRC))
REF_IMAGE=$(wildcard ../references/output-images/*.tga)
BASENAME_IMAGE=$(patsubst ../references/output-images/%.tga, %, $(REF_IMAGE))
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

depend.d: $(SRC) $(HDR)
	rm -f $@
	for f in $(BASENAME); do $(CC) src/$$f.c $(CFLAGS) -MM -E -MT obj/$$f.o >> $@; done

run_gpp: $(BIN)
	for f in $(BASENAME_IMAGE); do echo "[$$f] :"; $(BIN) ../references/output-images/$$f.tga ../gpp/output-images/$$f.tga ; done

run_sse: $(BIN)
	for f in $(BASENAME_IMAGE); do echo "[$$f] :"; $(BIN) ../references/output-images/$$f.tga ../gpp-sse/output-images/$$f.tga ; done

run_tbb: $(BIN)
	for f in $(BASENAME_IMAGE); do echo "[$$f] :"; $(BIN) ../references/output-images/$$f.tga ../gpp-tbb/output-images/$$f.tga ; done
include depend.d
