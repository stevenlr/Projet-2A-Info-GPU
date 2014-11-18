CC=gcc
CFLAGS=-I../libimage/include -Iinclude -W -Wall -std=c99
LDFLAGS=-L../libimage/lib -limage
BIN=bin/tests.exe
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c, obj/%.o, $(SRC))
BASENAME=$(patsubst, src/%.c, %, $(SRC))
HDR=$(wildcard include/*.h)

.PHONY: all clean cleanall

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

include depend.d