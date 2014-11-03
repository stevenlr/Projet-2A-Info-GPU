CC=gcc
CFLAGS=-Wall -W -std=c99 -I./include
LDFLAGS=
AR=ar

OUTPUTLIB=lib/libimage.a
OUTPUTTEST=test_libimage.exe

SRC=$(wildcard src/*.c)
HDR=$(wildcard include/*.c)
OBJ=$(patsubst src/%.c, obj/%.o, $(SRC))
BASENAME=$(patsubst src/%.c, %, $(SRC))

SRCTEST=$(wildcard src/test_*.c)
HDRTEST=$(wildcard include/test_*.c)
OBJTEST=$(patsubst src/%.c, obj/%.o, $(SRCTEST))
BASENAMETEST=$(patsubst src/%.c, %, $(SRCTEST))

SRCLIB=$(filter-out $(SRCTEST), $(SRC))
HDRLIB=$(filter-out $(HDRTEST), $(HDR))
OBJLIB=$(filter-out $(OBJTEST), $(OBJ))
BASENAMELIB=$(filter-out $(BASENAMETEST), $(BASENAME))

.PHONY: all clean libimage test

all: libimage

libimage: depend.d lib obj $(OUTPUTLIB)

$(OUTPUTLIB): $(OBJLIB)
	$(AR) rcs $@ $^

lib:
	mkdir lib

obj:
	mkdir obj

test: depend.d obj $(OUTPUTLIB) $(OUTPUTTEST)

$(OUTPUTTEST): $(OBJTEST)
	gcc $^ -o $@ $(LDFLAGS) -L./lib -limage

obj/%.o: src/%.c
	$(CC) $< -c -o $@ $(CFLAGS)

depend.d: $(SRC) $(HDRS)
	rm -f $@
	for f in $(BASENAME); do $(CC) src/$$f.c $(CFLAGS) -MM -E -MT obj/$$f.o >> $@; done

clean:
	rm -rf obj
	rm -f depend.d

include depend.d