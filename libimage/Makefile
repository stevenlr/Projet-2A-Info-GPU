CC=gcc
CLIBFLAGS=-Wall -W -std=c99 -I./include
CTESTSFLAGS=-Wall -W -std=c99 -I./include
LDTESTSFLAGS=-L./lib -limage
AR=ar

OUTPUTLIB=lib/libimage.a
OUTPUTTEST=test_libimage.exe

SRCLIB=$(wildcard src/*.c)
HDRLIB=$(wildcard include/*.h)
OBJLIB=$(patsubst src/%.c, obj/lib/%.o, $(SRCLIB))
BASENAMELIB=$(patsubst src/%.c, %, $(SRCLIB))

SRCTEST=$(wildcard tests/*.c)
HDRTEST=$(wildcard tests/*.h)
OBJTEST=$(patsubst tests/%.c, obj/tests/%.o, $(SRCTEST))
BASENAMETEST=$(patsubst tests/%.c, %, $(SRCTEST))

.PHONY: all clean libimage test

all: libimage

libimage: depend.d lib obj/lib $(OUTPUTLIB)

$(OUTPUTLIB): $(OBJLIB)
	$(AR) rcs $@ $^

lib:
	mkdir $@

obj:
	mkdir $@

obj/lib: obj
	mkdir $@

obj/tests: obj
	mkdir $@

test: depend.d obj/tests $(OUTPUTLIB) $(OUTPUTTEST)

$(OUTPUTTEST): $(OBJTEST)
	gcc $^ -o $@ $(LDTESTSFLAGS)

obj/lib/%.o: src/%.c
	$(CC) $< -c -o $@ $(CLIBFLAGS)

obj/tests/%.o: tests/%.c
	$(CC) $< -c -o $@ $(CTESTSFLAGS)

depend.d: $(SRCLIB) $(SRCTEST) $(HDRLIB) $(HDRTEST)
	rm -f $@
	for f in $(BASENAMELIB); do $(CC) src/$$f.c $(CLIBFLAGS) -MM -E -MT obj/lib/$$f.o >> $@; done
	for f in $(BASENAMETEST); do $(CC) tests/$$f.c $(CTESTSFLAGS) -MM -E -MT obj/tests/$$f.o >> $@; done

clean:
	rm -rf obj
	rm -f depend.d

include depend.d