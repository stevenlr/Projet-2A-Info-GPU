CC=gcc
CLIBFLAGS=-Wall -W -std=c99 -I./include
CTESTFLAGS=-Wall -W -std=c99 -I./include -I../unittest/include
LDTESTFLAGS=-L./lib -limage
AR=ar

OUTPUTLIB=lib/libimage.a
OUTPUTTEST=test_libimage.exe

SRCLIB=$(wildcard src/*.c)
HDRLIB=$(wildcard include/image/*.h)
OBJLIB=$(patsubst src/%.c, obj/lib/%.o, $(SRCLIB))
BASENAMELIB=$(patsubst src/%.c, %, $(SRCLIB))

SRCTEST=$(wildcard test/*.c)
HDRTEST=$(wildcard test/*.h)
OBJTEST=$(patsubst test/%.c, obj/test/%.o, $(SRCTEST))
BASENAMETEST=$(patsubst test/%.c, %, $(SRCTEST))

.PHONY: all clean cleanall libimage test

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

obj/test: obj
	mkdir $@

test: depend.d obj/test libimage $(OUTPUTTEST)

$(OUTPUTTEST): $(OBJTEST) $(OUTPUTLIB)
	gcc $(OBJTEST) -o $@ $(LDTESTFLAGS)

obj/lib/%.o: src/%.c
	$(CC) $< -c -o $@ $(CLIBFLAGS)

obj/test/%.o: test/%.c
	$(CC) $< -c -o $@ $(CTESTFLAGS)

depend.d: $(SRCLIB) $(SRCTEST) $(HDRLIB) $(HDRTEST)
	rm -f $@
	for f in $(BASENAMELIB); do $(CC) src/$$f.c $(CLIBFLAGS) -MM -E -MT obj/lib/$$f.o >> $@; done
	for f in $(BASENAMETEST); do $(CC) test/$$f.c $(CTESTFLAGS) -MM -E -MT obj/test/$$f.o >> $@; done

clean:
	rm -rf obj
	rm -f depend.d

cleanall: clean
	rm -rf lib
	rm -f $(OUTPUTTEST)

include depend.d