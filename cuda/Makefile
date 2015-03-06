TARGETS=invert threshold

all:
	for t in $(TARGETS); do cd $$t && $(MAKE) && cd ..; done

run:
	for t in $(TARGETS); do cd $$t && $(MAKE) run && cd ..; done

profile:
	for t in $(TARGETS); do cd $$t && $(MAKE) profile && cd ..; done