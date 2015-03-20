TARGETS=invert threshold add erosion convolution

all:
	for t in $(TARGETS); do cd $$t && $(MAKE) && cd ..; done

run:
	for t in $(TARGETS); do cd $$t && $(MAKE) run && cd ..; done

profile:
	for t in $(TARGETS); do cd $$t && $(MAKE) profile && cd ..; done

clean:
	for t in $(TARGETS); do cd $$t && $(MAKE) clean && cd ..; done

cleanall:
	for t in $(TARGETS); do cd $$t && $(MAKE) cleanall && cd ..; done