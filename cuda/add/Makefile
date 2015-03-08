BINNAME=add

include ../DefaultMakefile.inc

run: output-images $(BIN)
	$(BIN) ../../references/input-images/Lenna.tga ../../references/input-images/DS.tga ../output-images/add.tga

profile: output-profiling $(BIN)
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/add.txt $(BIN) ../../references/input-images/Lenna.tga ../../references/input-images/DS.tga ../output-images/add.tga