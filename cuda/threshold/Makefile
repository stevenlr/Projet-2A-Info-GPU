BINNAME=threshold

include ../DefaultMakefile.inc

run: output-images $(BIN)
	$(BIN) ../../references/input-images/Lenna.tga 127 ../output-images/threshold.tga

profile: output-profiling $(BIN)
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/threshold.txt $(BIN) ../../references/input-images/Lenna.tga 127 ../output-images/threshold.tga