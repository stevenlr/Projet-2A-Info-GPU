BINNAME=erosion

include ../DefaultMakefile.inc

run: output-images $(BIN)
	$(BIN) ../../references/input-images/binary.tga 3 3 ../output-images/erosion-binary.tga
	$(BIN) ../../references/input-images/binary.tga 10 10 ../output-images/erosion-binary-10.tga
	$(BIN) ../../references/input-images/Lenna.tga 1 1 ../output-images/erosion-colors.tga

profile: output-profiling $(BIN)
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/erosion-binary.txt $(BIN) ../../references/input-images/binary.tga 3 3 ../output-images/erosion-binary.tga
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/erosion-binary.txt $(BIN) ../../references/input-images/binary.tga 10 10 ../output-images/erosion-binary-10.tga
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/erosion-colors.txt $(BIN) ../../references/input-images/Lenna.tga 1 1 ../output-images/erosion-colors.tga
	