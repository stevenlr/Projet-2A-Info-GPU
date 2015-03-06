BINNAME=invert

include ../DefaultMakefile.inc

run: output-images $(BIN)
	$(BIN) ../../references/input-images/Lenna.tga ../output-images/invert.tga

profile: output-profiling $(BIN)
	nvprof --print-gpu-trace --log-file ../output-profiling/invert.txt $(BIN) ../../references/input-images/Lenna.tga ../output-images/invert.tga