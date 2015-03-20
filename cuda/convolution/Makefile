BINNAME=convolution

include ../DefaultMakefile.inc

run: output-images $(BIN)
	$(BIN) ../../references/input-images/Lenna.tga ../../references/matrices/7x7gaussian.txt ../output-images/convolution-7x7gaussian-extend.tga
	$(BIN) ../../references/input-images/Lenna.tga ../../references/matrices/3x3sharpen.txt ../output-images/convolution-3x3sharpen-extend.tga

profile: output-profiling $(BIN)
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/econvolution-7x7gaussian-extend.txt $(BIN) ../../references/input-images/Lenna.tga ../../references/matrices/7x7gaussian.txt ../output-images/convolution-7x7gaussian-extend.tga
	nvprof $(NVPROFFLAGS) --log-file ../output-profiling/convolution-3x3sharpen-extend.txt $(BIN) ../../references/input-images/Lenna.tga ../../references/matrices/3x3sharpen.txt ../output-images/convolution-3x3sharpen-extend.tga
	