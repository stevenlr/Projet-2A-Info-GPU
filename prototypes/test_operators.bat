proto-invert.exe src-Lenna.pan out-invert.pan
proto-add.exe src-Lenna.pan src-DS.pan out-add.pan
proto-threshold.exe 127 src-Lenna.pan out-threshold.pan
proto-erosion.exe 3 src-binary.pan out-erosion-binary.pan
proto-erosion.exe 1 src-Lenna.pan out-erosion-colors.pan
proto-convolution.exe matrices/7x7gaussian.txt 0 src-Lenna.pan out-convolution-7x7gaussian-warp.pan
proto-convolution.exe matrices/7x7gaussian.txt 1 src-Lenna.pan out-convolution-7x7gaussian-extend.pan
proto-convolution.exe matrices/7x7gaussian.txt 2 src-Lenna.pan out-convolution-7x7gaussian-crop.pan
proto-convolution.exe matrices/3x3sharpen.txt 1 src-Lenna.pan out-convolution-3x3sharpen-extend.pan