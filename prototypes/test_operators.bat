proto-invert.exe Lenna.pan out-invert.pan
proto-add.exe Lenna.pan DS.pan out-add.pan
proto-threshold.exe 127 Lenna.pan out-threshold.pan
proto-erosion.exe 3 binary.pan out-erosion-binary.pan
proto-erosion.exe 1 Lenna.pan out-erosion-colors.pan
proto-convolution.exe matrices/7x7gaussian.txt 0 Lenna.pan out-convolution-7x7gaussian-warp.pan
proto-convolution.exe matrices/7x7gaussian.txt 1 Lenna.pan out-convolution-7x7gaussian-extend.pan
proto-convolution.exe matrices/7x7gaussian.txt 2 Lenna.pan out-convolution-7x7gaussian-crop.pan
proto-convolution.exe matrices/3x3sharpen.txt 1 Lenna.pan out-convolution-3x3sharpen-extend.pan