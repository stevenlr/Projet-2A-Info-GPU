pinverse Lenna.pan ref-invert.pan
padd Lenna.pan DS.pan ref-add.pan
pbinarization 127 255 Lenna.pan ref-threshold.pan
perosion 1 3 binary.pan ref-erosion-binary.pan
perosion 1 1 Lenna.pan ref-erosion-colors.pan
pconvolution matrices/7x7gaussian.txt Lenna.pan ref-convolution-7x7gaussian.pan
pconvolution matrices/3x3sharpen.txt Lenna.pan ref-convolution-3x3sharpen.pan