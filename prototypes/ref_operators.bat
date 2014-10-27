pinverse src-Lenna.pan ref-invert.pan
padd src-Lenna.pan src-DS.pan ref-add.pan
pbinarization 127 255 src-Lenna.pan ref-threshold.pan
perosion 1 3 src-binary.pan ref-erosion-binary.pan
perosion 1 1 src-Lenna.pan ref-erosion-colors.pan
pconvolution matrices/7x7gaussian.txt src-Lenna.pan ref-convolution-7x7gaussian.pan
pconvolution matrices/3x3sharpen.txt src-Lenna.pan ref-convolution-3x3sharpen.pan