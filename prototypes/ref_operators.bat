pinverse Lenna.pan ref-invert.pan
padd Lenna.pan DS.pan ref-add.pan
pbinarization 127 255 Lenna.pan ref-threshold.pan
perosion 1 3 binary.pan ref-erosion-binary.pan
perosion 1 1 Lenna.pan ref-erosion-colors.pan