proto-invert.exe Lenna.pan out-invert.pan
proto-add.exe Lenna.pan DS.pan out-add.pan
proto-threshold.exe 127 Lenna.pan out-threshold.pan
proto-erosion.exe 3 binary.pan out-erosion-binary.pan
proto-erosion.exe 1 Lenna.pan out-erosion-colors.pan