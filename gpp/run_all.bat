bin\filters-gpp.exe invert ..\references\input-images\Lenna.tga invert.tga
bin\filters-gpp.exe threshold ..\references\input-images\Lenna.tga 127 threshold.tga
bin\filters-gpp.exe add ..\references\input-images\Lenna.tga ..\references\input-images\DS.tga add.tga
bin\filters-gpp.exe erosion ..\references\input-images\binary.tga 1 erosion-binary.tga
bin\filters-gpp.exe erosion ..\references\input-images\Lenna.tga 1 erosion-colors.tga
bin\filters-gpp.exe convolution ..\references\input-images\Lenna.tga ..\references\matrices\7x7gaussian.txt blur.tga
bin\filters-gpp.exe convolution ..\references\input-images\Lenna.tga ..\references\matrices\3x3sharpen.txt sharpen.tga