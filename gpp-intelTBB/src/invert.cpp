/**
 * @file invert.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>

#include "invert.h"
#include "benchmark.h"

using namespace std;
using namespace tbb;
class InvertParallel
{
public:
	InvertParallel(Image *output_image, int nbParts) :
		output_image(output_image), nbParts(nbParts)
 	{ }

	void operator()(int p) const
	{
		uint8_t *datao;
		int c, size, i, partSize, beginPart;
		size = output_image->width * output_image->height;
		partSize = size/nbParts;
		beginPart = partSize * p;

		for (c = 0; c < output_image->channels; ++c) {
			datao = output_image->data[c] + beginPart;

			for (i = 0; i < partSize; ++i) {
				*datao++ ^= 0xff;
			}
		}
	}

private:
	Image *output_image;
	int nbParts;
};

void invert(int argc, char *argv[])
{
	if (argc != 2) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		cout << "Error when opening image: " << error << endl;
		return;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		cout << "Error when copying image: " << error << endl;
		Image_delete(input_image);
		return;
	}

	int nbParts = 8; 

	InvertParallel invertParallel(output_image, nbParts);

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, nbParts, invertParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[1], output_image)) != 0) {
		cout << "Error when writing image: " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}
