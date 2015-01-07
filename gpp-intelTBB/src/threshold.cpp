/**
 * @file threshold.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>


#include "threshold.h"
#include "benchmark.h"

using namespace std;
using namespace tbb;

class ThresholdParallel 
{
public:
	ThresholdParallel(Image *output_image, uint8_t value, int nbParts) :
		output_image(output_image), value(value), nbParts(nbParts)
 	{ 
 		size = output_image->width * output_image->height;
		partSize = size/nbParts;
 	}

	void operator()(int p) const
	{
		int beginPart;
		uint8_t *out_data;

		beginPart = partSize * p;

		for (int c = 0; c < output_image->channels; ++c) {
			out_data = output_image->data[c] + beginPart;

			for (int i = 0; i < partSize; ++i) {
				*out_data = (*out_data >= value) ? 0xff : 0x00;
				out_data++;
			}
		}
	}

private:
	Image *output_image;
	uint8_t value;
	int nbParts;
	int size;
	int partSize;
};

void threshold(int argc, char *argv[])
{
	if (argc != 3) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;
	uint8_t value;

	value = (uint8_t) atoi(argv[1]);

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

	ThresholdParallel thresholdParallel(output_image, value, nbParts);

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, nbParts, thresholdParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}