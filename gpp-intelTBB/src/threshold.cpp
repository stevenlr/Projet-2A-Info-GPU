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
	ThresholdParallel(Image *output_image, uint8_t value) :
		output_image(output_image), value(value)
 	{ }

	void operator()(int i) const
	{
		for(int j = 0; j < output_image->channels; ++j) {
			output_image->data[j][i] = (output_image->data[j][i] >= value) ? 0xff : 0x00;
		}
	}

private:
	Image *output_image;
	uint8_t value;
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

	int size;
	ThresholdParallel thresholdParallel(output_image, value);
	size = input_image->width * input_image->height;

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, size, thresholdParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}