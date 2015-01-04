/**
 * @file add.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>

#include "add.h"
#include "benchmark.h"

using namespace std;
using namespace tbb;

class InvertParallel 
{
public:
	InvertParallel(Image *output_image) :
		output_image(output_image)
 	{ }

	void operator()(int i) const
	{
		for(int j = 0; j < output_image->channels; ++j) {
			output_image->data[j][i] ^= 0xff;
		}
	}

private:
	Image *output_image;
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

	int size;
	InvertParallel invertParallel(output_image);
	Benchmark bench;
	start_benchmark(&bench);

	size = input_image->width * input_image->height;
	
	parallel_for(0, size, invertParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[1], output_image)) != 0) {
		cout << "Error when writing image: " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}
