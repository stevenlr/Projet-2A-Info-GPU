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

void add(int argc, char *argv[])
{
	if (argc != 3) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image1;
	Image *input_image2;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[0], &input_image1)) != 0) {
		cout << "Error when opening image: " << error << endl;
		return;
	}

	if ((error = TGA_readImage(argv[1], &input_image2)) != 0) {
		cout << "Error when opening image: " << error << endl;
		Image_delete(input_image2);
		return;
	}

	if (input_image1->height != input_image2->height ||
		input_image1->width != input_image2->width ||
		input_image1->channels != input_image2->channels) {
		cout << "Error : Input images should be the same size." << endl;
		Image_delete(input_image1);
		Image_delete(input_image2);
		return;
	}

	if ((error = Image_new(input_image1->width,
			       input_image1->height,
			       input_image1->channels,
			       &output_image)) != 0) {
		cout << "Error when creating output image : " << error << endl;
		Image_delete(input_image1);
		Image_delete(input_image2);
		return;
	}

	int size, c;
	uint8_t *datao, *data1, *data2;
	uint16_t current_data;

	size = input_image1->width * input_image1->height;

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for (0, input_image1->channels, [&](int i) {
		datao = output_image->data[c];
		data1 = input_image1->data[c];
		data2 = input_image2->data[c];

		for (i = 0; i < size; ++i) {
			current_data = (*data1++) + (*data2++);
			*datao++ = (current_data > 0xff) ? 0xff : current_data;
		}
	} );

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error;
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);
}
