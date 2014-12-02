/**
 * @file invert.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <image/image.h>
#include <image/tga.h>

#include "invert.h"
#include "benchmark.h"

void invert(int argc, char *argv[])
{
	if (argc != 2) {
		printf("Invalid number of arguments.\n");
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		printf("Error when copying image: %d\n", error);
		Image_delete(input_image);
		return;
	}

	int c, size;
	Benchmark bench;
	start_benchmark(&bench);

	size = input_image->width * input_image->height;

	for (c = 0; c < input_image->channels; ++c) {
		int i;
		uint8_t *data;

		data = output_image->data[c];

		for (i = 0; i < size; ++i)
			*data++ ^= 0xff;
	}

	end_benchmark(&bench);
	printf("%lu ", bench.elapsed_ticks);
	printf("%lf\n", bench.elapsed_time);

	if ((error = TGA_writeImage(argv[1], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);
}